import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, LSTM
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

from pathlib import Path
import matplotlib.pyplot as plt
from ast import literal_eval

from db import DbContext
from process_module import Process
from word_embedding import WordEmbedding

class Main:
    config_path = None
    config_dict = {}
    maxlen = 300
    vocab_size = -1
    trained_model_history = model = None
    trained_model_path = './models/claims_model.h5'
    X_train = X_test = y_train = y_test = [] 
    processObj = dbContextObj = wordembeddingObj = None
        
    def __init__(self):        
        self.__print_meta_data()    
        self.config_path = Path(__file__).with_name('./config.yaml')    
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        self.processObj = Process()
        self.dbContextObj = DbContext(config_dict)
        self.wordembeddingObj = WordEmbedding(config_dict)

    def __print_meta_data(self):
        print("Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print("Hub version: ", hub.__version__)
        print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


    def process_data(self, plot=False):
        # Performing exploratory data analysis
        # Training data 
        claims_df = self.dbContextObj.get_claims()
        print("Length of claims_df: {}".format(len(claims_df)))
        # print(claims_df.head(5))

        # Training labels
        dep_df = self.dbContextObj.get_dependencies()
        # splitting labels into unique columns
        # print(dep_df.head(5))
        dep_df = dep_df['dependency'].str.split(',', expand = True) 
        print("Length of dep_df: {}".format(len(dep_df)))
        dep_df = dep_df.shift(1, axis=1)
        del dep_df[0]
        dep_df.fillna(value=0, inplace=True)
        dep_df = dep_df.astype(float)

        if plot:
            print("Length of modified dep_df: {}".format(len(dep_df)))
            print(dep_df.head(5))

            # Analysing the dataset
            dep_df = dep_df.astype(float)
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 10
            fig_size[1] = 8
            plt.rcParams["figure.figsize"] = fig_size
            dep_df.sum(axis=0).plot.bar()
            plt.show()

        X = []
        sentences = list(claims_df['claim_text'])
        for sen in sentences:
            X.append(self.processObj.preprocess(sen))

        y = dep_df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Exploring data
        print("Training entries: {}, test entries: {}".format(len(self.X_train), len(self.y_train)))

    def tokenize(self):
        tokenizer_obj = Tokenizer(num_words=3000)
        tokenizer_obj.fit_on_texts(self.X_train)
        self.X_train = tokenizer_obj.texts_to_sequences(self.X_train)
        self.X_test = tokenizer_obj.texts_to_sequences(self.X_test)
        self.X_train = pad_sequences(self.X_train, padding='post', maxlen = self.maxlen)
        self.X_test = pad_sequences(self.X_test, padding='post', maxlen = self.maxlen)
            
        self.vocab_size = len(tokenizer_obj.word_index) + 1
        return tokenizer_obj
            
    def get_embedding_matrix(self): 
        """Generates Embedding Matrix"""
        tokenizer_obj = self.tokenize()
        return self.wordembeddingObj.getEmbeddingMatrix(self.vocab_size, tokenizer_obj)
            
    def __create_model(self, print_model_details = False):
        # Functional API Keras
        deep_input = Input(shape=(self.maxlen,))
        embedding_matrix = self.get_embedding_matrix()
        embedding_layer = Embedding(self.vocab_size, 300, weights=[embedding_matrix], trainable=False)(deep_input)
        LSTM_layer = LSTM(256)(embedding_layer)
        dense_layer = Dense(22, activation='sigmoid')(LSTM_layer)
        model = Model(inputs=deep_input, outputs=dense_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print('Model compilation done.')
        
        if print_model_details:
            print(model.summary())
            plot_model(model, to_file=' model_plot.png', show_shapes=True, show_layer_names=True)
            
        return model

    def load_model(self, is_load_model = False):
        """Model fitting"""
        if is_load_model:
            self.model = load_model(self.trained_model_path)
            return
        
        self.model = self.__create_model()
        # self.trained_model = model.fit(self.X_train, self.y_train, batch_size=256, epochs=5, verbose=1, validation_split=0.2)
        self.trained_model_history = self.model.fit(self.X_train, self.y_train, batch_size=400, epochs=2, verbose=1, validation_split=0.33)
        self.model.save(self.trained_model_path)
        print("Saving trained model")
            
            
    def evaluate_model(self):
        # Evaluation
        score = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        if self.trained_model_history:
            
            # Plotting loss and accuracy values for training and test sets to see if our model is overfitting
            plt.plot(self.trained_model_history.history['val_acc'])

            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train','test'], loc='upper left')
            plt.show()

            plt.plot(self.trained_model_history.history['loss'])
            plt.plot(self.trained_model_history.history['val_loss'])

            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train','test'], loc='upper left')
            plt.show()





obj = Main()
obj.process_data(plot=False)
obj.load_model()
obj.evaluate_model()