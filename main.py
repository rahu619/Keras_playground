import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.model_selection import train_test_split
import yaml

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, LSTM
from keras.models import Model
from keras.utils.vis_utils import plot_model

from pathlib import Path
import matplotlib.pyplot as plt

config_path = Path(__file__).with_name('./config.yaml')
config_dict = {}

from db import DbContext
from process_module import Process
from word_embedding import WordEmbedding

with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

processObj = Process()
dbContextObj = DbContext(config_dict)
wordembeddingObj = WordEmbedding(config_dict)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# Performing exploratory data analysis
# Training data 
claims_df = dbContextObj.get_claims()
print("Length of claims_df: {}".format(len(claims_df)))
print(claims_df.head(4))

# Training labels
dep_df = dbContextObj.get_dependencies()
# splitting labels into unique columns
dep_df = dep_df['dependency'].str.split(',', expand = True) 
print("Length of dep_df: {}".format(len(dep_df)))
dep_df = dep_df.shift(1, axis=1)
del dep_df[0]
dep_df.fillna(value=0, inplace=True)
dep_df = dep_df.astype(float)
print("Length of modified dep_df: {}".format(len(dep_df)))

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
    X.append(processObj.preprocess(sen))

y = dep_df.values

# Exploring data
# print("Training entries: {}, test entries: {}".format(len(X), len(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training entries: {}, test entries: {}".format(len(X_train), len(y_train)))
# print(X_test[:5])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_matrix = wordembeddingObj.getEmbeddingMatrix(vocab_size, tokenizer)

# Functional API Keras
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(22, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# Plotting architecture of our network
# plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

print("X_train samples : {}".format(X_train[2:]))
print("y_train samples : {}".format(y_train[2:]))

print("Length of X_train: {}".format(len(X_train)))
print("Length of y_train: {}".format(len(y_train)))

history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=0, validation_split=0.2)

# Evaluation
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# Plotting loss and accuracy values for training and test sets to see if our model is overfitting
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()