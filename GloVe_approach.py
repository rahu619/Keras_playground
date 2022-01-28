
# import nltk
# import nltk.data
# from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.tokenize import blankline_tokenize, LineTokenizer
# from nltk.probability import FreqDist
# from nltk.util import bigrams, trigrams, ngrams
# from nltk.stem import PorterStemmer
from nltk.stem import wordnet, WordNetLemmatizer
# from nltk import ne_chunk
# import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import re
import scipy
from pathlib import Path
from db import DbContext
import yaml
import os.path
import itertools
class DepClaim:
    def __init__(self, claim_no, dependency):
        self.claim_no = claim_no
        self.dependency = dependency
        
class Glove:
    config_path = None
    config_dict = {}
    log_substr_length = 100
    regex_exp = "^(.+?)[\,\.\)]"
    total_operation_flag = dependency_occurence_flag = 0
    word_lem = WordNetLemmatizer()
    model = None
    def __init__(self):        
        self.config_path = Path(__file__).with_name('./config.yaml')    
        self.glove_path = Path(__file__).parent.joinpath('training.data/glove.6B.50d.txt')
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)            
        self.dbContextObj = DbContext(config_dict)
        self.model = self.loadGloveModel(self.glove_path)
        
    def loadGloveModel(self, gloveFile):
        print ("Loading Glove Model")
        with open(gloveFile, encoding="utf8" ) as f:
            content = f.readlines()
        model = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print ("Done.",len(model)," words loaded!")
        return model

    def preprocess(self, raw_text):
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
        words = letters_only_text.lower().split()

        # removing stopwords and performing lemmatization
        stopword_set = set(stopwords.words("english"))
        cleaned_words = set([self.word_lem.lemmatize(w) for w in words if w not in stopword_set])
        
        # selecting the words that exist in the glove model alone
        return list(cleaned_words.intersection(self.model))


    def cosine_distance_wordembedding_method(self, s1, s2):
        print(s1)
        vector_1 = np.mean([self.model[word] for word in s1], axis = 0)
        vector_2 = np.mean([self.model[word] for word in s2], axis = 0)
        return scipy.spatial.distance.cosine(vector_1, vector_2)
   

    def similarity_between_two_sentences(self, s1, s2):
        print(f'Claim : {s1[:20]} and Claim: {s2[:20]}')
        s1 = self.preprocess(s1)
        s2 = self.preprocess(s2)
        cosine = self.cosine_distance_wordembedding_method(s1, s2)
        percentage = round((1-cosine) * 100, 2)
        return percentage
        
    def Evaluate(self):
        print('Starting .. ')
        os.system('clear')
        patents_df = self.dbContextObj.get_patent_ids()

        patents = patents_df['id'].tolist()
        print('Patent length : {}'.format(len(patents)))
        for patent in patents[:5]:
            print('Patent : {}'.format(patent))
            claims_list = []
            dependency_list = []
            dep_df = self.dbContextObj.get_dependencies_by_patent_id(patent)
            
            for patent_id, claim_id, dependency in dep_df.values.tolist():
               claims_df = self.dbContextObj.get_claims_by_id(claim_id)
               claim_text = claims_df['claim_text'][0]
               
               try:
                  claim_no = self.get_claim_number(claim_text)
               except:
                  continue
              
               claims_list.append(claim_text)
               dependency_list.append( DepClaim(claim_no, dependency) )
               
            print('Claims count : {}'.format(len(claims_list)))
            
            # possible_combinations = self.get_combinations(claims_list)      
            every_first_and_second = zip(claims_list[0::2], claims_list[1::2])         
            for first_text, second_text in every_first_and_second:        
                similarity_percentage = self.similarity_between_two_sentences(first_text, second_text)
                print('Word Embedding method with a cosine distance axes that our two sentences are similar to ', similarity_percentage,'%')

                try:
                    first_text_claim_no = self.get_claim_number(first_text)
                    second_text_claim_no = self.get_claim_number(second_text)
                except:
                    continue
                
                # If score above a certain point then
                if float(similarity_percentage) > 0.75:
                    print('Logging high similarity')
                    # check if second claim number has dependencies
                    dependency = next((x.dependency for x in dependency_list if x.claim_no == second_text_claim_no), None)
                    
                    if dependency:
                        # if exists: check if first claim number is amongst them
                        print('Logging dependency existence')
                        if dependency in first_text_claim_no:
                            print('Logging dependency match')
                            # print('Patent_id : {} & Claim_no : {} & Dependency : ({})'.format(patent, first_text_claim_no, dependency) )
                            self.dependency_occurence_flag += 1
  
                # else ignore
                self.total_operation_flag += 1
                
                print("{} \t\t {} \t\t Score: {:.4f}".format(first_text, second_text, similarity_percentage))
                
            
        print('Total operations : {}'.format(self.total_operation_flag))
        print('Similarity operations : {}'.format(self.dependency_occurence_flag))
            
            
    def get_claim_number(self, claim_text):
        text = re.search(self.regex_exp, claim_text)[0]
        return text.rsplit('.', 1)[0]
    
    def get_combinations(self, input_list):
        combination_indices = list(itertools.combinations(range(len(input_list)), 2))
        print(combination_indices)

Glove().Evaluate()
