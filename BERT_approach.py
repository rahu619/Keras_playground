import os
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from db import DbContext
import yaml
from pathlib import Path
import numpy as np
import re

class DepClaim:
    def __init__(self, claim_no, dependency):
        self.claim_no = claim_no
        self.dependency = dependency
        
class BERT:
    config_path = None
    config_dict = {}
    log_substr_length = 100
    regex_exp = "^(.+?)[\,\.\)]"
    total_operation_flag = dependency_occurence_flag = 0
    def __init__(self):        
        self.config_path = Path(__file__).with_name('./config.yaml')    
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)            
        self.dbContextObj = DbContext(config_dict)
    
    def Evaluate(self):
        print('Starting .. ')
        os.system('clear')
        model = SentenceTransformer('all-mpnet-base-v2')
        patents_df = self.dbContextObj.get_patent_ids()
        
        for patent in patents_df['id'].tolist():
            print('Patent : {}'.format(patent))
            claims_list = []
            dependency_list = []
            dep_df = self.dbContextObj.get_dependencies_by_patent_id(patent)
            for patent_id, claim_id, dependency in dep_df.values.tolist():
               claims_df = self.dbContextObj.get_claims_by_id(claim_id)
               claim_text = claims_df['claim_text'][0]
               
               #print('Claim : {}'.format(claim_text))
               try:
                  claim_no = self.get_claim_number(claim_text)
               except:
                  continue
              
               claims_list.append(claim_text)
               dependency_list.append( DepClaim(claim_no, dependency) )
               
            print('Claims count : {}'.format(len(claims_list)))
               
            # for dep in dependency_list:
            #     print('Claim : {} and Dependency : {}'.format(dep.claim_no, dep.dependency))   
            # Compares the whole claim set and returns the top_k pair that has the highest cosine similarity score.
            # claim_texts = [x.claim for x in claims_list]
            
            # total operation counter
            # similarity counter
            
            paraphrases = util.paraphrase_mining(model, claims_list, top_k=1)
            for paraphrase in paraphrases:
                score, i, j = paraphrase
                first_text = claims_list[i][:self.log_substr_length]
                second_text = claims_list[j][:self.log_substr_length]
                
                try:
                    first_text_claim_no = self.get_claim_number(first_text)
                    second_text_claim_no = self.get_claim_number(second_text)
                except:
                    continue
                
                # If score above a certain point then
                if float(score) > 0.75:
                    print('Logging high similarity')
                    # check if second claim number has dependencies
                    dependency = next((x.dependency for x in dependency_list if x.claim_no == second_text_claim_no), None)
                    
                    if dependency:
                        # if exists: check if first claim number is amongst them
                        print('Logging dependency existence')
                        if dependency in first_text_claim_no:
                            print('Logging dependency match')
                            print('Patent_id : {} & Claim_no : {} & Dependency : ({})'.format(patent, first_text_claim_no, dependency) )
                            self.dependency_occurence_flag += 1
  
                # else ignore
                self.total_operation_flag += 1
                
                print("{} \t\t {} \t\t Score: {:.4f}".format(first_text, second_text, score))
                
            
