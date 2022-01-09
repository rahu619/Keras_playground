from numpy import array
from numpy import asarray
from numpy import zeros

class WordEmbedding:    
    embeddings_dictionary = dict()    
    def __init__(self, config_dict):
        app_dict = config_dict.get('APP', {})
        glove_file = open(app_dict['GLOVE_PATH'], encoding="utf8")
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            self.embeddings_dictionary[word] = vector_dimensions
        glove_file.close()


    def getEmbeddingMatrix(self, vocab_size, tokenizer):
        embedding_matrix = zeros((vocab_size, 300))
        for word, index in tokenizer.word_index.items():
            embedding_vector = self.embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        
        return embedding_matrix
            


