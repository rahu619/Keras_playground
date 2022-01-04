import re
class Process:
    def preprocess(self, text):
        # removing punctuations
        text = re.sub('[^a-zA-Z]', '', text)
        
        # Removing single character
        text = re.sub(r"\s+[a-zA-Z]\s+", '', text)
        
        # Removing spaces
        text  = re.sub(r'\s+', '', text)
        
        return text