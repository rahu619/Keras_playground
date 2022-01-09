import re
class Process:
    def preprocess(self, text):
        text = re.sub('[^a-zA-Z]', '', text)
        text = re.sub(r"\s+[a-zA-Z]\s+", '', text)
        text  = re.sub(r'\s+', '', text)
        return text