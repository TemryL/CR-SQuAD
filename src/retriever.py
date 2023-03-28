import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")


class Retriever:
    def __init__(self, contexts):
        self.contexts = contexts
    
    def retrieve(self, question):
        raise NotImplementedError
    
    def _preprocess(self, text):
        # Lower case
        text = text.lower()
        
        # Remove punctuation
        table = str.maketrans('', '', string.punctuation)
        text = text.translate(table)
        
        # Remove stopwords and perform stemming
        ps = PorterStemmer()
        cachedStopWords = stopwords.words("english")
        text = [ps.stem(word) for word in text.split() if word not in cachedStopWords]
        text = ' '.join(text)
        return text
    
    def _get_accuracy(self, questions):
        accuracy = 0
        for question in questions:
            true_ctx_id = question['context_id']
            retrieved_ctx_id, _ = self.retrieve(question['question'])
            if retrieved_ctx_id == true_ctx_id:
                accuracy += 1
        return accuracy/len(questions)