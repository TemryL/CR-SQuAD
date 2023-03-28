import numpy as np

from src.retriever import Retriever
from sklearn.feature_extraction.text import CountVectorizer


class BM25_Retriever(Retriever):
    def __init__(self, contexts, questions):
        super().__init__(contexts)
        self.tf = self._compute_tf()
        self.idf = self._compute_idf()
        self.bm25 = self._compute_bm25()
        self.accuracy = self._get_accuracy(questions)
    
    def retrieve(self, question):
        question = self._preprocess(question)
        vocabulary = list(self.vectorizer.get_feature_names_out())
        
        # Get indices of terms in the question
        q_term_idx = []
        for term in question.split():
            if term in vocabulary:
                q_term_idx.append(vocabulary.index(term))
        
        # Compute sum of bm25 scores over the set of terms of the question for each context
        scores = self.bm25[:,q_term_idx].sum(axis=1)
        
        # Retrieve context with the highest score
        context_id = np.argmax(scores)
        return context_id, self.contexts[context_id]
    
    def _compute_tf(self):
        documents = [self._preprocess(document) for document in self.contexts]
        # Get term-frequency (raw count)
        self.vectorizer = CountVectorizer()
        tf = self.vectorizer.fit_transform(documents).toarray()
        return tf
    
    def _compute_idf(self):
        # Calculate the IDF for each term 
        doc_freq = np.sum(self.tf > 0, axis=0)
        N = self.tf.shape[0]
        idf = np.log(N / (doc_freq))
        return idf
    
    def _compute_bm25(self):
        k1 = 1.2
        b = 0.75
        mask = np.zeros_like(self.tf)
        mask[self.tf>0] = 1
        D = mask.sum(axis=1)
        avgdl = D.mean()
        F = (self.tf*(k1+1)/(self.tf+k1*(1-b+b*D[:,np.newaxis]/avgdl)))
        scores = self.idf * F
        return scores