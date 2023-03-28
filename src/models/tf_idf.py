import numpy as np

from src.models.retriever import Retriever
from sklearn.feature_extraction.text import CountVectorizer


class TFIDF_Retriever(Retriever):
    def __init__(self, contexts, questions):
        super().__init__(contexts)
        self.tfidf = self._compute_TFIDF()
        self.accuracy = self._get_accuracy(questions)
    
    def retrieve(self, question):
        question = self._preprocess(question)
        vocabulary = list(self.vectorizer.get_feature_names_out())
        
        # Get indices of terms in the question
        q_term_idx = []
        for term in question.split():
            if term in vocabulary:
                q_term_idx.append(vocabulary.index(term))
        
        # Compute sum of tf-idf over the set of terms of the question for each context
        scores = self.tfidf[:,q_term_idx].sum(axis=1)
        
        # Retrieve context with the highest score
        context_id = np.argmax(scores)
        return context_id, self.contexts[context_id]
    
    def _compute_TFIDF(self):
        documents = [self._preprocess(document) for document in self.contexts]
        # Get term-frequency (raw count)
        self.vectorizer = CountVectorizer()
        tf = self.vectorizer.fit_transform(documents).toarray()
        
        # Calculate the IDF for each term 
        doc_freq = np.sum(tf > 0, axis=0)
        N = tf.shape[0]
        idf = np.log(N / (doc_freq))
        
        # Normalized term-frequency double normalization 0.5
        tf = 0.5 + 0.5*tf/np.max(tf, axis=1)[:, np.newaxis]
        
        # Compute TF-IDF
        tf_idf = tf*idf
        return tf_idf