from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

class TfidfRetriever:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.df['sinopsis'])
    
    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors).flatten()
        top_n = scores.argsort()[-top_k:][::-1]
        return self.df.iloc[top_n]
