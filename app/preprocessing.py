import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    stop_words = set(stopwords.words('indonesian'))
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(w) for w in text.split() if w not in stop_words])
