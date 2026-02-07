# FEATURE ENGINEERING
from sklearn.feature_extraction.text import TfidfVectorizer

def build_word_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b', # Keeps only words with 3+ letters (no numbers),
        min_df=5
    )
    
    
    
    

