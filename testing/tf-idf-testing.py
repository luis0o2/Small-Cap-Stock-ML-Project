from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. Sample Data (Documents)
corpus = [
    'The cat sat on the mat.',
    'The dog sat on the log.',
    'Cats and dogs are great pets.',
    'I love my pet cat.'
]

# 2. Initialize the Vectorizer
# stop_words='english' removes common words like 'the', 'is', etc.
vectorizer = TfidfVectorizer(stop_words='english')

# 3. Fit and Transform the data
tfidf_matrix = vectorizer.fit_transform(corpus)

# 4. Convert to a readable format (DataFrame)
feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("TF-IDF Feature Matrix:")
print(df)