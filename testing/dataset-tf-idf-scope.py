from src.features import build_word_vectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

DATA_PATH = 'data/cleaned/train_dataset.csv'

df = pd.read_csv(DATA_PATH)

text_data = df['headline'].astype(str).fillna('')

vectorizer = build_word_vectorizer()

tfidf_matrix = vectorizer.fit_transform(text_data)

feature_names = vectorizer.get_feature_names_out()
dense_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=feature_names
)

print(f"Successfully created a matrix of shape: {tfidf_matrix.shape}")
print(dense_df.head())

# 1. Sum the TF-IDF scores for each word
importance = tfidf_matrix.mean(axis=0).tolist()[0]
features = vectorizer.get_feature_names_out()

# 2. Create a DataFrame of words and their scores
feature_importance = pd.DataFrame({'word': features, 'score': importance})

# 3. Sort by score and look at the top 20
top_20 = feature_importance.sort_values(by='score', ascending=False).head(20)
print(top_20)

# 1. Compare the first headline (index 0) against all others
# tfidf_matrix is what you created in the previous step
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

# 2. Get the scores
import numpy as np
related_docs_indices = similarities.argsort()[0][-5:]  # Top 5 most similar

print("Most similar headlines to the first one:")
for i in related_docs_indices:
    print(f"Index {i}: {df['headline'].iloc[i]} (Score: {similarities[0][i]:.2f})")