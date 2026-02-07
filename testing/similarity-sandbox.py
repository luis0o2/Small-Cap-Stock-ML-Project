import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = 'data/cleaned/train_dataset.csv'

#pick 15 headlines
df = pd.read_csv(DATA_PATH)
sandbox_df = df.sample(n=15, random_state=42).reset_index(drop=True)

#create tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sandbox_df['headline'].astype(str))

#compute similarity
query_index = 5
query_headline = sandbox_df['headline'].iloc[query_index]

#cosine sim or compare against all headlines
similarities = cosine_similarity(tfidf_matrix[query_index], tfidf_matrix).flatten()

# rank and display

sandbox_df['similarity_score'] = similarities
results = sandbox_df.sort_values(by='similarity_score', ascending=False)

print(f"--- QUERY HEADLINE ---")
print(f"'{query_headline}'\n")
print(f"--- TOP 5 MATCHES ---")
print(results[['headline', 'similarity_score']].head(5))