from src.features import build_word_vectorizer
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

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# cosine sim, compare all headlines
#against each other 
# creates a 10k * 10k matrix

text_sim_matrix = cosine_similarity(tfidf_matrix)

"""normalize numeric features
using minmaxscalar to normalize price 
from 0-1
"""
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(df[['price_now']])

""" COMPUTE WEIGHTED SUM    
goal: score = (text sim * 0.7) + price proximity * 0.3
as example comparing everything
to first headline index 0
"""
text_scores = text_sim_matrix[0]

#calc how 'close' prices are 1-abs diff
price_diffs = np.abs(price_scaled - price_scaled[0]).flatten()
price_proximity = 1 - price_diffs

#Weighted Sum
w1, w2 = 0.7, 0.3
final_scores = (text_scores * w1) + (price_proximity * w2)

#applying threshold

threshold = 0.5
matches = np.where(final_scores > threshold)[0]

#results
results = df.iloc[matches].copy()
results['final_score'] = final_scores[matches]
print(results[['headline', 'price_now', 'final_score']].sort_values(by='final_score', ascending=False))