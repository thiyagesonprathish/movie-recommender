import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# ── Load data ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

clean_data = pd.read_csv(os.path.join(DATA_DIR, 'clean_data.csv'))
movies     = pd.read_csv(os.path.join(DATA_DIR, 'ml-latest-small', 'movies.csv'))

# ── Load optimal weights ────────────────────────────────────
weights_path = os.path.join(MODELS_DIR, 'hybrid_weights.json')
if os.path.exists(weights_path):
    with open(weights_path) as f:
        w = json.load(f)
    COLLAB_WEIGHT  = w['collab_weight']
    CONTENT_WEIGHT = w['content_weight']
else:
    COLLAB_WEIGHT  = 0.4
    CONTENT_WEIGHT = 0.6

# ── Build collaborative matrix ──────────────────────────────
print("Building collaborative matrix...")
user_movie_matrix = clean_data.pivot_table(
    index='userId', columns='title', values='rating'
).fillna(0)

collab_similarity_df = pd.DataFrame(
    cosine_similarity(user_movie_matrix.T),
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)
print("✅ Collaborative matrix ready")

# ── Build content matrix ────────────────────────────────────
print("Building content matrix...")
movies['genres_clean'] = movies['genres'].str.replace(
    '|', ' ', regex=False).str.replace(
    '(no genres listed)', '', regex=False).str.strip()
movies['year'] = pd.to_numeric(
    movies['title'].str.extract(r'\((\d{4})\)')[0],
    errors='coerce').fillna(0).astype(int)
movies['combined_features'] = (
    movies['genres_clean'] + ' ' +
    movies['genres_clean'] + ' ' +
    movies['year'].astype(str)
).fillna('')

tfidf = TfidfVectorizer(
    min_df=2, max_features=5000,
    strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{2,}', ngram_range=(1, 2),
    stop_words='english'
)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
content_similarity_df = pd.DataFrame(
    cosine_similarity(tfidf_matrix, tfidf_matrix),
    index=movies['title'],
    columns=movies['title']
)
print("✅ Content matrix ready")

# ── All movie titles for search ─────────────────────────────
ALL_TITLES = sorted(collab_similarity_df.columns.tolist())


# ── Hybrid recommendation ───────────────────────────────────
def hybrid_recommend(movie_title, n=10):
    if movie_title not in collab_similarity_df.columns and \
       movie_title not in content_similarity_df.columns:
        return []

    scores = {}

    if movie_title in collab_similarity_df.columns:
        s = collab_similarity_df[movie_title].drop(movie_title)
        mn, mx = s.min(), s.max()
        if mx > mn:
            s = (s - mn) / (mx - mn)
        for t, v in s.items():
            scores[t] = scores.get(t, 0) + COLLAB_WEIGHT * v

    if movie_title in content_similarity_df.columns:
        s = content_similarity_df[movie_title].drop(movie_title)
        mn, mx = s.min(), s.max()
        if mx > mn:
            s = (s - mn) / (mx - mn)
        for t, v in s.items():
            scores[t] = scores.get(t, 0) + CONTENT_WEIGHT * v

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [(t, round(s, 4)) for t, s in top if t != movie_title][:n]
    return top


# ── Cold start — popular movies ─────────────────────────────
def get_popular_movies(n=10):
    popular = (
        clean_data.groupby('title')['rating']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'num_ratings', 'mean': 'avg_rating'})
    )
    popular = popular[popular['num_ratings'] >= 50]
    popular['score'] = popular['avg_rating'] * np.log1p(popular['num_ratings'])
    top = popular.sort_values('score', ascending=False).head(n)
    return [(t, round(r['avg_rating'], 2)) for t, r in top.iterrows()]


# ── Search titles ───────────────────────────────────────────
def search_titles(query, n=8):
    q = query.lower()
    return [t for t in ALL_TITLES if q in t.lower()][:n]