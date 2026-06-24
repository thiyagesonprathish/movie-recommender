import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE    = 'https://api.themoviedb.org/3'
POSTER_BASE  = 'https://image.tmdb.org/t/p/w500'

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TMDB_DIR    = os.path.join(BASE_DIR, 'data', 'tmdb')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

# ── Load TMDB data ──────────────────────────────────────────
print("Loading TMDB dataset...")
movies  = pd.read_csv(os.path.join(TMDB_DIR, 'tmdb_movies.csv'))
ratings = pd.read_csv(os.path.join(TMDB_DIR, 'tmdb_ratings.csv'))
trending_df = pd.read_csv(os.path.join(TMDB_DIR, 'tmdb_trending.csv'))

movies['genres']   = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['cast']     = movies['cast'].fillna('')
movies['director'] = movies['director'].fillna('')
movies['overview'] = movies['overview'].fillna('')

movies['year'] = pd.to_numeric(
    movies['release_date'].str[:4], errors='coerce'
).fillna(0).astype(int)

clean_data = ratings.rename(columns={'title': 'title'})

print(f"✅ Loaded {len(movies)} movies, {len(ratings)} ratings")

# ── Load optimal weights ────────────────────────────────────
weights_path = os.path.join(MODELS_DIR, 'hybrid_weights.json')
if os.path.exists(weights_path):
    with open(weights_path) as f:
        w = json.load(f)
    COLLAB_WEIGHT  = w.get('collab_weight', 0.4)
    CONTENT_WEIGHT = w.get('content_weight', 0.6)
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
print(f"✅ Collaborative matrix ready: {collab_similarity_df.shape}")

# ── Build content matrix (richer features now) ──────────────
print("Building content matrix...")
movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
movies['keywords_clean'] = movies['keywords'].str.replace('|', ' ', regex=False)
movies['cast_clean'] = movies['cast'].str.replace('|', ' ', regex=False).str.replace(' ', '', regex=False)
movies['director_clean'] = movies['director'].str.replace(' ', '', regex=False)

movies['combined_features'] = (
    movies['genres_clean'] + ' ' +
    movies['genres_clean'] + ' ' +       # weight genres x2
    movies['keywords_clean'] + ' ' +
    movies['cast_clean'] + ' ' +
    movies['director_clean'] + ' ' +
    movies['year'].astype(str)
).fillna('')

tfidf = TfidfVectorizer(
    min_df=2, max_features=8000,
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
print(f"✅ Content matrix ready: {content_similarity_df.shape}")

ALL_TITLES = sorted(collab_similarity_df.columns.tolist())


# ── TMDB live lookups (poster/overview for any movie) ────────
def get_movie_details(movie_title):
    """Get poster, overview, rating from local dataset first, TMDB as fallback."""
    match = movies[movies['title'] == movie_title]
    if match.empty:
        return {'poster': None, 'overview': 'No summary available.',
                'rating': 'N/A', 'year': '', 'genres': []}

    row = match.iloc[0]
    poster = f"{POSTER_BASE}{row['poster_path']}" if pd.notna(row.get('poster_path')) and row.get('poster_path') else None

    return {
        'poster':   poster,
        'overview': row['overview'] if row['overview'] else 'No summary available.',
        'rating':   round(row['vote_average'], 1) if pd.notna(row['vote_average']) else 'N/A',
        'year':     str(row['year']) if row['year'] > 0 else '',
        'genres':   row['genres'].split('|')[:3] if row['genres'] else []
    }


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


# ── Cold start ──────────────────────────────────────────────
def get_popular_movies(n=10):
    popular = movies.copy()
    popular = popular[popular['vote_count'] >= 50]
    popular['score'] = popular['vote_average'] * np.log1p(popular['vote_count'])
    top = popular.sort_values('score', ascending=False).head(n)
    return [(row['title'], round(row['vote_average']/2, 2)) for _, row in top.iterrows()]


# ── Trending now ─────────────────────────────────────────────
def get_trending_movies(n=12):
    top = trending_df.head(n)
    results = []
    for _, row in top.iterrows():
        poster = f"{POSTER_BASE}{row['poster_path']}" if pd.notna(row.get('poster_path')) and row.get('poster_path') else None
        results.append({
            'title':    row['title'],
            'poster':   poster,
            'overview': row['overview'] if pd.notna(row['overview']) else '',
            'rating':   round(row['vote_average'], 1) if pd.notna(row['vote_average']) else 'N/A',
            'year':     row['release_date'][:4] if pd.notna(row['release_date']) and row['release_date'] else '',
            'genres':   row['genres'].split('|')[:2] if pd.notna(row['genres']) and row['genres'] else []
        })
    return results


# ── Hidden gems ───────────────────────────────────────────────
def get_hidden_gems(n=12):
    """High rating, low vote count = hidden gem."""
    gems = movies.copy()
    gems = gems[(gems['vote_average'] >= 7.0) & (gems['vote_count'] >= 20) & (gems['vote_count'] <= 500)]
    gems['gem_score'] = gems['vote_average'] / np.log1p(gems['vote_count'])
    top = gems.sort_values('gem_score', ascending=False).head(n)

    results = []
    for _, row in top.iterrows():
        poster = f"{POSTER_BASE}{row['poster_path']}" if pd.notna(row.get('poster_path')) and row.get('poster_path') else None
        results.append({
            'title':      row['title'],
            'poster':     poster,
            'overview':   row['overview'],
            'rating':     round(row['vote_average'], 1),
            'vote_count': int(row['vote_count']),
            'year':       str(row['year']) if row['year'] > 0 else '',
            'genres':     row['genres'].split('|')[:2] if row['genres'] else []
        })
    return results


# ── Search ──────────────────────────────────────────────────
def search_titles(query, n=8):
    q = query.lower()
    return [t for t in ALL_TITLES if q in t.lower()][:n]

# ── Mood to genre mapping ───────────────────────────────────
MOOD_MAP = {
    'laugh': {
        'label':  'Make Me Laugh 😂',
        'genres': ['Comedy'],
        'desc':   'Light-hearted comedies to brighten your day'
    },
    'cry': {
        'label':  'Make Me Cry 😢',
        'genres': ['Drama', 'Romance'],
        'desc':   'Emotional dramas that will move you deeply'
    },
    'scared': {
        'label':  'Scare Me 😱',
        'genres': ['Horror', 'Thriller'],
        'desc':   'Edge of your seat horror and thrillers'
    },
    'mindblown': {
        'label':  'Blow My Mind 🤯',
        'genres': ['Science Fiction', 'Mystery'],
        'desc':   'Mind-bending stories that will make you think'
    },
    'romance': {
        'label':  'Feel the Love ❤️',
        'genres': ['Romance', 'Drama'],
        'desc':   'Heartwarming love stories'
    },
    'action': {
        'label':  'Action Rush 💥',
        'genres': ['Action', 'Adventure'],
        'desc':   'High octane action and adventure'
    },
    'adventure': {
        'label':  'Take Me on an Adventure 🌍',
        'genres': ['Adventure', 'Fantasy'],
        'desc':   'Epic journeys to faraway worlds'
    },
    'inspired': {
        'label':  'Inspire Me 💪',
        'genres': ['Documentary', 'Drama'],
        'desc':   'True stories and dramas that inspire'
    }
}


def mood_recommend(mood_key, n=12):
    """Return top N movies matching a mood based on genre + TMDB popularity."""
    if mood_key not in MOOD_MAP:
        return [], {}

    mood   = MOOD_MAP[mood_key]
    genres = mood['genres']

    matched = movies[
        movies['genres'].apply(
            lambda g: any(genre in g for genre in genres)
        )
    ].copy()

    if matched.empty:
        return [], mood

    matched = matched[matched['vote_count'] >= 20]
    matched['score'] = matched['vote_average'] * np.log1p(matched['vote_count'])

    top = matched.sort_values('score', ascending=False).head(n)
    results = top[['title', 'vote_average', 'genres']].rename(
        columns={'vote_average': 'avg_rating'}
    ).to_dict('records')

    # Convert TMDB 0-10 scale to 0-5 for consistency with old template
    for r in results:
        r['avg_rating'] = round(r['avg_rating'] / 2, 2)

    return results, mood


# ── Movie DNA ───────────────────────────────────────────────
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'TV Movie', 'Thriller', 'War', 'Western'
]

def get_movie_dna(movie_title):
    """Returns the genre breakdown of a movie using TMDB data."""
    match = movies[movies['title'] == movie_title]
    if match.empty:
        return None

    movie_genres = [g for g in match.iloc[0]['genres'].split('|') if g]

    dna = []
    for genre in ALL_GENRES:
        if genre in movie_genres:
            genre_movies = movies[
                movies['genres'].str.contains(genre, na=False)
            ]
            avg = round(genre_movies['vote_average'].mean() / 2, 2) if len(genre_movies) > 0 else 0
            dna.append({
                'genre':      genre,
                'present':    True,
                'avg_rating': avg
            })

    movie_row = match.iloc[0]
    movie_avg = round(movie_row['vote_average'] / 2, 2) if pd.notna(movie_row['vote_average']) else 0

    return {
        'genres':      movie_genres,
        'dna':         dna,
        'movie_avg':   movie_avg,
        'num_ratings': int(movie_row['vote_count']) if pd.notna(movie_row['vote_count']) else 0
    }