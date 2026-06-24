import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv('TMDB_API_KEY')
BASE_URL = 'https://api.themoviedb.org/3'

# ── Config ──────────────────────────────────────────────────
TOTAL_PAGES   = 500   # 500 pages x 20 movies = 10,000 movies
OUTPUT_DIR    = 'data/tmdb'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper ──────────────────────────────────────────────────
def get(endpoint, params={}):
    params['api_key'] = API_KEY
    params['language'] = 'en-US'
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Error: {e}")
    return None

# ── Step 1: Fetch movie list ─────────────────────────────────
print("=" * 55)
print("Step 1: Fetching movie list from TMDB...")
print("=" * 55)

all_movies = []
seen_ids   = set()

# Fetch from multiple endpoints to get diverse movies
endpoints = [
    ('/movie/popular',     {'page': 1}),
    ('/movie/top_rated',   {'page': 1}),
    ('/movie/now_playing', {'page': 1}),
]

# Popular movies across all pages
for page in range(1, TOTAL_PAGES + 1):
    data = get('/movie/popular', {'page': page})
    if not data or not data.get('results'):
        break

    for movie in data['results']:
        if movie['id'] not in seen_ids and movie.get('vote_count', 0) >= 10:
            seen_ids.add(movie['id'])
            all_movies.append({
                'tmdb_id':      movie['id'],
                'title':        movie.get('title', ''),
                'overview':     movie.get('overview', ''),
                'release_date': movie.get('release_date', ''),
                'popularity':   movie.get('popularity', 0),
                'vote_average': movie.get('vote_average', 0),
                'vote_count':   movie.get('vote_count', 0),
                'genre_ids':    movie.get('genre_ids', []),
                'poster_path':  movie.get('poster_path', '')
            })

    if page % 50 == 0:
        print(f"  Fetched page {page}/{TOTAL_PAGES} — {len(all_movies)} movies so far")

    time.sleep(0.05)  # Respect rate limit

# Top rated movies
print("\nFetching top rated movies...")
for page in range(1, 201):
    data = get('/movie/top_rated', {'page': page})
    if not data or not data.get('results'):
        break

    for movie in data['results']:
        if movie['id'] not in seen_ids and movie.get('vote_count', 0) >= 10:
            seen_ids.add(movie['id'])
            all_movies.append({
                'tmdb_id':      movie['id'],
                'title':        movie.get('title', ''),
                'overview':     movie.get('overview', ''),
                'release_date': movie.get('release_date', ''),
                'popularity':   movie.get('popularity', 0),
                'vote_average': movie.get('vote_average', 0),
                'vote_count':   movie.get('vote_count', 0),
                'genre_ids':    movie.get('genre_ids', []),
                'poster_path':  movie.get('poster_path', '')
            })
    time.sleep(0.05)

print(f"\n✅ Total movies collected: {len(all_movies)}")

# ── Step 2: Fetch genre list ─────────────────────────────────
print("\n" + "=" * 55)
print("Step 2: Fetching genre list...")
print("=" * 55)

genre_data = get('/genre/movie/list')
genre_map  = {}
if genre_data:
    for g in genre_data['genres']:
        genre_map[g['id']] = g['name']
    print(f"✅ {len(genre_map)} genres fetched")

# Map genre IDs to names
for movie in all_movies:
    movie['genres'] = '|'.join([
        genre_map.get(gid, '')
        for gid in movie['genre_ids']
        if gid in genre_map
    ])
    del movie['genre_ids']

# ── Step 3: Fetch detailed info for each movie ───────────────
print("\n" + "=" * 55)
print("Step 3: Fetching detailed info (keywords, cast, runtime)...")
print("This will take 15-20 minutes. Please wait...")
print("=" * 55)

detailed_movies = []
failed          = 0

for i, movie in enumerate(all_movies):
    tmdb_id = movie['tmdb_id']

    # Fetch movie details
    details = get(f'/movie/{tmdb_id}')
    credits = get(f'/movie/{tmdb_id}/credits')
    keywords_data = get(f'/movie/{tmdb_id}/keywords')

    if not details:
        failed += 1
        continue

    # Extract runtime
    runtime = details.get('runtime', 0) or 0

    # Extract cast (top 5)
    cast = []
    if credits and credits.get('cast'):
        cast = [c['name'] for c in credits['cast'][:5]]

    # Extract director
    director = ''
    if credits and credits.get('crew'):
        directors = [c['name'] for c in credits['crew'] if c['job'] == 'Director']
        director  = directors[0] if directors else ''

    # Extract keywords
    keywords = []
    if keywords_data and keywords_data.get('keywords'):
        keywords = [k['name'] for k in keywords_data['keywords'][:10]]

    detailed_movies.append({
        **movie,
        'runtime':   runtime,
        'cast':      '|'.join(cast),
        'director':  director,
        'keywords':  '|'.join(keywords),
    })

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(all_movies)} movies — {failed} failed")

    time.sleep(0.1)  # Respect rate limit

print(f"\n✅ Detailed info fetched for {len(detailed_movies)} movies")
print(f"   Failed: {failed}")

# ── Step 4: Save movies CSV ──────────────────────────────────
print("\n" + "=" * 55)
print("Step 4: Saving movies CSV...")
print("=" * 55)

movies_df = pd.DataFrame(detailed_movies)

# Clean up
movies_df = movies_df[movies_df['title'] != '']
movies_df = movies_df[movies_df['vote_count'] >= 10]
movies_df = movies_df.drop_duplicates(subset=['tmdb_id'])
movies_df = movies_df.reset_index(drop=True)

movies_df.to_csv(f'{OUTPUT_DIR}/tmdb_movies.csv', index=False)
print(f"✅ Saved {len(movies_df)} movies to data/tmdb/tmdb_movies.csv")

# ── Step 5: Build simulated ratings matrix ───────────────────
print("\n" + "=" * 55)
print("Step 5: Building simulated user ratings matrix...")
print("=" * 55)

import numpy as np

np.random.seed(42)

# We simulate 2000 users rating movies
# Based on TMDB vote distribution
NUM_USERS    = 2000
movies_list  = movies_df['tmdb_id'].tolist()
ratings_list = []

for user_id in range(1, NUM_USERS + 1):
    # Each user rates between 20 and 150 movies
    n_ratings = np.random.randint(20, 150)

    # Users tend to rate more popular movies
    popularity_weights = movies_df['popularity'].values
    popularity_weights = popularity_weights / popularity_weights.sum()

    rated_indices = np.random.choice(
        len(movies_list),
        size=min(n_ratings, len(movies_list)),
        replace=False,
        p=popularity_weights
    )

    for idx in rated_indices:
        movie       = movies_df.iloc[idx]
        base_rating = movie['vote_average'] / 2  # Convert 0-10 to 0-5

        # Add personal variation
        personal_bias = np.random.normal(0, 0.5)
        rating        = base_rating + personal_bias
        rating        = round(max(0.5, min(5.0, rating)) * 2) / 2  # Round to 0.5

        ratings_list.append({
            'userId':  user_id,
            'tmdb_id': movies_list[idx],
            'title':   movie['title'],
            'rating':  rating
        })

    if user_id % 500 == 0:
        print(f"  Generated ratings for {user_id}/{NUM_USERS} users")

ratings_df = pd.DataFrame(ratings_list)
ratings_df.to_csv(f'{OUTPUT_DIR}/tmdb_ratings.csv', index=False)
print(f"✅ Saved {len(ratings_df)} ratings to data/tmdb/tmdb_ratings.csv")

# ── Step 6: Fetch trending movies ────────────────────────────
print("\n" + "=" * 55)
print("Step 6: Fetching trending movies...")
print("=" * 55)

trending_list = []
for page in range(1, 6):
    data = get('/trending/movie/week', {'page': page})
    if data and data.get('results'):
        for movie in data['results']:
            trending_list.append({
                'tmdb_id':      movie['id'],
                'title':        movie.get('title', ''),
                'overview':     movie.get('overview', ''),
                'release_date': movie.get('release_date', ''),
                'popularity':   movie.get('popularity', 0),
                'vote_average': movie.get('vote_average', 0),
                'vote_count':   movie.get('vote_count', 0),
                'poster_path':  movie.get('poster_path', ''),
                'genre_ids':    movie.get('genre_ids', [])
            })

for movie in trending_list:
    movie['genres'] = '|'.join([
        genre_map.get(gid, '')
        for gid in movie.get('genre_ids', [])
        if gid in genre_map
    ])
    movie.pop('genre_ids', None)

trending_df = pd.DataFrame(trending_list).drop_duplicates(subset=['tmdb_id'])
trending_df.to_csv(f'{OUTPUT_DIR}/tmdb_trending.csv', index=False)
print(f"✅ Saved {len(trending_df)} trending movies to data/tmdb/tmdb_trending.csv")

# ── Final summary ────────────────────────────────────────────
print("\n" + "=" * 55)
print("✅ TMDB DATA COLLECTION COMPLETE!")
print("=" * 55)
print(f"  Movies:          {len(movies_df)}")
print(f"  Ratings:         {len(ratings_df)}")
print(f"  Trending:        {len(trending_df)}")
print(f"  Unique users:    {NUM_USERS}")
print(f"  Year range:      {movies_df['release_date'].str[:4].min()} — {movies_df['release_date'].str[:4].max()}")
print(f"\n  Files saved to:  data/tmdb/")
print("=" * 55)