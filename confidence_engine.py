import pandas as pd
import numpy as np
import os

# ── Load data references from recommender ───────────────────
# These are passed in from recommender.py so we don't reload data

def compute_confidence(
    movie_title,
    recommended_title,
    collab_sim_df,
    content_sim_df,
    clean_data,
    movies_df,
    collab_weight=0.4,
    content_weight=0.6
):
    """
    Computes a full confidence breakdown for a single recommendation.

    Returns a dictionary with:
    - overall_score       : weighted hybrid score (0-100%)
    - collab_score        : collaborative filtering similarity (0-100%)
    - content_score       : content-based similarity (0-100%)
    - popularity_score    : how popular/well-rated the movie is (0-100%)
    - novelty_score       : how non-mainstream the movie is (0-100%)
    - collab_user_count   : number of users behind the collab score
    - genre_overlap       : list of shared genres
    - reason              : plain English explanation
    """

    result = {
        'overall_score':     0,
        'collab_score':      0,
        'content_score':     0,
        'popularity_score':  0,
        'novelty_score':     0,
        'collab_user_count': 0,
        'genre_overlap':     [],
        'reason':            ''
    }

    # ── 1. Collaborative score ───────────────────────────────
    collab_raw = 0
    if movie_title in collab_sim_df.columns and \
       recommended_title in collab_sim_df.columns:
        sim = collab_sim_df[movie_title]
        if isinstance(sim, pd.DataFrame):
            sim = sim.iloc[:, 0]
        if recommended_title in sim.index:
            collab_raw = float(sim[recommended_title])

    # Normalise against the max similarity for this movie
    if movie_title in collab_sim_df.columns:
        sim_all = collab_sim_df[movie_title]
        if isinstance(sim_all, pd.DataFrame):
            sim_all = sim_all.iloc[:, 0]
        max_sim = sim_all.drop(movie_title, errors='ignore').max()
        if max_sim > 0:
            result['collab_score'] = round((collab_raw / max_sim) * 100, 1)

    # Estimate user count behind the score
    rec_ratings = clean_data[clean_data['title'] == recommended_title]
    result['collab_user_count'] = len(rec_ratings)

    # ── 2. Content score ─────────────────────────────────────
    content_raw = 0
    if movie_title in content_sim_df.columns and \
       recommended_title in content_sim_df.columns:
        sim = content_sim_df[movie_title]
        if isinstance(sim, pd.DataFrame):
            sim = sim.iloc[:, 0]
        if recommended_title in sim.index:
            content_raw = float(sim[recommended_title])

    # Normalise against the max similarity for this movie
    if movie_title in content_sim_df.columns:
        sim_all = content_sim_df[movie_title]
        if isinstance(sim_all, pd.DataFrame):
            sim_all = sim_all.iloc[:, 0]
        max_sim = sim_all.drop(movie_title, errors='ignore').max()
        if max_sim > 0:
            result['content_score'] = round((content_raw / max_sim) * 100, 1)

    # ── 3. Genre overlap ─────────────────────────────────────
    src_row  = movies_df[movies_df['title'] == movie_title]
    rec_row  = movies_df[movies_df['title'] == recommended_title]

    src_genres = set()
    rec_genres = set()

    if not src_row.empty and pd.notna(src_row.iloc[0].get('genres', '')):
        src_genres = set(src_row.iloc[0]['genres'].split('|'))
    if not rec_row.empty and pd.notna(rec_row.iloc[0].get('genres', '')):
        rec_genres = set(rec_row.iloc[0]['genres'].split('|'))

    overlap = src_genres & rec_genres
    overlap.discard('')
    result['genre_overlap'] = sorted(list(overlap))

    # ── 4. Popularity score ──────────────────────────────────
    if not rec_row.empty:
        row        = rec_row.iloc[0]
        vote_avg   = float(row.get('vote_average', 0) or 0)
        vote_count = float(row.get('vote_count',   0) or 0)

        # Normalise vote_average (0-10) to 0-100
        avg_score   = (vote_avg / 10) * 100

        # Normalise vote_count using log scale
        # 10,000+ votes = full score, 50 votes = low score
        count_score = min(np.log1p(vote_count) / np.log1p(10000) * 100, 100)

        result['popularity_score'] = round((avg_score * 0.7 + count_score * 0.3), 1)

    # ── 5. Novelty score ─────────────────────────────────────
    # High novelty = high quality but low mainstream exposure
    # Inverse of popularity_count_score — good rating, fewer votes = more novel
    if not rec_row.empty:
        row        = rec_row.iloc[0]
        vote_avg   = float(row.get('vote_average', 0) or 0)
        vote_count = float(row.get('vote_count',   0) or 0)

        quality_score  = (vote_avg / 10) * 100
        exposure_score = min(np.log1p(vote_count) / np.log1p(10000) * 100, 100)

        # High quality + low exposure = high novelty
        novelty = quality_score * (1 - (exposure_score / 100) * 0.7)
        result['novelty_score'] = round(min(novelty, 100), 1)

    # ── 6. Overall score ─────────────────────────────────────
    overall = (
        collab_raw  * collab_weight  * 100 +
        content_raw * content_weight * 100
    )
    result['overall_score'] = round(min(overall, 100), 1)

    # ── 7. Plain English reason ──────────────────────────────
    reasons = []

    if result['genre_overlap']:
        genres_str = ', '.join(result['genre_overlap'][:3])
        reasons.append(f"strong genre match ({genres_str})")

    if result['collab_score'] >= 70:
        reasons.append(
            f"highly rated by {result['collab_user_count']} "
            f"users who also enjoyed {movie_title.split('(')[0].strip()}"
        )
    elif result['collab_score'] >= 40:
        reasons.append(
            f"liked by users who enjoyed "
            f"{movie_title.split('(')[0].strip()}"
        )

    if result['content_score'] >= 70:
        reasons.append("very similar content and style")
    elif result['content_score'] >= 40:
        reasons.append("similar content and themes")

    if result['novelty_score'] >= 70:
        reasons.append("a hidden gem worth discovering")

    if result['popularity_score'] >= 80:
        reasons.append("critically acclaimed and widely loved")

    if not reasons:
        reasons.append("matches your viewing preferences")

    result['reason'] = "Recommended because: " + ", ".join(reasons) + "."

    return result


def compute_confidence_batch(
    movie_title,
    recommendations,
    collab_sim_df,
    content_sim_df,
    clean_data,
    movies_df,
    collab_weight=0.4,
    content_weight=0.6
):
    """
    Compute confidence scores for a list of recommendations.
    recommendations = list of (title, score) tuples from hybrid_recommend()
    Returns list of dicts with title, score, and full confidence breakdown.
    """
    results = []
    for rec_title, hybrid_score in recommendations:
        conf = compute_confidence(
            movie_title,
            rec_title,
            collab_sim_df,
            content_sim_df,
            clean_data,
            movies_df,
            collab_weight,
            content_weight
        )
        conf['title']        = rec_title
        conf['hybrid_score'] = hybrid_score
        results.append(conf)

    return results