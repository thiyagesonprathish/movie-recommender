import numpy as np
import pandas as pd
from datetime import datetime

# ── Context to genre weight mapping ─────────────────────────
CONTEXT_GENRE_WEIGHTS = {
    'time_of_day': {
        'morning':    {'Comedy': 1.3, 'Animation': 1.3, 'Family': 1.2, 'Documentary': 1.1, 'Thriller': 0.7, 'Horror': 0.6},
        'afternoon':  {'Action': 1.2, 'Adventure': 1.2, 'Comedy': 1.1, 'Drama': 1.0},
        'evening':    {'Drama': 1.3, 'Thriller': 1.2, 'Romance': 1.2, 'Crime': 1.1, 'Horror': 1.1},
        'late_night': {'Thriller': 1.4, 'Horror': 1.4, 'Mystery': 1.3, 'Crime': 1.2, 'Science Fiction': 1.2}
    },
    'day_type': {
        'weekday':  {'Drama': 1.1, 'Documentary': 1.2, 'Comedy': 1.1, 'Action': 0.9},
        'weekend':  {'Action': 1.3, 'Adventure': 1.3, 'Fantasy': 1.2, 'Animation': 1.2, 'Family': 1.2}
    },
    'situation': {
        'solo':    {'Thriller': 1.3, 'Horror': 1.3, 'Drama': 1.2, 'Science Fiction': 1.2, 'Mystery': 1.2},
        'date':    {'Romance': 1.5, 'Comedy': 1.3, 'Drama': 1.2, 'Thriller': 1.1},
        'family':  {'Animation': 1.5, 'Family': 1.5, 'Comedy': 1.3, 'Adventure': 1.3, 'Horror': 0.3, 'Thriller': 0.4},
        'friends': {'Comedy': 1.4, 'Action': 1.3, 'Adventure': 1.3, 'Horror': 1.2}
    },
    'energy': {
        'relaxed':    {'Drama': 1.3, 'Romance': 1.2, 'Documentary': 1.2, 'Comedy': 1.1, 'Action': 0.7},
        'normal':     {'Action': 1.1, 'Comedy': 1.1, 'Drama': 1.1, 'Thriller': 1.1},
        'energised':  {'Action': 1.4, 'Adventure': 1.4, 'Science Fiction': 1.3, 'Thriller': 1.2, 'Drama': 0.8}
    },
    'duration': {
        'short':   {'Comedy': 1.2, 'Animation': 1.2, 'Documentary': 1.1},
        'medium':  {'Drama': 1.1, 'Thriller': 1.1, 'Action': 1.1, 'Romance': 1.1},
        'any':     {}
    }
}

# Runtime limits per duration preference (in minutes)
RUNTIME_LIMITS = {
    'short':  100,
    'medium': 150,
    'any':    99999
}

# ── Auto-detect time of day ──────────────────────────────────
def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'late_night'

def get_day_type():
    return 'weekend' if datetime.now().weekday() >= 5 else 'weekday'

# ── Build context profile ────────────────────────────────────
def build_context_profile(situation='solo', energy='normal', duration='any',
                           time_of_day=None, day_type=None):
    """
    Builds a genre weight profile from context factors.
    time_of_day and day_type are auto-detected if not provided.
    """
    if time_of_day is None:
        time_of_day = get_time_of_day()
    if day_type is None:
        day_type = get_day_type()

    # Combine all weights
    combined = {}

    for factor, value in [
        ('time_of_day', time_of_day),
        ('day_type',    day_type),
        ('situation',   situation),
        ('energy',      energy),
        ('duration',    duration)
    ]:
        weights = CONTEXT_GENRE_WEIGHTS.get(factor, {}).get(value, {})
        for genre, weight in weights.items():
            combined[genre] = combined.get(genre, 1.0) * weight

    return {
        'genre_weights': combined,
        'time_of_day':   time_of_day,
        'day_type':      day_type,
        'situation':     situation,
        'energy':        energy,
        'duration':      duration,
        'runtime_limit': RUNTIME_LIMITS.get(duration, 99999)
    }

# ── Apply context to recommendations ────────────────────────
def apply_context(recommendations, movies_df, context_profile):
    """
    Re-ranks recommendations based on context profile.
    recommendations = list of (title, score) tuples
    Returns re-ranked list with context scores.
    """
    genre_weights  = context_profile['genre_weights']
    runtime_limit  = context_profile['runtime_limit']
    scored         = []

    for title, base_score in recommendations:
        row = movies_df[movies_df['title'] == title]
        if row.empty:
            scored.append((title, base_score, base_score, 0.0))
            continue

        row = row.iloc[0]

        # Genre boosting
        genres = str(row.get('genres', '') or '').split('|')
        genre_multiplier = 1.0
        for genre in genres:
            if genre in genre_weights:
                genre_multiplier *= genre_weights[genre]

        # Runtime filter — soft penalty rather than hard cutoff
        runtime = float(row.get('runtime', 0) or 0)
        runtime_penalty = 1.0
        if runtime > 0 and runtime > runtime_limit:
            over_by = runtime - runtime_limit
            runtime_penalty = max(0.5, 1.0 - (over_by / 120))

        # Context score
        context_score   = base_score * genre_multiplier * runtime_penalty
        context_fit_pct = round(min((genre_multiplier - 0.5) / 1.5 * 100, 100), 1)
        context_fit_pct = max(context_fit_pct, 0)

        scored.append((title, base_score, context_score, context_fit_pct))

    # Re-rank by context score
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored

# ── Generate context summary ─────────────────────────────────
def context_summary(context_profile):
    """Returns a human-readable summary of the current context."""
    time_labels = {
        'morning':    'morning',
        'afternoon':  'afternoon',
        'evening':    'evening',
        'late_night': 'late night'
    }
    situation_labels = {
        'solo': 'watching alone',
        'date': 'on a date',
        'family': 'with family',
        'friends': 'with friends'
    }
    energy_labels = {
        'relaxed':   'feeling relaxed',
        'normal':    'feeling normal',
        'energised': 'feeling energised'
    }
    duration_labels = {
        'short':  'under 100 mins',
        'medium': 'under 150 mins',
        'any':    'any length'
    }

    return (
        f"{time_labels.get(context_profile['time_of_day'], '')} · "
        f"{situation_labels.get(context_profile['situation'], '')} · "
        f"{energy_labels.get(context_profile['energy'], '')} · "
        f"{duration_labels.get(context_profile['duration'], '')}"
    )