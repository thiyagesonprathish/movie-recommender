from recommender import (hybrid_recommend, get_popular_movies, search_titles,
                         get_movie_details, ALL_TITLES,
                         collab_similarity_df, content_similarity_df,
                         mood_recommend, MOOD_MAP, get_movie_dna)
from recommender import (hybrid_recommend, get_popular_movies, search_titles,
                         get_movie_details, ALL_TITLES,
                         collab_similarity_df, content_similarity_df,
                         mood_recommend, MOOD_MAP)
from flask import Flask, render_template, request, redirect, url_for, session, flash
from recommender import (hybrid_recommend, get_popular_movies, search_titles,
                         get_movie_details, ALL_TITLES,
                         collab_similarity_df, content_similarity_df)
from recommender import (hybrid_recommend, get_popular_movies, search_titles,
                         get_movie_details, ALL_TITLES,
                         collab_similarity_df, content_similarity_df,
                         mood_recommend, MOOD_MAP, get_movie_dna,
                         get_trending_movies, get_hidden_gems,
                         COLLAB_SIM_DF, CONTENT_SIM_DF, MOVIES_DF, CLEAN_DATA,
                         COLLAB_WEIGHT, CONTENT_WEIGHT)
from context_engine import (build_context_profile, apply_context,
                             context_summary, get_time_of_day, get_day_type)

from confidence_engine import compute_confidence_batch
import json
import os
from datetime import date

app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_2024'

USERS_FILE   = 'users.json'
GUEST_LIMIT  = 10

# ── User storage ────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# ── Guest search limit helpers ───────────────────────────────
def get_guest_searches():
    """Get how many searches guest has done today."""
    today = str(date.today())
    if session.get('search_date') != today:
        session['search_date']  = today
        session['search_count'] = 0
    return session.get('search_count', 0)

def increment_guest_searches():
    """Increment guest search count."""
    today = str(date.today())
    if session.get('search_date') != today:
        session['search_date']  = today
        session['search_count'] = 0
    session['search_count'] = session.get('search_count', 0) + 1

def remaining_searches():
    """How many searches remain for guest today."""
    if 'user' in session:
        return None  # unlimited for logged in users
    return max(0, GUEST_LIMIT - get_guest_searches())

# ── Context processor — available in all templates ───────────
@app.context_processor
def inject_search_info():
    return {
        'remaining': remaining_searches(),
        'guest_limit': GUEST_LIMIT,
        'is_logged_in': 'user' in session
    }

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    popular_raw = get_popular_movies(10)
    popular = []
    for title, rating in popular_raw:
        details = get_movie_details(title)
        popular.append({
            'title':  title,
            'rating': rating,
            'poster': details['poster'],
            'genres': details['genres']
        })
    return render_template('index.html', popular=popular)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    movie = request.args.get('movie', '').strip()

    if movie:
        if 'user' not in session:
            if get_guest_searches() >= GUEST_LIMIT:
                return render_template('limit_reached.html')
            increment_guest_searches()

        recs             = hybrid_recommend(movie, n=10)
    searched_details = get_movie_details(movie)
    movie_dna        = get_movie_dna(movie)

    # Compute confidence scores for all recommendations
    confidence_data  = compute_confidence_batch(
        movie,
        recs,
        COLLAB_SIM_DF,
        CONTENT_SIM_DF,
        CLEAN_DATA,
        MOVIES_DF,
        COLLAB_WEIGHT,
        CONTENT_WEIGHT
    )

    recs_with_details = []
    for i, (title, score) in enumerate(recs):
        details = get_movie_details(title)
        conf    = confidence_data[i]
        recs_with_details.append({
            'title':           title,
            'score':           score,
            'poster':          details['poster'],
            'overview':        details['overview'],
            'rating':          details['rating'],
            'year':            details['year'],
            'genres':          details['genres'],
            'collab_score':    conf['collab_score'],
            'content_score':   conf['content_score'],
            'popularity_score':conf['popularity_score'],
            'novelty_score':   conf['novelty_score'],
            'genre_overlap':   conf['genre_overlap'],
            'user_count':      conf['collab_user_count'],
            'reason':          conf['reason']
        })

    return render_template(
        'results.html',
        movie=movie,
        searched_details=searched_details,
        recommendations=recs_with_details,
        movie_dna=movie_dna,
        logged_in='user' in session,
        username=session.get('user', '')
    )

@app.route('/compare')
def compare():
    query = request.args.get('q', '').strip()
    movie = request.args.get('movie', '').strip()

    if movie:
        # Check guest limit
        if 'user' not in session:
            if get_guest_searches() >= GUEST_LIMIT:
                return render_template('limit_reached.html')
            increment_guest_searches()

        # Collaborative
        collab_recs = []
        if movie in collab_similarity_df.columns:
            collab_scores = (
                collab_similarity_df[movie]
                .drop(movie)
                .sort_values(ascending=False)
                .head(5)
            )
            for title, score in collab_scores.items():
                details = get_movie_details(title)
                collab_recs.append({
                    'title':  title,
                    'score':  round(float(score), 4),
                    'poster': details['poster'],
                    'rating': details['rating'],
                    'year':   details['year']
                })

        # Content
        content_recs = []
        if movie in content_similarity_df.columns:
            content_scores = (
                content_similarity_df[movie]
                .drop(movie)
                .sort_values(ascending=False)
                .head(5)
            )
            for title, score in content_scores.items():
                details = get_movie_details(title)
                content_recs.append({
                    'title':  title,
                    'score':  round(float(score), 4),
                    'poster': details['poster'],
                    'rating': details['rating'],
                    'year':   details['year']
                })

        # Hybrid
        hybrid_recs = []
        for title, score in hybrid_recommend(movie, n=5):
            details = get_movie_details(title)
            hybrid_recs.append({
                'title':  title,
                'score':  score,
                'poster': details['poster'],
                'rating': details['rating'],
                'year':   details['year']
            })

        searched_details = get_movie_details(movie)

        return render_template(
            'compare.html',
            movie=movie,
            searched_details=searched_details,
            collab_recs=collab_recs,
            content_recs=content_recs,
            hybrid_recs=hybrid_recs
        )

    suggestions = search_titles(query) if query else []
    return render_template('compare.html',
                           movie=None,
                           suggestions=suggestions,
                           query=query)

@app.route('/mood')
def mood():
    selected_mood = request.args.get('mood', '').strip()

    if selected_mood and selected_mood in MOOD_MAP:
        # Check guest limit
        if 'user' not in session:
            if get_guest_searches() >= GUEST_LIMIT:
                return render_template('limit_reached.html')
            increment_guest_searches()

        results, mood_info = mood_recommend(selected_mood, n=12)

        # Fetch TMDB details for each movie
        movies_with_details = []
        for item in results:
            details = get_movie_details(item['title'])
            movies_with_details.append({
                'title':      item['title'],
                'avg_rating': round(item['avg_rating'], 2),
                'genres':     item['genres'],
                'poster':     details['poster'],
                'overview':   details['overview'],
                'tmdb_rating': details['rating'],
                'year':       details['year'],
                'tmdb_genres': details['genres']
            })

        return render_template(
            'mood.html',
            selected_mood=selected_mood,
            mood_info=mood_info,
            movies=movies_with_details,
            all_moods=MOOD_MAP
        )

    return render_template('mood.html',
                           selected_mood=None,
                           mood_info=None,
                           movies=[],
                           all_moods=MOOD_MAP)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users    = load_users()

        if username in users:
            flash('Username already exists. Please choose another.', 'error')
        elif len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
        elif len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
        else:
            users[username] = {'password': password, 'history': []}
            save_users(users)
            session['user'] = username
            flash(f'Welcome, {username}! You now have unlimited searches.', 'success')
            return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        users    = load_users()

        if username in users and users[username]['password'] == password:
            session['user'] = username
            flash(f'Welcome back, {username}! Unlimited searches restored.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect username or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out. You have 10 free searches per day as a guest.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/context', methods=['GET', 'POST'])
def context():
    if 'user' not in session:
        if get_guest_searches() >= GUEST_LIMIT:
            return render_template('limit_reached.html')

    movie       = request.args.get('movie', '').strip()
    situation   = request.form.get('situation',  request.args.get('situation',  'solo'))
    energy      = request.form.get('energy',     request.args.get('energy',     'normal'))
    duration    = request.form.get('duration',   request.args.get('duration',   'any'))
    time_of_day = request.form.get('time_of_day',request.args.get('time_of_day', get_time_of_day()))
    day_type    = get_day_type()

    context_profile = build_context_profile(
        situation=situation,
        energy=energy,
        duration=duration,
        time_of_day=time_of_day,
        day_type=day_type
    )

    ctx_summary = context_summary(context_profile)

    recs_with_details = []

    if movie:
        if 'user' not in session:
            increment_guest_searches()

        recs     = hybrid_recommend(movie, n=15)
        reranked = apply_context(recs, MOVIES_DF, context_profile)

        for title, base_score, ctx_score, ctx_fit in reranked[:10]:
            details = get_movie_details(title)
            recs_with_details.append({
                'title':       title,
                'base_score':  round(base_score * 100),
                'ctx_score':   round(ctx_score * 100, 1),
                'ctx_fit':     ctx_fit,
                'poster':      details['poster'],
                'overview':    details['overview'],
                'rating':      details['rating'],
                'year':        details['year'],
                'genres':      details['genres']
            })

    return render_template(
        'context.html',
        movie=movie,
        situation=situation,
        energy=energy,
        duration=duration,
        time_of_day=time_of_day,
        ctx_summary=ctx_summary,
        recommendations=recs_with_details,
        auto_time=get_time_of_day(),
        suggestions=search_titles(movie) if movie and not recs_with_details else []
    )