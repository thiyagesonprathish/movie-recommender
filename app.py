from recommender import hybrid_recommend, get_popular_movies, search_titles, get_movie_details, ALL_TITLES
from flask import Flask, render_template, request, redirect, url_for, session, flash
from recommender import hybrid_recommend, get_popular_movies, search_titles, ALL_TITLES
import json
import os

app = Flask(__name__)
app.secret_key = 'movie_recommender_secret_2024'

# ── User storage (JSON file) ────────────────────────────────
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# ── Routes ──────────────────────────────────────────────────
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
        recs = hybrid_recommend(movie, n=10)

        # Fetch TMDB details for searched movie
        searched_details = get_movie_details(movie)

        # Fetch TMDB details for each recommendation
        recs_with_details = []
        for title, score in recs:
            details = get_movie_details(title)
            recs_with_details.append({
                'title':    title,
                'score':    score,
                'poster':   details['poster'],
                'overview': details['overview'],
                'rating':   details['rating'],
                'year':     details['year'],
                'genres':   details['genres']
            })

        return render_template(
            'results.html',
            movie=movie,
            searched_details=searched_details,
            recommendations=recs_with_details,
            logged_in='user' in session,
            username=session.get('user', '')
        )

    suggestions = search_titles(query) if query else []
    return render_template('index.html',
                           popular=get_popular_movies(10),
                           suggestions=suggestions,
                           query=query)

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
            flash(f'Welcome, {username}! Account created.', 'success')
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
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect username or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)