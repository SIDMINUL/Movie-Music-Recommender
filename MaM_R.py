import streamlit as st
import pandas as pd
import ast
import requests
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

# ----------------------
# CONFIG
OMDB_API_KEY = "dfb94c1e"

st.set_page_config(page_title="Movies & Music Recommender", layout="wide")
st.title("🎭 Welcome to Recommender Hub")

# Sidebar
st.sidebar.header("⚡ Navigation")
app_type = st.sidebar.radio("Choose mode:", ["🎬 Movie Recommender", "🎵 Music Recommender"])

# ----------------------
# MOVIE RECOMMENDER
if app_type == "🎬 Movie Recommender":

    @st.cache_data
    def load_movie_data():
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        movies = movies.merge(credits, on='title')

        def convert(obj):
            return [i['name'] for i in ast.literal_eval(obj)]

        def convert3(obj):
            return [i['name'] for i in ast.literal_eval(obj)[:3]]

        def fetch_director(obj):
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return [i['name']]
            return []

        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        movies['genres'] = movies['genres'].apply(convert)
        movies['keywords'] = movies['keywords'].apply(convert)
        movies['cast'] = movies['cast'].apply(convert3)
        movies['crew'] = movies['crew'].apply(fetch_director)
        movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
        for col in ['genres', 'keywords', 'cast', 'crew']:
            movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        new_df = movies[['movie_id', 'title', 'tags']]
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
        ps = PorterStemmer()
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        return new_df, similarity

    def fetch_movie_details_omdb(title):
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        try:
            res = requests.get(url, timeout=5)
            data = res.json()
            return {
                'poster_url': data.get('Poster') if data.get('Poster') != 'N/A' else None,
                'plot': data.get('Plot') if data.get('Plot') != 'N/A' else "No description available.",
                'rating': data.get('imdbRating') if data.get('imdbRating') != 'N/A' else "N/A"
            }
        except:
            return {'poster_url': None, 'plot': 'Error loading details.', 'rating': 'N/A'}

    def recommend_movie(movie):
        movie = movie.lower()
        if movie not in new_df['title'].str.lower().values:
            return pd.DataFrame()
        idx = new_df[new_df['title'].str.lower() == movie].index[0]
        dists = list(enumerate(similarity[idx]))
        movie_ids = sorted(dists, reverse=True, key=lambda x: x[1])[1:6]
        return new_df.iloc[[i[0] for i in movie_ids]][['movie_id', 'title']]

    new_df, similarity = load_movie_data()

    selected_movie = st.selectbox("🎬 Choose a movie:", new_df['title'].sort_values().values)

    if st.button("Recommend Movies 🎥"):
        results = recommend_movie(selected_movie)
        if results.empty:
            st.error("❌ Movie not found.")
        else:
            st.subheader("🍿 Recommended Movies:")
            for _, row in results.iterrows():
                title = row['title']
                details = fetch_movie_details_omdb(title)

                col1, col2 = st.columns([1, 3])
                with col1:
                    if details['poster_url']:
                        st.image(details['poster_url'], width=150)
                with col2:
                    st.markdown(f"### {title}")
                    st.write(details['plot'])
                    st.write(f"⭐ IMDb Rating: {details['rating']}")
                st.markdown("---")

# ----------------------
# MUSIC RECOMMENDER
elif app_type == "🎵 Music Recommender":

    @st.cache_data
    def load_music_data():
        df = pd.read_csv('spotify_millsongdata.csv')
        df.drop(columns='link', inplace=True)
        df = df.sample(5000)  # smaller sample for speed
        df['text'] = df['text'].str.lower().replace(r'\n', ' ', regex=True)
        stemmer = PorterStemmer()
        def token(txt):
            tokenized = nltk.word_tokenize(txt)
            return ' '.join([stemmer.stem(word) for word in tokenized])
        df['text'] = df['text'].apply(token)
        tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
        matrix = tfidf.fit_transform(df['text'])
        similarity = cosine_similarity(matrix)
        return df, similarity

    def recommend_song(song_name):
        song_name = song_name.lower()
        matches = df[df['song'].str.lower() == song_name]
        if matches.empty:
            return []
        idx = matches.index[0]
        dist = list(enumerate(similarity[idx]))
        song_ids = sorted(dist, reverse=True, key=lambda x: x[1])[1:6]
        # return (song, artist) pairs
        return [(df.iloc[i[0]].song, df.iloc[i[0]].artist) for i in song_ids]

    def fetch_deezer_preview(song, artist=None):
        try:
            query = f"{song}"
            if artist:
                query += f" {artist}"
            url = f"https://api.deezer.com/search?q={query}"
            res = requests.get(url, timeout=5)
            data = res.json()
            if data.get("data"):
                track = data["data"][0]
                return {
                    "title": track["title"],
                    "artist": track["artist"]["name"],
                    "preview": track.get("preview"),
                    "cover": track["album"]["cover_medium"],
                    "link": track["link"]
                }
            return None
        except Exception:
            return None

    df, similarity = load_music_data()

    selected_song = st.selectbox("🎵 Choose a song:", df['song'].sort_values().unique())

    if st.button("Recommend Songs 🎶"):
        results = recommend_song(selected_song)
        if results:
            st.subheader("🎧 You might also like:")
            for song, artist in results:
                song_data = fetch_deezer_preview(song, artist)

                if song_data:
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(song_data["cover"], width=120)
                        with col2:
                            st.markdown(f"**{song_data['title']}** - {song_data['artist']}")
                            if song_data["preview"]:
                                st.audio(song_data["preview"])
                            st.markdown(f"[🔗 Listen on Deezer]({song_data['link']})")
                    st.markdown("---")
                else:
                    st.markdown(f"- {song} (by {artist})")
        else:
            st.error("❌ Song not found.")