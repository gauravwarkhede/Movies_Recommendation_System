import streamlit as st
import pandas as pd
import pickle
import requests

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Movie Recommendation AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# TMDB API KEY
# Replace this with your own key
# ==========================================================

API_KEY = "a4dac9957dc0d8f5a743d826a47d4015"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_resource
def load_model():

    with open("movies.pkl", "rb") as file:
        movies = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    vectors = vectorizer.transform(movies["tags"])

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_model()

# ==========================================================
# FETCH MOVIE DETAILS FROM TMDB
# ==========================================================

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_name):

    url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={API_KEY}&query={movie_name}"
    )

    try:

        response = requests.get(url, timeout=10)

        data = response.json()

        if not data.get("results"):

            return {
                "poster": "https://placehold.co/500x750?text=No+Poster",
                "rating": "N/A",
                "year": "----",
                "overview": "Overview not available.",
                "language": "-",
                "popularity": "-",
                "genres": "Unknown"
            }

        movie = data["results"][0]

        poster_path = movie.get("poster_path")

        if poster_path:
            poster = IMAGE_BASE_URL + poster_path
        else:
            poster = "https://placehold.co/500x750?text=No+Poster"

        release_date = movie.get("release_date", "")

        if release_date:
            year = release_date[:4]
        else:
            year = "----"

        genre_map = {

            28: "Action",
            12: "Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            14: "Fantasy",
            36: "History",
            27: "Horror",
            10402: "Music",
            9648: "Mystery",
            10749: "Romance",
            878: "Sci-Fi",
            53: "Thriller",
            10752: "War"

        }

        genres = []

        for gid in movie.get("genre_ids", []):

            if gid in genre_map:
                genres.append(genre_map[gid])

        return {

            "poster": poster,
            "rating": round(movie.get("vote_average", 0), 1),
            "year": year,
            "overview": movie.get(
                "overview",
                "Overview not available."
            ),
            "language": movie.get(
                "original_language",
                "-"
            ).upper(),
            "popularity": round(
                movie.get("popularity", 0),
                1
            ),
            "genres": ", ".join(genres)

        }

    except Exception:

        return {

            "poster": "https://placehold.co/500x750?text=Error",
            "rating": "-",
            "year": "-",
            "overview": "Unable to fetch movie details.",
            "language": "-",
            "popularity": "-",
            "genres": "-"

        }

# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================

def recommend(movie):

    index = movies[movies["title"] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(

        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]

    )[1:7]

    recommendations = []

    for item in movie_list:

        title = movies.iloc[item[0]].title

        info = fetch_movie_details(title)

        recommendations.append({

            "title": title,
            "poster": info["poster"],
            "rating": info["rating"],
            "year": info["year"],
            "overview": info["overview"],
            "genres": info["genres"],
            "language": info["language"],
            "popularity": info["popularity"]

        })

    return recommendations

# ==========================================================
# THEME OPTIONS
# ==========================================================

THEMES = {

    "Netflix": {

        "background": "#0B0B0B",
        "card": "#171717",
        "text": "#FFFFFF",
        "accent": "#E50914"

    },

    "Dark Blue": {

        "background": "#06141B",
        "card": "#11212D",
        "text": "#FFFFFF",
        "accent": "#4DA8DA"

    },

    "Cyberpunk": {

        "background": "#0A001F",
        "card": "#1A0038",
        "text": "#FFFFFF",
        "accent": "#FF00AA"

    },

    "Light": {

        "background": "#F5F5F5",
        "card": "#FFFFFF",
        "text": "#000000",
        "accent": "#2563EB"

    }

}

selected_theme = st.sidebar.selectbox(

    "🎨 Theme",

    list(THEMES.keys())

)

theme = THEMES[selected_theme]
# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.title("🎬 Movie Recommendation AI")

st.sidebar.markdown(
"""
Discover amazing movies using
AI-powered recommendations.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset")

st.sidebar.metric(
    "Movies",
    f"{len(movies):,}"
)

st.sidebar.metric(
    "Recommendations",
    "6"
)

st.sidebar.metric(
    "Algorithm",
    "Cosine Similarity"
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙ Tech Stack")

st.sidebar.markdown("""
- Python
- Streamlit
- NLP
- CountVectorizer
- Scikit-Learn
- TMDB API
""")

st.sidebar.markdown("---")

st.sidebar.success(
"""
👨‍💻 Developed by

**Gaurav Warkhede**
"""
)

# ==========================================================
# CSS
# ==========================================================

st.markdown(
f"""
<style>

.stApp{{
background:{theme['background']};
color:{theme['text']};
}}

header{{
visibility:hidden;
}}

footer{{
visibility:hidden;
}}

section[data-testid="stSidebar"]{{
background:{theme['card']};
}}

.main-title{{
font-size:60px;
font-weight:800;
text-align:center;
margin-top:10px;
color:{theme['text']};
}}

.sub-title{{
text-align:center;
font-size:22px;
opacity:0.8;
margin-bottom:30px;
}}

.hero{{
background:linear-gradient(
135deg,
{theme['accent']},
{theme['card']}
);

padding:45px;
border-radius:22px;
margin-bottom:30px;

box-shadow:0px 12px 35px rgba(0,0,0,.35);
}}

.hero h1{{
color:white;
font-size:52px;
font-weight:800;
}}

.hero p{{
color:white;
font-size:19px;
}}

.stats{{
background:{theme['card']};
padding:22px;
border-radius:18px;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.25);
}}

.stats h2{{
color:{theme['accent']};
}}

.stats h4{{
color:{theme['text']};
}}

.movie-card{{
background:{theme['card']};
padding:15px;
border-radius:18px;
transition:0.3s;
}}

.movie-card:hover{{
transform:translateY(-8px);
}}

.stButton>button{{
width:100%;
background:{theme['accent']};
color:white;
border:none;
padding:15px;
font-size:18px;
font-weight:bold;
border-radius:12px;
}}

.stButton>button:hover{{
transform:scale(1.02);
}}

img{{
border-radius:15px;
}}

</style>
""",
unsafe_allow_html=True
)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
"""
<div class="hero">

<h1>🎬 Movie Recommendation AI</h1>

<p>

Find movies you'll love using
Natural Language Processing,
Machine Learning and TMDB.

</p>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# DASHBOARD
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
    f"""
    <div class="stats">
    <h2>{len(movies)}</h2>
    <h4>Total Movies</h4>
    </div>
    """,
    unsafe_allow_html=True
    )

with c2:

    st.markdown(
    """
    <div class="stats">
    <h2>5000</h2>
    <h4>Vocabulary Size</h4>
    </div>
    """,
    unsafe_allow_html=True
    )

with c3:

    st.markdown(
    """
    <div class="stats">
    <h2>6</h2>
    <h4>Recommendations</h4>
    </div>
    """,
    unsafe_allow_html=True
    )

with c4:

    st.markdown(
    """
    <div class="stats">
    <h2>AI</h2>
    <h4>Powered</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ==========================================================
# SEARCH SECTION
# ==========================================================

st.markdown(
"## 🔍 Search Your Favourite Movie"
)

selected_movie = st.selectbox(

    "",

    sorted(movies["title"].unique())

)

recommend_button = st.button(
    "🎥 Recommend Movies"
)

st.write("")
# ==========================================================
# RECOMMENDATION SECTION
# ==========================================================

if recommend_button:

    with st.spinner("Finding the best movies for you..."):

        recommendations = recommend(selected_movie)

    st.success(f"Top recommendations based on **{selected_movie}**")

    st.write("")

    cols = st.columns(3)

    for index, movie in enumerate(recommendations):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="movie-card">
                """,
                unsafe_allow_html=True
            )

            st.image(
                movie["poster"],
                use_container_width=True
            )

            st.markdown(
                f"### 🎬 {movie['title']}"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "⭐ Rating",
                    movie["rating"]
                )

            with col2:
                st.metric(
                    "📅 Year",
                    movie["year"]
                )

            st.markdown(
                f"**🎭 Genre:** {movie['genres']}"
            )

            st.markdown(
                f"**🌍 Language:** {movie['language']}"
            )

            st.markdown(
                f"**🔥 Popularity:** {movie['popularity']}"
            )

            with st.expander("📖 Overview"):

                st.write(movie["overview"])

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

    st.write("")
    st.divider()

# ==========================================================
# FEATURED MOVIES
# ==========================================================

st.markdown("## 🍿 Popular Movies To Try")

featured = [
    "Avatar",
    "The Dark Knight",
    "Inception",
    "Interstellar",
    "Titanic",
    "The Avengers"
]

feature_cols = st.columns(3)

for i, name in enumerate(featured):

    info = fetch_movie_details(name)

    with feature_cols[i % 3]:

        st.image(
            info["poster"],
            use_container_width=True
        )

        st.markdown(f"### {name}")

        st.write(f"⭐ {info['rating']}")

        st.write(f"📅 {info['year']}")

        st.write(info["genres"])

st.write("")
st.divider()

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("## 🚀 How It Works")

step1, step2, step3 = st.columns(3)

with step1:

    st.info("""
### 1️⃣ NLP

Movie overviews, genres,
cast and keywords are
combined into one text.
""")

with step2:

    st.info("""
### 2️⃣ Vectorization

CountVectorizer converts
text into numerical vectors.
""")

with step3:

    st.info("""
### 3️⃣ Recommendation

Cosine Similarity finds
movies with similar content.
""")

st.write("")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.divider()

st.markdown("# 📖 About This Project")

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("""
This **Movie Recommendation AI** recommends movies based on their similarity using
Natural Language Processing.

### Machine Learning Pipeline

- Merge Movie & Credits datasets
- Data Cleaning
- Feature Engineering
- Stemming
- CountVectorizer
- Cosine Similarity
- TMDB API Integration
- Streamlit Deployment

The recommendation engine analyzes the movie's **overview, genres, keywords, cast, and director**
to find similar movies.
""")

with col2:

    st.metric("Dataset Size", f"{len(movies):,}")

    st.metric("Recommendation Model", "Content-Based")

    st.metric("Framework", "Streamlit")

    st.metric("Language", "Python")

# ==========================================================
# PROJECT FEATURES
# ==========================================================

st.divider()

st.markdown("# ✨ Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.success("""
✅ NLP

✅ CountVectorizer

✅ Cosine Similarity

✅ Streamlit
""")

with feature2:

    st.success("""
✅ Movie Posters

✅ Ratings

✅ Genres

✅ Overview
""")

with feature3:

    st.success("""
✅ TMDB API

✅ Multiple Themes

✅ Fast Caching

✅ Responsive UI
""")

# ==========================================================
# MOVIE RECOMMENDATION TIPS
# ==========================================================

st.divider()

st.markdown("# 💡 Recommendation Tips")

st.info("""
• Search for any famous movie.

• Movies with rich descriptions usually produce better recommendations.

• If posters don't appear, check your TMDB API key.

• The first recommendation is intentionally skipped because it is the selected movie itself.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

current_year = "2026"

st.markdown(
f"""
<div style="text-align:center;padding:20px;">

<h3>🎬 Movie Recommendation AI</h3>

<p>
Built using ❤️ with
<b>Python</b>,
<b>Streamlit</b>,
<b>Scikit-Learn</b>,
<b>NLP</b>,
and
<b>TMDB API</b>
</p>

<p>
Developed by <b>Gaurav Warkhede</b>
</p>

<p>
© {current_year} All Rights Reserved.
</p>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# BALLOONS
# ==========================================================

if recommend_button:
    st.balloons()




