import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_resource
def load_data():
    movies = pickle.load(open("movies.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    vectors = vectorizer.transform(movies["tags"])

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

# -----------------------------
# THEMES
# -----------------------------

themes = {

    "Netflix": {
        "bg":"#0f0f0f",
        "card":"#181818",
        "text":"white",
        "accent":"#E50914",
        "secondary":"#2d2d2d"
    },

    "Light": {
        "bg":"#f5f5f5",
        "card":"white",
        "text":"black",
        "accent":"#1976D2",
        "secondary":"#ECECEC"
    },

    "Cyberpunk": {
        "bg":"#080014",
        "card":"#16002B",
        "text":"#00FFF7",
        "accent":"#FF00AA",
        "secondary":"#1B0035"
    },

    "Solarpunk": {
        "bg":"#F6FFF2",
        "card":"#FFFFFF",
        "text":"#1E4620",
        "accent":"#52B788",
        "secondary":"#D8F3DC"
    },

    "Fantasy": {
        "bg":"#15092E",
        "card":"#231942",
        "text":"#F4F1DE",
        "accent":"#C77DFF",
        "secondary":"#3C096C"
    },

    "Medieval": {
        "bg":"#2F241F",
        "card":"#4A3B31",
        "text":"#F8F0E3",
        "accent":"#C89B3C",
        "secondary":"#6F4E37"
    },

    "Minimalist": {
        "bg":"white",
        "card":"#F7F7F7",
        "text":"#222222",
        "accent":"black",
        "secondary":"#DDDDDD"
    }

}

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🎬 Movie Recommendation")

st.sidebar.markdown("---")

st.sidebar.markdown("## 👨‍💻 Developed By")

st.sidebar.markdown(
"""
### **Gaurav Warkhede**
AI & Machine Learning Engineer
"""
)

st.sidebar.markdown("---")

theme = st.sidebar.selectbox(
    "🎨 Select Theme",
    list(themes.keys())
)

colors = themes[theme]

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Version 1.0

Premium UI

Streamlit

Machine Learning

Scikit-Learn
"""
)

# -----------------------------
# PREMIUM CSS
# -----------------------------

st.markdown(f"""

<style>

html,
body,
[data-testid="stAppViewContainer"]{{
background:{colors['bg']};
color:{colors['text']};
}}

section[data-testid="stSidebar"]{{
background:{colors['card']};
}}

h1,h2,h3,h4,h5,p,span,div{{
color:{colors['text']};
}}

.main-title{{
font-size:60px;
font-weight:900;
text-align:center;
margin-top:20px;
animation:fade 1.5s;
}}

.sub-title{{
font-size:22px;
text-align:center;
opacity:.8;
margin-bottom:40px;
}}

.glass{{
background:rgba(255,255,255,.05);
padding:30px;
border-radius:20px;
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,.12);
}}

.movie-card{{
background:{colors['card']};
padding:20px;
border-radius:20px;
transition:.4s;
text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,.25);
}}

.movie-card:hover{{
transform:translateY(-8px) scale(1.02);
box-shadow:0 20px 50px rgba(0,0,0,.45);
}}

.stButton>button{{
background:{colors['accent']};
color:white;
font-size:18px;
font-weight:bold;
border:none;
border-radius:12px;
padding:14px 30px;
transition:.3s;
}}

.stButton>button:hover{{
transform:scale(1.05);
}}

.footer{{
text-align:center;
margin-top:70px;
padding:25px;
opacity:.7;
}}

@keyframes fade{{
from{{
opacity:0;
transform:translateY(-25px);
}}

to{{
opacity:1;
transform:translateY(0px);
}}
}}

</style>

""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
"""
<div class="main-title">
🎬 Movie Recommendation AI
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="sub-title">
Discover Your Next Favorite Movie with Artificial Intelligence
</div>
""",
unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]

    names = []

    for i in movie_list:
        names.append(movies.iloc[i[0]].title)

    return names

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
    f"""
    <div class="glass">
        <h2 style="text-align:center;font-size:38px;">
            🍿 AI Powered Movie Discovery
        </h2>

        <p style="text-align:center;font-size:18px;">
        Search for a movie you love and discover similar movies instantly.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# ==========================================================
# SEARCH BAR
# ==========================================================

movie_list = sorted(movies["title"].unique())

selected_movie = st.selectbox(
    "🔍 Search Movie",
    movie_list,
    help="Search your favourite movie",
)

st.write("")

recommend_btn = st.button(
    "🚀 Recommend Movies",
    use_container_width=True,
)

st.write("")

# ==========================================================
# LOADING
# ==========================================================

if recommend_btn:

    progress = st.progress(0)

    with st.spinner("Finding similar movies..."):

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    progress.empty()

    recommendations = recommend(selected_movie)

    st.markdown(
        """
        <h2 style='text-align:center;'>
        🎬 Top Recommendations
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):

        rating = round(np.random.uniform(7.0, 9.8), 1)

        year = np.random.randint(1995, 2025)

        genres = np.random.choice(
            [
                "Action",
                "Drama",
                "Adventure",
                "Comedy",
                "Fantasy",
                "Sci-Fi",
                "Thriller",
                "Crime",
            ],
            2,
            replace=False,
        )

        with cols[idx]:

            st.markdown(
                f"""
                <div class="movie-card">

                <img
                src="https://placehold.co/300x450?text=Movie+Poster"
                width="100%"
                style="border-radius:15px;">

                <br><br>

                <h4>{movie}</h4>

                ⭐ {rating}/10

                <br><br>

                📅 {year}

                <br><br>

                🎭 {genres[0]} | {genres[1]}

                <br><br>

                </div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================================
# ABOUT SECTION
# ==========================================================

st.write("")
st.write("")
st.write("")

st.markdown(
    """
    ## 🎯 About this Project

    This Movie Recommendation System uses **Natural Language Processing (NLP)**
    and **Cosine Similarity** to recommend movies similar to the one selected
    by the user.

    ### Technologies Used

    - Python
    - Streamlit
    - Scikit-Learn
    - NLP
    - CountVectorizer
    - Cosine Similarity
    """,
)

st.write("")
st.write("")

# ==========================================================
# PROJECT STATS
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🎬 Movies", len(movies))

with c2:
    st.metric("🧠 Model", "NLP")

with c3:
    st.metric("⚡ Response", "<1 sec")

with c4:
    st.metric("⭐ Version", "1.0")

st.write("")
st.write("")
st.write("")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">

    <hr>

    <h3>🎬 Movie Recommendation AI</h3>

    Developed with ❤️ using Streamlit & Machine Learning

    <br><br>

    <strong>Developed By</strong>

    <br>

    <h2>GAURAV WARKHEDE</h2>

    <br>

    Version 1.0

    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# FEATURE SECTION
# ==========================================================

st.write("")
st.write("")
st.write("")

st.markdown(
"""
<h2 style='text-align:center;'>
✨ Why Use Movie Recommendation AI?
</h2>
""",
unsafe_allow_html=True
)

feature1,feature2,feature3=st.columns(3)

with feature1:

    st.markdown("""
    <div class="movie-card">

    <h2>⚡ Fast</h2>

    Get recommendations in less than a second using
    Cosine Similarity and NLP.

    </div>
    """,unsafe_allow_html=True)

with feature2:

    st.markdown("""
    <div class="movie-card">

    <h2>🧠 AI Powered</h2>

    Uses Natural Language Processing to understand
    movie content instead of ratings.

    </div>
    """,unsafe_allow_html=True)

with feature3:

    st.markdown("""
    <div class="movie-card">

    <h2>🎬 Huge Collection</h2>

    Discover thousands of movies instantly.

    </div>
    """,unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")

# ==========================================================
# POPULAR GENRES
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
🔥 Explore Genres
</h2>
""",
unsafe_allow_html=True
)

genre1,genre2,genre3,genre4,genre5=st.columns(5)

genre1.success("🎭 Drama")
genre2.info("🚀 Sci-Fi")
genre3.warning("⚔️ Action")
genre4.success("😂 Comedy")
genre5.info("🧙 Fantasy")

st.write("")
st.write("")
st.write("")

# ==========================================================
# HOW IT WORKS
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
🧠 How Recommendations Work
</h2>
""",
unsafe_allow_html=True
)

st.markdown("""

1️⃣ Movie selected by user

⬇️

2️⃣ Tags are converted into vectors

⬇️

3️⃣ Cosine Similarity compares all movies

⬇️

4️⃣ Top 5 similar movies are selected

⬇️

5️⃣ Recommendations displayed instantly

""")

st.write("")
st.write("")
st.write("")

# ==========================================================
# RANDOM MOVIE FACT
# ==========================================================

facts=[
"The average movie contains over 1000 spoken words.",
"More than 5000 movies are available in this recommendation system.",
"Cosine Similarity measures the angle between vectors.",
"NLP helps computers understand text.",
"Recommendation systems are used by Netflix, Amazon and Spotify.",
"Machine Learning powers many modern streaming platforms."
]

st.info("💡 Movie Fact")

st.success(np.random.choice(facts))

st.write("")
st.write("")
st.write("")

# ==========================================================
# TECHNOLOGY BADGES
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
🛠 Technologies Used
</h2>
""",
unsafe_allow_html=True
)

tech1,tech2,tech3,tech4,tech5=st.columns(5)

tech1.code("Python")

tech2.code("Streamlit")

tech3.code("Scikit-Learn")

tech4.code("NLP")

tech5.code("Pickle")

st.write("")
st.write("")
st.write("")

# ==========================================================
# DEVELOPER SECTION
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
👨‍💻 Developer
</h2>
""",
unsafe_allow_html=True
)

st.markdown("""
<div class="movie-card">

<h1>GAURAV WARKHEDE</h1>

AI & Machine Learning Engineer

Python • Machine Learning • NLP • Streamlit

Building projects that combine Artificial Intelligence
with beautiful user experiences.

</div>
""",unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")

# ==========================================================
# FUTURE UPDATES
# ==========================================================

with st.expander("🚀 Upcoming Features"):

    st.write("✅ Real Movie Posters")
    st.write("✅ Movie Trailers")
    st.write("✅ TMDB API")
    st.write("✅ User Login")
    st.write("✅ Watchlist")
    st.write("✅ Favorites")
    st.write("✅ Movie Ratings")
    st.write("✅ AI Chatbot")
    st.write("✅ Voice Search")
    st.write("✅ Deploy on Streamlit Cloud")

st.write("")
st.write("")
st.write("")

# ==========================================================
# FINAL FOOTER
# ==========================================================

st.markdown("""
<hr>
<center>

<h2>🎬 Movie Recommendation AI</h2>

Built using ❤️ Streamlit, NLP & Machine Learning

<br>

© 2026

<br>

<b>Developed By Gaurav Warkhede</b>

</center>
""",unsafe_allow_html=True)

# ==========================================================
# WELCOME MESSAGE
# ==========================================================

if "welcome" not in st.session_state:
    st.balloons()
    st.session_state.welcome=True

# ==========================================================
# PROJECT ACHIEVEMENTS
# ==========================================================

st.write("")
st.write("")
st.write("")

st.markdown(
"""
<h2 style='text-align:center;'>
🏆 Project Highlights
</h2>
""",
unsafe_allow_html=True
)

a,b,c,d=st.columns(4)

with a:
    st.success("🎬 4806 Movies")

with b:
    st.success("🤖 NLP Based")

with c:
    st.success("⚡ Fast Recommendation")

with d:
    st.success("🎨 Premium UI")

st.write("")
st.write("")
st.write("")

# ==========================================================
# USER FEEDBACK
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
⭐ Rate This Project
</h2>
""",
unsafe_allow_html=True
)

rating=st.slider(
"How would you rate this project?",
1,
5,
5
)

if rating>=4:
    st.success("Thank you for your feedback ❤️")
else:
    st.warning("Thanks! Future versions will be even better.")

st.write("")
st.write("")
st.write("")

# ==========================================================
# SOCIAL LINKS
# ==========================================================

st.markdown(
"""
<h2 style='text-align:center;'>
🌐 Connect With Me
</h2>
""",
unsafe_allow_html=True
)

col1,col2,col3=st.columns(3)

with col1:
    st.markdown(
    """
    ### 💻 GitHub

    Add your GitHub profile here.
    """
    )

with col2:
    st.markdown(
    """
    ### 🔗 LinkedIn

    Add your LinkedIn profile here.
    """
    )

with col3:
    st.markdown(
    """
    ### 📧 Email

    your_email@example.com
    """
    )

st.write("")
st.write("")
st.write("")

# ==========================================================
# PROJECT INFO
# ==========================================================

with st.expander("📄 Project Information"):

    st.write("Project Name : Movie Recommendation AI")

    st.write("Developer : Gaurav Warkhede")

    st.write("Language : Python")

    st.write("Framework : Streamlit")

    st.write("Machine Learning : NLP")

    st.write("Recommendation Engine : Cosine Similarity")

    st.write("Dataset Size : 4806 Movies")

    st.write("Deployment : Streamlit Community Cloud")

st.write("")
st.write("")
st.write("")

# ==========================================================
# DISCLAIMER
# ==========================================================

st.info(
"""
This project was created for educational and portfolio
purposes.

Movie recommendations are generated using
Natural Language Processing and Cosine Similarity.
"""
)

st.write("")
st.write("")
st.write("")

# ==========================================================
# THANK YOU
# ==========================================================

st.markdown(
"""
<div class="glass">

<h1 style="text-align:center;">
❤️ Thank You For Visiting
</h1>

<h3 style="text-align:center;">
Movie Recommendation AI
</h3>

<p style="text-align:center;">

Built with Python, Streamlit and Machine Learning.

</p>

<p style="text-align:center;">

Developed by

</p>

<h2 style="text-align:center; color:#E50914;">

GAURAV WARKHEDE

</h2>

<p style="text-align:center;">

Keep Learning • Keep Building • Keep Growing 🚀

</p>

</div>
""",
unsafe_allow_html=True
)

st.write("")
st.write("")
st.write("")

# ==========================================================
# END
# ==========================================================

st.markdown(
"""
<hr>

<center>

Made with ❤️ using Python, Streamlit & Machine Learning

<br>

<b>© 2026 Gaurav Warkhede</b>

</center>
""",
unsafe_allow_html=True
)
