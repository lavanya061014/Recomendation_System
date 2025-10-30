import streamlit as st
import pandas as pd
import numpy as np


#-- Step 1: Load and preprocess data ---#


# Load the data

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Filter to top 500 movies and top 1000 users
top_movie_ids = ratings['movieId'].value_counts().head(500).index
top_user_ids = ratings['userId'].value_counts().head(1000).index

filtered_ratings = ratings[
    (ratings['movieId'].isin(top_movie_ids)) &
    (ratings['userId'].isin(top_user_ids))
]

# Merge ratings with movie titles
movie_ratings = pd.merge(filtered_ratings, movies, on="movieId")

# Create user-movie matrix
user_movie_matrix = movie_ratings.pivot_table(index="userId", columns="title", values="rating")
user_movie_matrix = user_movie_matrix.astype(np.float32)


#--- Step 2: Recommendation function ---#


def recommend_movies(movie_name, matrix=user_movie_matrix, top_n=10, min_ratings=35):
    if movie_name not in matrix.columns:
        return pd.DataFrame({'Movie Title': [], 'Similarity': []})
    
    # Calculates Pearson correlation
    sim_scores = matrix.corrwith(matrix[movie_name])
    sim_scores = sim_scores.dropna()

    # Filter out movies with too few ratings
    movie_counts = matrix.count()
    sim_scores = sim_scores[movie_counts[sim_scores.index] > min_ratings]

    # Sort and remove the selected movie itself
    sim_scores = sim_scores.sort_values(ascending=False)
    sim_scores = sim_scores[sim_scores.index != movie_name]

    # Format the result
    result = sim_scores.head(top_n).reset_index()
    result.columns = ['Movie Title', 'Similarity']
    
    # Converts similarity into percentage
    result['Similarity'] = result['Similarity']*100
    return result


#--- Step 3: Streamlit App ---#

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"

if "recs" not in st.session_state:
    st.session_state.recs = None

# Custom CSS for card styling (smaller sizes)
st.markdown("""
    <style>
        .card {
            background-color: #262730;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.2);
            margin-bottom: 10px;
            color: white;
            font-size: 14px;
        }
        .movie-title {
            font-size: 16px;
            font-weight: bold;
        }
        .similarity {
            font-size: 13px;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Make the layout narrower
st.set_page_config(layout="centered")


#--- Page 1: Movie Selection ---#

if st.session_state.page == "home":
    st.title("🎬 Movie Recommendation System")
    st.markdown("Choose a movie to get similar suggestions.")

    movie_list = user_movie_matrix.columns.sort_values().tolist()
    selected_movie = st.selectbox(" Select a movie here  ", movie_list)

    
    top_n = st.slider("Number of recommendations:", 1, 10, 10)
    min_ratings = st.slider("Minimum ratings required:", 1, 50, 35)

    if st.button("Get Recommendations") and selected_movie:
        recs = recommend_movies(selected_movie, top_n=top_n, min_ratings=min_ratings)
        st.session_state.recs = recs
        st.session_state.selected_movie = selected_movie
        st.session_state.page = "results"
        st.rerun()


#--- Page 2: Recommendations ---#

elif st.session_state.page == "results":
    st.title("🎬 Recommended Movies")

    recs = st.session_state.recs
    if recs is None or recs.empty:
        st.warning("⚠️ Not enough data for this movie. Try another.")
    else:
        st.success("✅ Here are your recommendations:")

        # Layout: recommendations (left) and bar chart (right)
        col1, col2 = st.columns([1, 1])

        with col1:
            for _, row in recs.iterrows():
                st.markdown(f"""
                    <div class="card">
                        <div class="movie-title">{row['Movie Title']}</div>
                        <div class="similarity">Similarity: {row['Similarity']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.bar_chart(recs.set_index("Movie Title")["Similarity"])

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

