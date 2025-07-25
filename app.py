import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python", names=["MovieID", "Title", "Genres"])
    data = pd.merge(ratings, movies, on="MovieID")
    return data

data = load_data()

user_movie_matrix = data.pivot_table(index='UserID', columns='Title', values='Rating').fillna(0)
cosine_sim = cosine_similarity(user_movie_matrix.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Choose a movie:", user_movie_matrix.columns)

def recommend_movies(title, top_n=5):
    if title not in cosine_sim_df:
        return []
    sim_scores = cosine_sim_df[title].sort_values(ascending=False)[1:top_n+1]
    return sim_scores.index.tolist()

if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    recommendations = recommend_movies(movie)
    for rec in recommendations:
        st.write("✅", rec)
