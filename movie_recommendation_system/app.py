# Movie Recommendation System (Streamlit + Content & Collaborative Filtering)

# Install Required Libraries before running (if not already):
# pip install -r requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# Load Data
def load_data():
    movies = pd.read_csv("movies.csv")  # contains movieId, title, genres
    ratings = pd.read_csv("ratings.csv")  # contains userId, movieId, rating
    return movies, ratings

# Content-Based Filtering
@st.cache_data
def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title'].str.lower())
    return cosine_sim, indices

def recommend_content_based(title, cosine_sim, indices, movies):
    idx = indices.get(title.lower())
    if idx is None:
        return ["Movie not found"]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Collaborative Filtering
@st.cache_data
def train_collaborative_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model, trainset

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def recommend_collaborative(user_id, model, trainset, movies):
    all_movie_ids = movies['movieId'].tolist()
    rated = set(trainset.ur[trainset.to_inner_uid(user_id)]) if user_id in trainset._raw2inner_id_users else set()
    predictions = [model.predict(user_id, iid) for iid in all_movie_ids if iid not in rated]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = [movies[movies['movieId'] == pred.iid]['title'].values[0] for pred in predictions[:10]]
    return top_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movies, ratings = load_data()
cosine_sim, indices = build_content_model(movies)
model, trainset = train_collaborative_model(ratings)

st.sidebar.header("Choose Recommendation Type")
rec_type = st.sidebar.radio("", ["Content-Based", "Collaborative"])

if rec_type == "Content-Based":
    movie_name = st.text_input("Enter a movie title (e.g., Toy Story):")
    if movie_name:
        results = recommend_content_based(movie_name, cosine_sim, indices, movies)
        st.subheader("Recommended Movies:")
        for r in results:
            st.write("-", r)

elif rec_type == "Collaborative":
    user_id = st.number_input("Enter User ID (1 to 600)", min_value=1, max_value=600, step=1)
    if st.button("Get Recommendations"):
        try:
            recommendations = recommend_collaborative(int(user_id), model, trainset, movies)
            st.subheader("Recommended Movies:")
            for r in recommendations:
                st.write("-", r)
        except:
            st.write("User ID not found in dataset.")
