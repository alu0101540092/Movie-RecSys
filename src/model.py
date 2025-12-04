import os
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from src.data_loader import load_ratings, load_movies
from src.database import get_user_ratings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_model.pkl")


def train_model():
    print("Training SVD model...")
    ratings_df = load_ratings()

    # Use the Reader object to parse the dataframe
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_df[["userId", "movieId", "rating"]], reader
    )

    # Build full trainset
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(algo, f)
    print("Model trained and saved.")
    return algo


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "algo" in data:
                return data["algo"]
            return data
    else:
        return train_model()


def get_recommendations(user_id, n=10):
    algo = load_model()
    movies_df = load_movies()

    # Get all movie IDs
    all_movie_ids = movies_df["movieId"].unique()

    # Get movies already rated by user (from DB + potentially dataset if we mapped users,
    # but for this prototype new users are in DB only)
    user_ratings_df = get_user_ratings(user_id)
    rated_movie_ids = (
        set(user_ratings_df["movie_id"].values)
        if not user_ratings_df.empty
        else set()
    )

    # Predict for unrated movies
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            est = algo.predict(user_id, movie_id).est
            predictions.append((movie_id, est))

    # Sort by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = predictions[:n]

    # Enrich with movie details
    results = []
    for movie_id, score in top_n:
        title = movies_df[movies_df["movieId"] == movie_id].iloc[0]["title"]
        genres = movies_df[movies_df["movieId"] == movie_id].iloc[0]["genres"]
        results.append(
            {
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "score": score,
            }
        )

    return results
