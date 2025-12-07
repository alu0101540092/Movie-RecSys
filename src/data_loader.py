import os
import pandas as pd
import urllib.request
import zipfile
import io

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
DATA_DIR = "datasets"
MOVIES_FILE = os.path.join(DATA_DIR, "ml-32m", "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ml-32m", "ratings.csv")


def ensure_dataset_exists():
    if not os.path.exists(MOVIES_FILE) or not os.path.exists(RATINGS_FILE):
        print("Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        with urllib.request.urlopen(DATASET_URL) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                z.extractall(DATA_DIR)
        print("Dataset downloaded.")


def load_movies():
    ensure_dataset_exists()
    return pd.read_csv(MOVIES_FILE)


def load_ratings():
    ensure_dataset_exists()
    return pd.read_csv(RATINGS_FILE)


def search_movies(query, movies_df):
    if not query:
        return movies_df.head(20)

    # Simple case-insensitive search in title and genres
    mask = movies_df["title"].str.contains(
        query, case=False, na=False
    ) | movies_df["genres"].str.contains(query, case=False, na=False)
    return movies_df[mask]


def get_movie_title(movie_id, movies_df):
    movie = movies_df[movies_df["movieId"] == movie_id]
    if not movie.empty:
        return movie.iloc[0]["title"]
    return "Unknown"
