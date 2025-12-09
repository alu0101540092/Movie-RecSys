import os
import pandas as pd
import urllib.request
import zipfile
import io
import sys

# Add project root to path to import scripts
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.file_manager import join_file

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
DATA_DIR = "datasets"
MOVIES_FILE = os.path.join(DATA_DIR, "ml-32m", "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ml-32m", "ratings.csv")
TAGS_FILE = os.path.join(DATA_DIR, "ml-32m", "tags.csv")


def ensure_dataset_exists():
    """
    Ensures that the MovieLens dataset exists locally.

    It first attempts to reconstruct larger files from parts (if applicable).
    Then it checks if the core files (movies.csv, ratings.csv) exist.
    If not, it downloads the dataset from the official URL and extracts it.
    """
    # Check if we need to reconstruct files from parts
    join_file(RATINGS_FILE)
    join_file(TAGS_FILE)

    if not os.path.exists(MOVIES_FILE) or not os.path.exists(RATINGS_FILE):
        print("Downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)
        with urllib.request.urlopen(DATASET_URL) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                z.extractall(DATA_DIR)
        print("Dataset downloaded.")


def load_movies():
    """
    Loads the movies dataset into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing movie information (movieId, title, genres).
    """
    ensure_dataset_exists()
    return pd.read_csv(MOVIES_FILE)


def load_ratings():
    """
    Loads the ratings dataset into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing ratings (userId, movieId, rating, timestamp).
    """
    ensure_dataset_exists()
    return pd.read_csv(RATINGS_FILE)


def search_movies(query, movies_df):
    """
    Searches for movies based on a query string.

    The search is performed in two steps:
    1. It attempts to translate the query (if it's a Spanish genre) to English.
    2. It filters the DataFrame for matches in either the 'title' or 'genres' columns.

    Args:
        query (str): The search term entered by the user.
        movies_df (pd.DataFrame): The DataFrame containing movie data.

    Returns:
        pd.DataFrame: A filtered DataFrame containing matching movies.
    """
    if not query:
        return movies_df.head(20)

    # Convert potential Spanish genre in query to English
    from src.utils import get_english_genre

    # Try to map the whole query as a genre
    # (Simple approach: if query matches a Spanish genre, use the English version)
    query_as_english_genre = get_english_genre(query.strip().title())

    # If the mapping returned something different, it means it was a known Spanish genre
    # Use that for the genre search part.
    # Keep original query for title search.

    search_term = (
        query_as_english_genre
        if query_as_english_genre != query.strip().title()
        else query
    )

    # Simple case-insensitive search in title and genres
    # If search_term differs from query, it means we detected a genre.
    # We search for that genre in 'genres' OR the original query in 'title'.

    mask = movies_df["title"].str.contains(
        query, case=False, na=False
    ) | movies_df["genres"].str.contains(search_term, case=False, na=False)

    return movies_df[mask]


def get_movie_title(movie_id, movies_df):
    """
    Retrieves the title of a movie given its ID.

    Args:
        movie_id (int): The ID of the movie.
        movies_df (pd.DataFrame): The DataFrame containing movie data.

    Returns:
        str: The title of the movie, or "Unknown" if not found.
    """
    movie = movies_df[movies_df["movieId"] == movie_id]
    if not movie.empty:
        return movie.iloc[0]["title"]
    return "Unknown"
