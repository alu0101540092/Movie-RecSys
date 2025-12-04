import os
import pickle
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from src.data_loader import load_ratings, load_movies
from src.database import get_user_ratings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_model.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")


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


def load_optimized_components():
    """Load optimized model components using memory mapping."""
    try:
        pu = np.load(os.path.join(MODELS_DIR, 'svd_pu.npy'), mmap_mode='r')
        qi = np.load(os.path.join(MODELS_DIR, 'svd_qi.npy'), mmap_mode='r')
        bu = np.load(os.path.join(MODELS_DIR, 'svd_bu.npy'), mmap_mode='r')
        bi = np.load(os.path.join(MODELS_DIR, 'svd_bi.npy'), mmap_mode='r')
        global_mean = np.load(os.path.join(MODELS_DIR, 'svd_global_mean.npy'))[0]
        
        with open(os.path.join(MODELS_DIR, 'svd_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            
        return pu, qi, bu, bi, global_mean, mappings
    except FileNotFoundError:
        return None

def get_recommendations(user_id, n=10):
    # Try to load optimized components
    components = load_optimized_components()
    
    if components:
        # Optimized path
        pu, qi, bu, bi, global_mean, mappings = components
        movies_df = load_movies()
        
        # Get user inner ID
        u_inner_id = mappings['users'].get(user_id)
        
        # Calculate scores for ALL items in the model
        # Score = mu + bu + bi + qi . pu
        
        if u_inner_id is not None:
            # User is known
            user_bias = bu[u_inner_id]
            user_factors = pu[u_inner_id]
            
            # Vectorized calculation for all items
            # scores = global_mean + user_bias + bi + (qi @ user_factors)
            scores = np.dot(qi, user_factors)
            scores += bi
            scores += user_bias
            scores += global_mean
        else:
            # User is new (cold start)
            # Score = global_mean + bi
            scores = bi + global_mean
            
        # Create a DataFrame for scores
        # We need to map inner item IDs back to raw item IDs (movieId)
        # mappings['items'] is raw -> inner. We need inner -> raw.
        # Since 'items' dict values are 0..N-1, we can create a list/array where index is inner_id
        
        # Create reverse mapping (inner -> raw)
        # This might be slow if done every time. Ideally should be saved too.
        # But for now let's do it efficiently.
        raw_item_ids = [None] * len(mappings['items'])
        for raw_id, inner_id in mappings['items'].items():
            if inner_id < len(raw_item_ids):
                # Convert to int if possible to match movies_df
                try:
                    raw_item_ids[inner_id] = int(raw_id)
                except (ValueError, TypeError):
                    raw_item_ids[inner_id] = raw_id
                
        # Filter out movies already rated by user
        user_ratings_df = get_user_ratings(user_id)
        rated_movie_ids = set(user_ratings_df["movie_id"].values) if not user_ratings_df.empty else set()
        
        # Prepare results
        recommendations = []
        
        # Get top N indices
        # We can use argpartition for efficiency if N is small compared to total items
        top_indices = np.argpartition(scores, -n)[-n:]
        # Sort these top indices
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        for i in top_indices:
            movie_id = raw_item_ids[i]
            if movie_id in rated_movie_ids:
                continue
                
            score = scores[i]
            
            # Get metadata
            movie_row = movies_df[movies_df["movieId"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]["title"]
                genres = movie_row.iloc[0]["genres"]
                recommendations.append({
                    "movieId": movie_id,
                    "title": title,
                    "genres": genres,
                    "score": float(score)
                })
                
        # If we filtered out some rated movies, we might have fewer than n
        # In that case we should take more from argpartition, but for simplicity:
        return recommendations[:n]

    else:
        # Fallback to original slow method
        print("Optimized model not found, falling back to slow method...")
        algo = load_model()
        movies_df = load_movies()
        all_movie_ids = movies_df["movieId"].unique()
        user_ratings_df = get_user_ratings(user_id)
        rated_movie_ids = (
            set(user_ratings_df["movie_id"].values)
            if not user_ratings_df.empty
            else set()
        )
        predictions = []
        for movie_id in all_movie_ids:
            if movie_id not in rated_movie_ids:
                est = algo.predict(user_id, movie_id).est
                predictions.append((movie_id, est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
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
