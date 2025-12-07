import os
import pickle
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from src.data_loader import load_ratings, load_movies
from src.database import get_user_ratings

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.file_manager import join_file

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
    # Ensure all components are ready (reconstructed if needed)
    for filename in ['svd_pu.npy', 'svd_qi.npy', 'svd_bu.npy', 'svd_bi.npy', 'svd_global_mean.npy']:
        join_file(os.path.join(MODELS_DIR, filename))

    try:
        pu = np.load(os.path.join(MODELS_DIR, 'svd_pu.npy'), mmap_mode='r', allow_pickle=True)
        qi = np.load(os.path.join(MODELS_DIR, 'svd_qi.npy'), mmap_mode='r', allow_pickle=True)
        bu = np.load(os.path.join(MODELS_DIR, 'svd_bu.npy'), mmap_mode='r', allow_pickle=True)
        bi = np.load(os.path.join(MODELS_DIR, 'svd_bi.npy'), mmap_mode='r', allow_pickle=True)
        global_mean = np.load(os.path.join(MODELS_DIR, 'svd_global_mean.npy'), allow_pickle=True)[0]
        
        with open(os.path.join(MODELS_DIR, 'svd_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            
        return pu, qi, bu, bi, global_mean, mappings
    except FileNotFoundError:
        return None

def fold_in_user(user_ratings_df, qi, bi, global_mean, mappings, n_epochs=20, lr=0.005, reg=0.02):
    """
    Optimizes the user latent factors (pu) and user bias (bu) for a specific user
    based on their current ratings, performing a mini SGD 'fold-in' at runtime.
    """
    n_factors = qi.shape[1]
    # Initialize with small random values to break symmetry
    pu = np.random.normal(0, 0.1, n_factors)
    bu = 0.0

    # Prepare training samples from user ratings
    samples = []
    
    # Mappings keys might be strings or ints depending on how they were saved.
    item_map = mappings['items']
    
    for _, row in user_ratings_df.iterrows():
        raw_id = row['movie_id']
        inner_id = None
        
        # Try direct lookup, string conversion, and int conversion
        if raw_id in item_map:
             inner_id = item_map[raw_id]
        elif str(raw_id) in item_map:
             inner_id = item_map[str(raw_id)]
        else:
             try:
                 if int(raw_id) in item_map:
                     inner_id = item_map[int(raw_id)]
             except (ValueError, TypeError):
                 pass
                 
        if inner_id is not None:
            samples.append((inner_id, float(row['rating'])))
            
    if not samples:
        return pu, bu
        
    # SGD Optimization
    for _ in range(n_epochs):
        for i, r in samples:
            # Prediction: global + bu + bi + pu . qi
            dot = np.dot(qi[i], pu)
            est = global_mean + bu + bi[i] + dot
            
            err = r - est
            
            # Update rules
            bu += lr * (err - reg * bu)
            new_pu = pu + lr * (err * qi[i] - reg * pu)
            pu = new_pu
            
    return pu, bu


def get_recommendations(user_id, n=10):
    # Try to load optimized components
    components = load_optimized_components()
    
    if components:
        # Optimized path
        pu, qi, bu, bi, global_mean, mappings = components
        movies_df = load_movies()
        
        # Fetch user ratings to fold-in
        user_ratings_df = get_user_ratings(user_id)
        rated_movie_ids = set(user_ratings_df["movie_id"].values) if not user_ratings_df.empty else set()
        
        # Determine User Factors
        if not user_ratings_df.empty:
            # Fold-in: dynamically compute user factors based on current ratings
            user_factors, user_bias = fold_in_user(user_ratings_df, qi, bi, global_mean, mappings)
        else:
            # No ratings -> Pure Cold Start (Global Mean + Item Bias)
            user_factors = np.zeros(qi.shape[1])
            user_bias = 0.0

        # Calculate scores
        # Score = global_mean + user_bias + bi + (qi . user_factors)
        scores = np.dot(qi, user_factors)
        scores += bi
        scores += user_bias
        scores += global_mean

        # Create reverse mapping (inner -> raw)
        raw_item_ids = [None] * len(mappings['items'])
        for raw_id, inner_id in mappings['items'].items():
            if inner_id < len(raw_item_ids):
                try:
                    raw_item_ids[inner_id] = int(raw_id)
                except (ValueError, TypeError):
                    raw_item_ids[inner_id] = raw_id
        
        # Prepare results
        recommendations = []
        
        # Get top N indices
        top_indices = np.argpartition(scores, -n)[-n:]
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
