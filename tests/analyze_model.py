import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_optimized_components, fold_in_user, load_movies

def analyze():
    print("Loading model components...")
    components = load_optimized_components()
    if not components:
        print("Model components not found!")
        return
        
    pu, qi, bu, bi, global_mean, mappings = components
    
    print(f"Item factors shape: {qi.shape}")
    print(f"Item biases shape: {bi.shape}")
    
    # 1. Inspect Mappings
    print("\n--- Mapping Check ---")
    movies_df = load_movies()
    # Pick a known movie, e.g., Toy Story (1)
    toy_story = movies_df[movies_df['movieId'] == 1].iloc[0]
    print(f"Movie: {toy_story['title']} (ID: 1)")
    
    ts_inner_id = mappings['items'].get(1)
    if ts_inner_id is None:
         ts_inner_id = mappings['items'].get('1')
         
    print(f"Inner ID for Toy Story: {ts_inner_id}")
    
    if ts_inner_id is not None:
        print(f"Toy Story Vector Norm: {np.linalg.norm(qi[ts_inner_id]):.4f}")
        print(f"Toy Story Bias: {bi[ts_inner_id]:.4f}")

    # 2. Test Fold-In Quality
    print("\n--- Fold-In Quality Check ---")
    # Simulate a user who loves Animation
    # Let's pick 5 random Animation movies
    animation_movies = movies_df[movies_df['genres'].str.contains('Animation', na=False)].head(5)
    
    user_ratings = []
    vector_indices = []
    
    print("User rates:")
    for _, row in animation_movies.iterrows():
        mid = row['movieId']
        print(f"  - {row['title']} (5.0)")
        user_ratings.append({'movie_id': mid, 'rating': 5.0})
        
        # Get inner id for analysis
        iid = mappings['items'].get(mid)
        if iid is None: iid = mappings['items'].get(str(mid))
        if iid is not None: vector_indices.append(iid)

    user_ratings_df = pd.DataFrame(user_ratings)
    
    # Fold in
    print("\n...Folding in user...")
    # Using defaults first
    user_factors, user_bias = fold_in_user(user_ratings_df, qi, bi, global_mean, mappings, n_epochs=20, lr=0.005)
    
    print(f"Resulting User Bias: {user_bias:.4f}")
    print(f"Resulting User Vector Norm: {np.linalg.norm(user_factors):.4f}")
    
    # Check similarity with rated items
    if vector_indices:
        rated_vectors = qi[vector_indices]
        # Cosine sim
        sims = cosine_similarity([user_factors], rated_vectors)[0]
        print(f"Avg Similarity with rated items: {np.mean(sims):.4f}")
        print(f"Sims: {sims}")
        
        # Check similarity with random items
        random_indices = np.random.choice(qi.shape[0], 100)
        random_vectors = qi[random_indices]
        random_sims = cosine_similarity([user_factors], random_vectors)[0]
        print(f"Avg Similarity with random items: {np.mean(random_sims):.4f}")
    
    # Check what this user would get recommended
    scores = np.dot(qi, user_factors) + bi + user_bias + global_mean
    
    top_indices = np.argsort(scores)[::-1][:10]
    
    # Map back to titles
    # Create reverse mapping
    raw_item_ids = {}
    for raw, inner in mappings['items'].items():
        try:
            raw_item_ids[inner] = int(raw)
        except:
            raw_item_ids[inner] = raw
            
    print("\nTop Recommendations:")
    for i in top_indices:
        mid = raw_item_ids.get(i)
        if mid:
            row = movies_df[movies_df['movieId'] == mid]
            if not row.empty:
                print(f"  - {row.iloc[0]['title']} ({row.iloc[0]['genres']}) [Score: {scores[i]:.4f}]")

if __name__ == "__main__":
    analyze()
