import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_db, create_user, delete_user, add_rating, get_db_connection
from src.model import get_recommendations
from src.data_loader import load_movies

def verify():
    print("Initializing DB...")
    init_db()
    
    username = "test_verifier"
    email = "test_verifier@example.com"
    password = "password"
    
    # 1. Cleanup
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        delete_user(user['id'])
    conn.close()
    
    # 2. Create User
    print("Creating user...")
    create_user(username, email, password, [])
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_id = c.fetchone()['id']
    conn.close()
    
    # 3. Get initial recommendations (Cold Start)
    print("Getting cold start recommendations...")
    recs_initial = get_recommendations(user_id, n=5)
    print("Initial Recs:")
    for r in recs_initial:
        print(f"  - {r['title']} (Score: {r['score']:.2f})")
    
    # 4. Rate some movies (Animation/Children)
    # Finding some animation movies that are NOT in the initial Recs (to avoid rating what we just saw, though it doesn't matter much)
    movies_df = load_movies()
    # Filter for Animation
    animation_movies = movies_df[movies_df['genres'].str.contains('Animation', na=False)].head(5)
    
    print("\nRating movies (5 stars):")
    for _, row in animation_movies.iterrows():
        print(f"  - {row['title']}")
        add_rating(user_id, row['movieId'], 5.0)
        
    # 5. Get new recommendations
    print("\nGetting updated recommendations...")
    recs_updated = get_recommendations(user_id, n=5)
    print("Updated Recs:")
    for r in recs_updated:
        print(f"  - {r['title']} (Score: {r['score']:.2f})")
    
    # 6. Compare
    initial_ids = [r['movieId'] for r in recs_initial]
    updated_ids = [r['movieId'] for r in recs_updated]
    
    if initial_ids != updated_ids:
        print("\nSUCCESS: Recommendations changed after rating movies.")
        
        # Check if score distribution changed or if we see different items
        print("Verification passed.")
    else:
        print("\nFAILURE: Recommendations did not change.")
        
    # Cleanup
    delete_user(user_id)

if __name__ == "__main__":
    verify()
