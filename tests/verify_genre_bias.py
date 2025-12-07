import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_db, create_user, delete_user, get_db_connection
from src.model import get_recommendations

def verify():
    print("Initializing DB...")
    init_db()
    
    username = "test_bias"
    email = "test_bias@example.com"
    password = "password"
    
    # 1. Cleanup
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        delete_user(user['id'])
    
    # 2. Create User
    create_user(username, email, password, [])
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_id = c.fetchone()['id']
    conn.close()
    
    # 3. Test Hybrid for "Horror"
    # We want to see if multi-genre movies like "Get Out" are ranked high
    print("\n--- Testing 'Horror' Selection (Alpha=0.5) ---")
    recs = get_recommendations(user_id, n=15, selected_genres=["Horror"], alpha=0.5)
    
    found_multi_genre = False
    
    print("Top 15 Recommendations with Horror:")
    for r in recs:
        score = r['hybrid_score']
        genres = r['genres']
        is_multi = '|' in genres
        print(f"{r['title']} ({genres}) - Hybrid: {score:.2f} {'[Multi]' if is_multi else '[Single]'}")
        
        if is_multi and "Horror" in genres:
            found_multi_genre = True
            
    if found_multi_genre:
        print("\nSUCCESS: Multi-genre horror movies are appearing in top results.")
    else:
        print("\nWARNING: mostly or only single-genre movies found. Check catalog or scoring.")

    # Cleanup
    delete_user(user_id)

if __name__ == "__main__":
    verify()
