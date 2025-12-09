import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import (
    init_db,
    create_user,
    delete_user,
    add_rating,
    get_db_connection,
)
from src.model import get_recommendations


def verify():
    print("Initializing DB...")
    init_db()

    username = "test_hybrid"
    email = "test_hybrid@example.com"
    password = "password"

    # 1. Cleanup
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        delete_user(user["id"])

    # 2. Create User
    print("Creating user...")
    create_user(username, email, password, [])
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_id = c.fetchone()["id"]
    conn.close()

    # 3. Test Clipping (Pure SVD)
    print("\n--- Testing Clipping (Pure SVD, Alpha=1.0) ---")
    recs = get_recommendations(user_id, n=5, selected_genres=[], alpha=1.0)
    for r in recs:
        print(f"{r['title']} - Score: {r['score']:.2f}")
        if r["score"] > 5.0 or r["score"] < 1.0:
            print("FAILURE: Score out of bounds!")
            return

    print("SUCCESS: Scores are within bounds.")

    # 4. Test Hybrid (Alpha=0.0 -> Pure Genre)
    # Target "Horror"
    print("\n--- Testing Hybrid (Pure Genre 'Horror', Alpha=0.0) ---")
    recs_horror = get_recommendations(
        user_id, n=5, selected_genres=["Horror"], alpha=0.0
    )

    horror_count = 0
    for r in recs_horror:
        print(
            f"{r['title']} ({r['genres']}) - Hybrid Score: {r['hybrid_score']:.2f}"
        )
        if "Horror" in r["genres"]:
            horror_count += 1

    if horror_count >= 3:
        print(f"SUCCESS: {horror_count}/5 movies contain Horror.")
    else:
        print(f"FAILURE: Only {horror_count}/5 movies contain Horror.")

    # 5. Test Hybrid Balance (Alpha=0.5)
    print("\n--- Testing Hybrid Balanced (Horror, Alpha=0.5) ---")
    recs_balanced = get_recommendations(
        user_id, n=5, selected_genres=["Horror"], alpha=0.5
    )
    for r in recs_balanced:
        print(
            f"{r['title']} ({r['genres']}) - SVD: {r['score']:.2f}, Hybrid: {r['hybrid_score']:.2f}"
        )

    # Cleanup
    delete_user(user_id)
    print("\nVerification Passed.")


if __name__ == "__main__":
    verify()
