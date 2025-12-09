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

    username = "test_determ"
    email = "test_determ@example.com"
    password = "password"

    # 1. Cleanup
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    if user:
        delete_user(user["id"])

    # 2. Create User
    create_user(username, email, password, [])
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user_id = c.fetchone()["id"]
    conn.close()

    # Rate some movies to trigger fold-in
    add_rating(user_id, 1, 5.0)  # Toy Story
    add_rating(user_id, 47, 5.0)  # Seven

    # Run 1
    print("Run 1...")
    recs_1 = get_recommendations(user_id, n=10, selected_genres=[], alpha=1.0)
    scores_1 = [r["score"] for r in recs_1]

    # Run 2
    print("Run 2...")
    recs_2 = get_recommendations(user_id, n=10, selected_genres=[], alpha=1.0)
    scores_2 = [r["score"] for r in recs_2]

    print(f"Scores 1: {scores_1[:3]}")
    print(f"Scores 2: {scores_2[:3]}")

    if scores_1 == scores_2:
        print("SUCCESS: Recommendations are deterministic.")
    else:
        print("FAILURE: Recommendations changed between runs!")

    delete_user(user_id)


if __name__ == "__main__":
    verify()
