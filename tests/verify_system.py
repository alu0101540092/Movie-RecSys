import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import (
    init_db,
    create_user,
    authenticate_user,
    add_rating,
    get_user_ratings,
)
from src.data_loader import load_movies, search_movies
from src.model import get_recommendations, train_model


def test_system():
    print("1. Initializing Database...")
    init_db()

    print("2. Creating User...")
    username = "testuser"
    password = "password123"
    email = "test@example.com"
    genres = ["Action", "Sci-Fi"]

    # Clean up if exists
    # (In a real test we'd use a temp db, but here we just want to verify it works)

    if create_user(username, email, password, genres):
        print("   User created.")
    else:
        print("   User already exists (expected if re-running).")

    print("3. Authenticating...")
    user = authenticate_user(username, password)
    assert user is not None
    print(f"   Authenticated as {user['username']}")

    print("4. Searching Movies...")
    movies = load_movies()
    results = search_movies("Star Wars", movies)
    assert not results.empty
    print(f"   Found {len(results)} movies for 'Star Wars'")

    movie_id = results.iloc[0]["movieId"]
    movie_title = results.iloc[0]["title"]
    print(f"   Selected: {movie_title} (ID: {movie_id})")

    print("5. Adding Rating...")
    add_rating(user["id"], movie_id, 5.0)
    ratings = get_user_ratings(user["id"])
    assert not ratings.empty
    print("   Rating added.")

    print("6. Generating Recommendations...")
    # Force train if needed (might take a moment)
    if not os.path.exists("svd_model.pkl"):
        train_model()

    recs = get_recommendations(user["id"], n=5)
    print("   Top 5 Recommendations:")
    for rec in recs:
        print(f"   - {rec['title']} ({rec['score']:.2f})")

    print("\nSystem Verification Passed!")


if __name__ == "__main__":
    test_system()
