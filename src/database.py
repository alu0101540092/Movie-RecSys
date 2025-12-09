import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "movie_recsys.db")


def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: A connection object with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the database schema.
    
    Creates 'users' and 'ratings' tables if they do not exist.
    Also handles schema migrations, such as adding the 'timestamp' column to 'ratings'.
    """
    conn = get_db_connection()
    c = conn.cursor()

    # Users table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            favorite_genres TEXT
        )
    """
    )

    # Ratings table
    # We use a composite primary key or just an index.
    # For simplicity in this prototype, we'll just index user_id and movie_id.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            movie_id INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (user_id, movie_id)
        )
    """
    )
    
    # Check if timestamp column exists (migration for existing DBs)
    c.execute("PRAGMA table_info(ratings)")
    columns = [info[1] for info in c.fetchall()]
    if "timestamp" not in columns:
        print("Migrating database: Adding timestamp column to ratings table...")
        c.execute("ALTER TABLE ratings ADD COLUMN timestamp INTEGER")
        # Optional: Backfill with current time for existing ratings
        current_time = int(datetime.now().timestamp())
        c.execute("UPDATE ratings SET timestamp = ? WHERE timestamp IS NULL", (current_time,))
        print("Migration completed.")

    conn.commit()
    conn.close()


def hash_password(password):
    """
    Hashes a password using SHA-256.
    
    Args:
        password (str): The plain text password.
        
    Returns:
        str: The hexadecimal digest of the hashed password.
    """
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, email, password, favorite_genres):
    """
    Creates a new user in the database.
    
    Args:
        username (str): The username.
        email (str): The user's email.
        password (str): The user's password (will be hashed).
        favorite_genres (list): A list of favorite genres (strings).
        
    Returns:
        bool: True if the user was created successfully, False if the username or email already exists.
    """
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash, favorite_genres) VALUES (?, ?, ?, ?)",
            (
                username,
                email,
                hash_password(password),
                ",".join(favorite_genres),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    """
    Authenticates a user by checking their username and password.
    
    Args:
        username (str): The username.
        password (str): The plain text password.
        
    Returns:
        sqlite3.Row or None: The user row if authentication is successful, None otherwise.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password)),
    )
    user = c.fetchone()
    conn.close()
    return user


def delete_user(user_id):
    """
    Deletes a user and their associated ratings from the database.
    
    Args:
        user_id (int): The ID of the user to delete.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    c.execute("DELETE FROM ratings WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def add_rating(user_id, movie_id, rating):
    """
    Adds or updates a rating for a specific movie by a user.
    
    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the movie.
        rating (float): The rating value (0.5 to 5.0).
    """
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = int(datetime.now().timestamp())
    c.execute(
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, movie_id, rating, timestamp),
    )
    conn.commit()
    conn.close()


def get_user_ratings(user_id):
    """
    Retrieves all ratings made by a specific user.
    
    Args:
        user_id (int): The ID of the user.
        
    Returns:
        pd.DataFrame: A DataFrame containing the user's ratings (movie_id, rating, timestamp).
    """
    conn = get_db_connection()
    query = "SELECT movie_id, rating, timestamp FROM ratings WHERE user_id = ?"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df



def get_all_ratings():
    """
    Retrieves all ratings from the database.
    
    Returns:
        pd.DataFrame: A DataFrame containing all ratings (user_id, movie_id, rating, timestamp).
    """
    conn = get_db_connection()
    query = "SELECT user_id, movie_id, rating, timestamp FROM ratings"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_user_genres(user_id):
    """
    Retrieves the favorite genres of a specific user.
    
    Args:
        user_id (int): The ID of the user.
        
    Returns:
        list: A list of genre strings.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT favorite_genres FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    if result and result["favorite_genres"]:
        return result["favorite_genres"].split(",")
    return []


def update_user_genres(user_id, genres):
    """
    Updates the favorite genres for a specific user.
    
    Args:
        user_id (int): The ID of the user.
        genres (list): A list of new favorite genres (strings).
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET favorite_genres = ? WHERE id = ?",
        (",".join(genres), user_id),
    )
    conn.commit()
    conn.close()
