import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

DB_PATH = "movie_recsys.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            favorite_genres TEXT
        )
    ''')
    
    # Ratings table
    # We use a composite primary key or just an index. 
    # For simplicity in this prototype, we'll just index user_id and movie_id.
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            movie_id INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (user_id, movie_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, favorite_genres):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash, favorite_genres) VALUES (?, ?, ?, ?)",
            (username, email, hash_password(password), ",".join(favorite_genres))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def delete_user(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    c.execute("DELETE FROM ratings WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def add_rating(user_id, movie_id, rating):
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = int(datetime.now().timestamp())
    c.execute(
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, movie_id, rating, timestamp)
    )
    conn.commit()
    conn.close()

def get_user_ratings(user_id):
    conn = get_db_connection()
    query = "SELECT movie_id, rating FROM ratings WHERE user_id = ?"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

def get_all_ratings():
    conn = get_db_connection()
    query = "SELECT user_id, movie_id, rating, timestamp FROM ratings"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
