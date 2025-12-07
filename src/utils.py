
GENRE_MAP = {
    "Action": "Acción",
    "Adventure": "Aventura",
    "Animation": "Animación",
    "Children": "Infantil",
    "Comedy": "Comedia",
    "Crime": "Crimen",
    "Documentary": "Documental",
    "Drama": "Drama",
    "Fantasy": "Fantasía",
    "Film-Noir": "Cine Negro",
    "Horror": "Terror",
    "Musical": "Musical",
    "Mystery": "Misterio",
    "Romance": "Romance",
    "Sci-Fi": "Ciencia Ficción",
    "Thriller": "Suspense",
    "War": "Guerra",
    "Western": "Western",
    "(no genres listed)": "Sin género",
}

REVERSE_GENRE_MAP = {v: k for k, v in GENRE_MAP.items()}

def translate_genres(genres_str):
    """
    Translates a pipe-separated string of genres (EN) to a comma-separated string (ES).
    """
    if not isinstance(genres_str, str):
        return genres_str
    parts = genres_str.split("|")
    translated = [GENRE_MAP.get(p, p) for p in parts]
    return ", ".join(translated)

def get_spanish_genres_list():
    """Returns a sorted list of genres in Spanish."""
    valid_genres = [v for k, v in GENRE_MAP.items() if k != "(no genres listed)"]
    return sorted(valid_genres)

def get_english_genre(spanish_genre):
    """Returns the English equivalent of a Spanish genre."""
    return REVERSE_GENRE_MAP.get(spanish_genre, spanish_genre)
