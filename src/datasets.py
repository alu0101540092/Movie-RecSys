from surprise import Dataset


def load_ml100k():
    """Carga el dataset MovieLens 100k desde Surprise."""
    return Dataset.load_builtin("ml-100k", prompt=False)
