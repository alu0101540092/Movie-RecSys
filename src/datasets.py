from surprise import Dataset  # type: ignore


def load_ml100k() -> Dataset:
    """
    Carga el dataset MovieLens 100k desde Surprise.

    Returns:
        Dataset: El dataset cargado.
    """
    return Dataset.load_builtin("ml-100k", prompt=False)
