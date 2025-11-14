from surprise import Dataset


# Carga el dataset MovieLens 100k desde Surprise
def load_ml100k():
    return Dataset.load_builtin("ml-100k", prompt=False)
