from surprise import SVD, NMF, SlopeOne, KNNBasic, CoClustering


def default_algorithms():
    """Devuelve un diccionario nombre->algoritmo con los modelos por defecto."""
    return {
        "SVD": SVD(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "CoClustering": CoClustering(),
    }
