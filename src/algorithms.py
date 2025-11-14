from surprise import SVD, NMF, SlopeOne, KNNBasic, CoClustering


# Devuelve un diccionario nombre->algoritmo con los modelos por defecto
def default_algorithms():
    return {
        "SVD": SVD(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "CoClustering": CoClustering(),
    }
