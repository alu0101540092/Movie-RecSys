from surprise import SVD, NMF, SlopeOne, KNNBasic, CoClustering, AlgoBase  # type: ignore


def default_algorithms() -> dict[str, AlgoBase]:
    """
    Devuelve un diccionario nombre->algoritmo con los modelos por defecto.

    Returns:
        dict[str, AlgoBase]: Diccionario con los algoritmos por defecto.
    """
    return {
        "SVD": SVD(),
        "NMF": NMF(),
        "SlopeOne": SlopeOne(),
        "KNNBasic": KNNBasic(),
        "CoClustering": CoClustering(),
    }
