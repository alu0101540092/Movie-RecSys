from typing import Dict, Iterable
import numpy as np  # type: ignore
from surprise.model_selection import cross_validate  # type: ignore


def evaluate_algorithms(
    data,
    algorithms: Dict[str, object],
    measures: Iterable[str] = ("RMSE", "MSE", "MAE", "FCP"),
    cv: int = 5,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Ejecuta cross_validate para cada algoritmo y devuelve sus resultados.

    Args:
        data: Dataset de Surprise.
        algorithms (Dict[str, object]): Diccionario de algoritmos.
        measures (Iterable[str]): Métricas a evaluar.
        cv (int): Número de folds.
        verbose (bool): Si mostrar output detallado.

    Returns:
        Dict[str, dict]: Diccionario con los resultados.
    """
    results: Dict[str, dict] = {}
    for name, algo in algorithms.items():
        results[name] = cross_validate(
            algo, data, measures=list(measures), cv=cv, verbose=verbose
        )
    return results


def summarize_means(
    results: Dict[str, dict], metrics: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calcula medias por métrica y algoritmo (RMSE/MSE/MAE/FCP + fit_time/test_time).

    Args:
        results (Dict[str, dict]): Resultados de la evaluación.
        metrics (Iterable[str]): Métricas a resumir.

    Returns:
        Dict[str, Dict[str, float]]: Diccionario de medias.
    """
    means = {}
    for metric in metrics:
        agg = {}
        for name, res in results.items():
            key = (
                f"test_{metric.lower()}"
                if metric in ("RMSE", "MSE", "MAE", "FCP")
                else metric
            )
            arr = np.array(res[key])
            agg[name] = float(arr.mean())
        means[metric] = agg
    return means


def summarize_stds(
    results: Dict[str, dict], metrics: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calcula desviación estándar por métrica y algoritmo (ddof=1).

    Args:
        results (Dict[str, dict]): Resultados de la evaluación.
        metrics (Iterable[str]): Métricas a resumir.

    Returns:
        Dict[str, Dict[str, float]]: Diccionario de desviaciones estándar.
    """
    stds = {}
    for metric in metrics:
        agg = {}
        for name, res in results.items():
            key = (
                f"test_{metric.lower()}"
                if metric in ("RMSE", "MSE", "MAE", "FCP")
                else metric
            )
            arr = np.array(res[key])
            agg[name] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        stds[metric] = agg
    return stds


def results_to_long_df(
    results: Dict[str, dict],
    measures: Iterable[str] = ("RMSE", "MSE", "MAE", "FCP"),
    include_time: bool = True,
) -> "pd.DataFrame":
    """
    Convierte resultados de cross_validate a DataFrame largo (algorithm, metric, fold, value).

    Args:
        results (Dict[str, dict]): Resultados de la evaluación.
        measures (Iterable[str]): Métricas evaluadas.
        include_time (bool): Si incluir tiempos.

    Returns:
        pd.DataFrame: DataFrame en formato largo.
    """
    import pandas as pd  # type: ignore

    rows = []
    for algo, res in results.items():
        for metric in measures:
            key = f"test_{metric.lower()}"
            values = res[key]
            for i, v in enumerate(values):
                rows.append(
                    {
                        "algorithm": algo,
                        "metric": metric,
                        "fold": i + 1,
                        "value": float(v),
                    }
                )
        if include_time:
            for key in ("fit_time", "test_time"):
                values = res[key]
                for i, v in enumerate(values):
                    rows.append(
                        {
                            "algorithm": algo,
                            "metric": key,
                            "fold": i + 1,
                            "value": float(v),
                        }
                    )
    return pd.DataFrame(rows)
