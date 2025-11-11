from typing import Dict, Iterable
import numpy as np
from surprise.model_selection import cross_validate


def evaluate_algorithms(
    data,
    algorithms: Dict[str, object],
    measures: Iterable[str] = ("RMSE", "MSE", "MAE", "FCP"),
    cv: int = 5,
    verbose: bool = True,
):
    """Ejecuta cross_validate para cada algoritmo y devuelve sus resultados."""
    results: Dict[str, dict] = {}
    for name, algo in algorithms.items():
        results[name] = cross_validate(
            algo, data, measures=list(measures), cv=cv, verbose=verbose
        )
    return results


def summarize_means(results: Dict[str, dict], metrics: Iterable[str]):
    """Calcula medias por métrica y algoritmo, compatible con RMSE/MSE/MAE/FCP + fit_time/test_time."""
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


def summarize_stds(results: Dict[str, dict], metrics: Iterable[str]):
    """Calcula desviación estándar por métrica y algoritmo (ddof=1)."""
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


# ...existing code...
def results_to_long_df(
    results: Dict[str, dict],
    measures: Iterable[str] = ("RMSE", "MSE", "MAE", "FCP"),
    include_time: bool = True,
):
    """Convierte resultados de cross_validate a un DataFrame largo: algorithm, metric, fold, value."""
    import pandas as pd

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
