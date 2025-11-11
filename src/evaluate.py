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
