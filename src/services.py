import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import os
from src.datasets import load_ml100k
from src.algorithms import default_algorithms
from src.evaluate import (
    evaluate_algorithms,
    summarize_means,
    summarize_stds,
    results_to_long_df,
)


# Cachea y devuelve el dataset MovieLens 100k
@st.cache_resource
def get_data():
    return load_ml100k()


# Cachea y devuelve el diccionario de algoritmos por defecto
@st.cache_resource
def get_algorithms():
    return default_algorithms()


# Ejecuta la evaluación con cache y devuelve resultados, medias, desviaciones y DataFrame largo
@st.cache_data(show_spinner="Ejecutando evaluación de algoritmos...")
def run_evaluation(
    cv: int, measures: tuple[str, ...], verbose: bool, include_time: bool
):
    data = get_data()
    algorithms = get_algorithms()
    results = evaluate_algorithms(
        data, algorithms, measures=measures, cv=cv, verbose=verbose
    )
    metrics_to_summarize = list(measures) + (
        ["fit_time", "test_time"] if include_time else []
    )
    means = summarize_means(results, metrics_to_summarize)
    stds = summarize_stds(results, metrics_to_summarize)
    df_long = results_to_long_df(results, measures=measures, include_time=include_time)
    return results, means, stds, df_long


# Extrae los valores numéricos (folds, media, std) de una línea de texto.
def _extract_numeric_values(
    parts: list[str],
) -> tuple[list[float], float, float] | None:
    try:
        numeric_parts = [
            p
            for p in parts
            if p.replace(".", "", 1).isdigit()
            or (p.startswith("-") and p[1:].replace(".", "", 1).isdigit())
        ]
        values = [float(x) for x in numeric_parts]

        if len(values) < 3:  # Necesitamos al menos fold1, mean, std
            return None

        mean_val = values[-2]
        std_val = values[-1]
        folds_vals = values[:-2]
        return folds_vals, mean_val, std_val
    except (ValueError, IndexError):
        return None


# Parsea una línea del fichero de resultados para extraer una métrica.
def _parse_line_for_metric(line, algo_name, results_mean, results_std, long_rows):
    parts = line.split()
    if not parts:
        return

    metric_map = {
        "RMSE": "RMSE",
        "MSE": "MSE",
        "MAE": "MAE",
        "FCP": "FCP",
        "Fit time": "fit_time",
        "Test time": "test_time",
    }
    line_start = " ".join(parts[:2])
    metric = metric_map.get(parts[0]) or metric_map.get(line_start)

    if not metric:
        return

    numeric_values = _extract_numeric_values(parts)
    if not numeric_values:
        return

    folds_vals, mean_val, std_val = numeric_values
    results_mean[metric][algo_name] = mean_val
    results_std[metric][algo_name] = std_val

    for i, v in enumerate(folds_vals):
        long_rows.append(
            {"algorithm": algo_name, "metric": metric, "fold": i + 1, "value": v}
        )


def parse_static_results(filepath: str):
    if not os.path.exists(filepath):
        return {}, {}, pd.DataFrame()

    with open(filepath, "r") as f:
        content = f.read()

    metrics = ["RMSE", "MSE", "MAE", "FCP", "fit_time", "test_time"]
    results_mean = {m: {} for m in metrics}
    results_std = {m: {} for m in metrics}
    long_rows = []

    blocks = content.split("Evaluating RMSE, MSE, MAE, FCP of algorithm ")
    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")
        algo_name = lines[0].split(" ")[0]

        for line in lines:
            _parse_line_for_metric(
                line, algo_name, results_mean, results_std, long_rows
            )

    df_long = pd.DataFrame(long_rows)
    return results_mean, results_std, df_long


@st.cache_data
def get_static_results():
    return parse_static_results("results/ml-32m.txt")
