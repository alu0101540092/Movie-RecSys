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


def parse_static_results(filepath: str):
    if not os.path.exists(filepath):
        return {}, {}, pd.DataFrame()

    with open(filepath, "r") as f:
        content = f.read()

    results_mean = {}
    results_std = {}
    long_rows = []

    metrics = ["RMSE", "MSE", "MAE", "FCP", "fit_time", "test_time"]
    for m in metrics:
        results_mean[m] = {}
        results_std[m] = {}

    blocks = content.split("Evaluating RMSE, MSE, MAE, FCP of algorithm ")

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split("\n")
        algo_name = lines[0].split(" ")[0]

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            metric = None
            if parts[0] in ("RMSE", "MSE", "MAE", "FCP"):
                metric = parts[0]
            elif parts[0] == "Fit" and parts[1] == "time":
                metric = "fit_time"
            elif parts[0] == "Test" and parts[1] == "time":
                metric = "test_time"

            if metric:
                try:
                    # Last 7 values are: fold1, fold2, fold3, fold4, fold5, mean, std
                    values = [float(x) for x in parts[-7:]]
                    folds_vals = values[:5]
                    mean_val = values[5]
                    std_val = values[6]

                    results_mean[metric][algo_name] = mean_val
                    results_std[metric][algo_name] = std_val

                    for i, v in enumerate(folds_vals):
                        long_rows.append(
                            {
                                "algorithm": algo_name,
                                "metric": metric,
                                "fold": i + 1,
                                "value": v,
                            }
                        )
                except ValueError:
                    continue

    df_long = pd.DataFrame(long_rows)
    return results_mean, results_std, df_long


@st.cache_data
def get_static_results():
    return parse_static_results("results/ml-32m.txt")
