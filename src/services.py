import streamlit as st
import pandas as pd
import os
from typing import Any
from surprise import AlgoBase, Dataset
from src.datasets import load_ml100k
from src.algorithms import default_algorithms
from src.evaluate import (
    evaluate_algorithms,
    summarize_means,
    summarize_stds,
    results_to_long_df,
)


@st.cache_resource
def get_data() -> Dataset:
    """
    Loads and caches the MovieLens 100k dataset.

    Returns:
        Dataset: The loaded dataset.
    """
    return load_ml100k()


@st.cache_resource
def get_algorithms() -> dict[str, AlgoBase]:
    """
    Returns and caches the default algorithms.

    Returns:
        dict[str, AlgoBase]: A dictionary of algorithms.
    """
    return default_algorithms()


@st.cache_data(show_spinner="Ejecutando evaluación de algoritmos...")
def run_evaluation(
    cv: int, measures: tuple[str, ...], verbose: bool, include_time: bool
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]], pd.DataFrame]:
    """
    Runs the evaluation of algorithms with the specified parameters.

    Args:
        cv (int): Number of cross-validation folds.
        measures (tuple[str, ...]): Metrics to evaluate.
        verbose (bool): Whether to print verbose output.
        include_time (bool): Whether to include execution time in results.

    Returns:
        tuple: A tuple containing raw results, mean results, standard deviation results, and a long-format DataFrame.
    """
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


def _extract_numeric_values(
    parts: list[str],
) -> tuple[list[float], float, float] | None:
    """
    Extracts numeric values from a list of string parts.

    Args:
        parts (list[str]): List of string parts from a line.

    Returns:
        tuple[list[float], float, float] | None: A tuple containing fold values, mean, and std, or None if extraction fails.
    """
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


def _parse_line_for_metric(
    line: str,
    algo_name: str,
    results_mean: dict[str, dict[str, float]],
    results_std: dict[str, dict[str, float]],
    long_rows: list[dict[str, Any]],
) -> None:
    """
    Parses a line from the results file to extract metric data.

    Args:
        line (str): The line to parse.
        algo_name (str): The name of the algorithm.
        results_mean (dict): Dictionary to store mean results.
        results_std (dict): Dictionary to store std results.
        long_rows (list): List to append long-format rows.
    """
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


def parse_static_results(
    filepath: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], pd.DataFrame]:
    """
    Parses static results from a file.

    Args:
        filepath (str): Path to the results file.

    Returns:
        tuple: Mean results, std results, and a long-format DataFrame.
    """
    if not os.path.exists(filepath):
        return {}, {}, pd.DataFrame()

    with open(filepath, "r") as f:
        content = f.read()

    metrics = ["RMSE", "MSE", "MAE", "FCP", "fit_time", "test_time"]
    results_mean: dict[str, dict[str, float]] = {m: {} for m in metrics}
    results_std: dict[str, dict[str, float]] = {m: {} for m in metrics}
    long_rows: list[dict[str, Any]] = []

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
def get_static_results() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], pd.DataFrame]:
    """
    Retrieves static results for ML-32m.

    Returns:
        tuple: Mean results, std results, and a long-format DataFrame.
    """
    return parse_static_results("results/ml-32m.txt")
