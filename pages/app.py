from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

from src.datasets import load_ml100k
from src.algorithms import default_algorithms
from src.evaluate import evaluate_algorithms, summarize_means

st.title("Comparación de Algoritmos de Recomendación")


@st.cache_resource
def get_data():
    return load_ml100k()


@st.cache_resource
def get_algorithms():
    return default_algorithms()


data = get_data()
algorithms = get_algorithms()

measures = ("RMSE", "MSE", "MAE", "FCP")
metrics = list(measures) + ["fit_time", "test_time"]


@st.cache_data(show_spinner=True)
def run_eval(cv: int = 5, verbose: bool = True):
    results = evaluate_algorithms(
        data, algorithms, measures=measures, cv=cv, verbose=verbose
    )
    means = summarize_means(results, metrics)
    return results, means


results, results_mean = run_eval(cv=5, verbose=True)

for metric in metrics:
    st.subheader(f"Resultados para {metric}")
    df = pd.DataFrame.from_dict(results_mean[metric], orient="index", columns=[metric])
    st.bar_chart(df)
