import streamlit as st
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
