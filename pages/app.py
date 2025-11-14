from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import altair as alt


# Asegura que el directorio raíz del proyecto esté en sys.path para importar src.*
def ensure_sys_path():
    ROOT = str(Path(__file__).resolve().parents[1])
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


ensure_sys_path()

from src.datasets import load_ml100k
from src.algorithms import default_algorithms
from src.evaluate import (
    evaluate_algorithms,
    summarize_means,
    summarize_stds,
    results_to_long_df,
)


# Configura la página de Streamlit (título, icono, layout)
def configure_streamlit():
    st.set_page_config(page_title="Movie-RecSys", page_icon="🎬", layout="wide")
    st.title("Comparación de Algoritmos de Recomendación")


# Cachea y devuelve el dataset MovieLens 100k
@st.cache_resource
def get_data():
    return load_ml100k()


# Cachea y devuelve el diccionario de algoritmos por defecto
@st.cache_resource
def get_algorithms():
    return default_algorithms()


# Dibuja los controles en la barra lateral y devuelve los parámetros seleccionados
def sidebar_controls(all_measures: tuple[str, ...]):
    with st.sidebar:
        st.header("Parámetros")
        cv = st.slider("Número de particiones (CV)", 3, 10, 5)
        chosen_measures = st.multiselect(
            "Métricas",
            all_measures,
            default=list(all_measures),
            help="Selecciona qué métricas evaluar y visualizar.",
        )
        include_time = st.toggle("Incluir tiempos (fit/test)", value=True)
        verbose = st.checkbox("Verbose", value=True)
    return cv, chosen_measures, include_time, verbose


# Ejecuta la evaluación con cache y devuelve resultados, medias, desviaciones y DF largo
@st.cache_data(show_spinner=True)
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


# Construye un DataFrame resumen para graficar barras + error por métrica
def build_metric_summary_df(
    metric: str, results_mean: dict, results_std: dict
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "algorithm": list(results_mean[metric].keys()),
            "mean": list(results_mean[metric].values()),
        }
    )
    df["std"] = df["algorithm"].map(results_std[metric])
    df["ymin"] = df["mean"] - df["std"]
    df["ymax"] = df["mean"] + df["std"]
    asc = metric in ("RMSE", "MSE", "MAE")
    return df.sort_values("mean", ascending=asc)


# Crea un gráfico de barras con barras de error para un DataFrame dado
def chart_bar_with_error(df: pd.DataFrame, y_title: str, add_tooltip: bool = False):
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("algorithm:N", sort=list(df["algorithm"])),
            y=alt.Y("mean:Q", title=y_title),
            color=alt.Color("algorithm:N", legend=None),
        )
    )
    if add_tooltip:
        bars = bars.encode(
            tooltip=[
                alt.Tooltip("algorithm:N", title="Algoritmo"),
                alt.Tooltip("mean:Q", title=f"{y_title} (media)", format=".4f"),
                alt.Tooltip("std:Q", title="Desv. Est.", format=".4f"),
            ]
        )
    errs = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            x=alt.X("algorithm:N", sort=list(df["algorithm"])),
            y=alt.Y("ymin:Q", title=y_title),
            y2="ymax:Q",
            color=alt.Color("algorithm:N", legend=None),
        )
    )
    return bars + errs


# Construye la tabla de resultados para una métrica (media y desviación)
def build_metric_table(
    metric: str, results_mean: dict, results_std: dict
) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(
        results_mean[metric], orient="index", columns=[metric]
    )
    table["std"] = pd.Series(results_std[metric])
    asc = metric in ("RMSE", "MSE", "MAE")
    return table.sort_values(metric, ascending=asc)


# Dibuja el boxplot por fold para una métrica si hay datos
def draw_boxplot(metric: str, df_long: pd.DataFrame):
    df_metric = df_long[df_long["metric"] == metric]
    if df_metric.empty:
        st.info("Sin datos para esta métrica.")
        return
    box = (
        alt.Chart(df_metric)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("algorithm:N", title="Algoritmo"),
            y=alt.Y("value:Q", title=metric),
            color="algorithm:N",
        )
        .properties(height=320)
    )
    st.altair_chart(box, width="stretch")


# Dibuja el contenido de una pestaña de métrica (gráfico, tabla y boxplot)
def draw_metric_tab(
    metric: str, results_mean: dict, results_std: dict, df_long: pd.DataFrame
):
    st.subheader(f"Métrica: {metric}")
    df_plot = build_metric_summary_df(metric, results_mean, results_std)
    chart = chart_bar_with_error(df_plot, metric, add_tooltip=True).properties(
        height=380
    )
    st.altair_chart(chart, width="stretch")

    table = build_metric_table(metric, results_mean, results_std)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(table, width="stretch")
    with c2:
        st.download_button(
            "Descargar CSV",
            table.to_csv(index=True).encode("utf-8"),
            file_name=f"results_{metric.lower()}.csv",
            mime="text/csv",
        )

    with st.expander("Ver distribución por fold (boxplot)"):
        draw_boxplot(metric, df_long)


# Dibuja la sección de tiempos (fit_time y test_time) con gráficos de barras+error
def render_time_section(results_mean: dict, results_std: dict):
    st.markdown("### Tiempos")
    time_tabs = st.tabs(["fit_time", "test_time"])
    for tab, tmetric in zip(time_tabs, ("fit_time", "test_time")):
        with tab:
            df_t = pd.DataFrame(
                {
                    "algorithm": list(results_mean[tmetric].keys()),
                    "mean": list(results_mean[tmetric].values()),
                }
            )
            df_t["std"] = df_t["algorithm"].map(results_std[tmetric])
            df_t["ymin"] = df_t["mean"] - df_t["std"]
            df_t["ymax"] = df_t["mean"] + df_t["std"]
            df_t = df_t.sort_values("mean", ascending=True)
            chart = chart_bar_with_error(df_t, tmetric, add_tooltip=False).properties(
                height=320
            )
            st.altair_chart(chart, width="stretch")


# Función principal que orquesta la app y garantiza funciones < 40 líneas
def run_app():
    configure_streamlit()
    ALL_MEASURES = ("RMSE", "MSE", "MAE", "FCP")
    cv, chosen_measures, include_time, verbose = sidebar_controls(ALL_MEASURES)

    _, results_mean, results_std, df_long = run_evaluation(
        cv=cv,
        measures=tuple(chosen_measures),
        verbose=verbose,
        include_time=include_time,
    )

    tabs = st.tabs(chosen_measures if chosen_measures else ["Sin métricas"])
    for t, metric in zip(tabs, chosen_measures):
        with t:
            draw_metric_tab(metric, results_mean, results_std, df_long)

    if include_time:
        render_time_section(results_mean, results_std)


# Punto de entrada del módulo
if __name__ == "__main__":
    run_app()
