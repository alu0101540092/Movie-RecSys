from pathlib import Path
import sys


# Función principal que construye la aplicación Streamlit
def run_app():
    ROOT = str(Path(__file__).resolve().parents[1])
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import streamlit as st
    import pandas as pd
    import altair as alt

    from src.datasets import load_ml100k
    from src.algorithms import default_algorithms
    from src.evaluate import evaluate_algorithms, summarize_means
    from src.evaluate import summarize_stds, results_to_long_df

    st.set_page_config(page_title="Movie-RecSys", page_icon="🎬", layout="wide")
    st.title("Comparación de Algoritmos de Recomendación")

    # Cachea carga de datos
    @st.cache_resource
    def get_data():
        return load_ml100k()

    # Cachea diccionario de algoritmos
    @st.cache_resource
    def get_algorithms():
        return default_algorithms()

    data = get_data()
    algorithms = get_algorithms()

    # Métricas de calidad disponibles
    ALL_MEASURES = ("RMSE", "MSE", "MAE", "FCP")

    # Sidebar de control
    with st.sidebar:
        st.header("Parámetros")
        cv = st.slider("Número de particiones (CV)", 3, 10, 5)
        chosen_measures = st.multiselect(
            "Métricas",
            ALL_MEASURES,
            default=list(ALL_MEASURES),
            help="Selecciona qué métricas evaluar y visualizar.",
        )
        include_time = st.toggle("Incluir tiempos (fit/test)", value=True)
        verbose = st.checkbox("Verbose", value=True)

    # Ejecuta evaluación cacheada
    @st.cache_data(show_spinner=True)
    def run_eval(cv: int, measures: tuple[str, ...], verbose: bool, include_time: bool):
        results = evaluate_algorithms(
            data, algorithms, measures=measures, cv=cv, verbose=verbose
        )
        metrics_to_summarize = list(measures) + (
            ["fit_time", "test_time"] if include_time else []
        )
        means = summarize_means(results, metrics_to_summarize)
        stds = summarize_stds(results, metrics_to_summarize)
        df_long = results_to_long_df(
            results, measures=measures, include_time=include_time
        )
        return results, means, stds, df_long

    results, results_mean, results_std, df_long = run_eval(
        cv=cv,
        measures=tuple(chosen_measures),
        verbose=verbose,
        include_time=include_time,
    )

    # Tabs por métrica seleccionada
    tabs = st.tabs(chosen_measures if chosen_measures else ["Sin métricas"])
    for t, metric in zip(tabs, chosen_measures):
        with t:
            st.subheader(f"Métrica: {metric}")
            df_plot = pd.DataFrame(
                {
                    "algorithm": list(results_mean[metric].keys()),
                    "mean": list(results_mean[metric].values()),
                }
            )
            df_plot["std"] = df_plot["algorithm"].map(results_std[metric])
            df_plot["ymin"] = df_plot["mean"] - df_plot["std"]
            df_plot["ymax"] = df_plot["mean"] + df_plot["std"]
            asc = metric in ("RMSE", "MSE", "MAE")
            df_plot = df_plot.sort_values("mean", ascending=asc)

            bars = (
                alt.Chart(df_plot)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("algorithm:N", sort=list(df_plot["algorithm"])),
                    y=alt.Y("mean:Q", title=metric),
                    color=alt.Color("algorithm:N", legend=None),
                    tooltip=[
                        alt.Tooltip("algorithm:N", title="Algoritmo"),
                        alt.Tooltip("mean:Q", title=f"{metric} (media)", format=".4f"),
                        alt.Tooltip("std:Q", title="Desv. Est.", format=".4f"),
                    ],
                )
            )
            errs = (
                alt.Chart(df_plot)
                .mark_errorbar()
                .encode(
                    x=alt.X("algorithm:N", sort=list(df_plot["algorithm"])),
                    y=alt.Y("ymin:Q", title=metric),
                    y2="ymax:Q",
                    color=alt.Color("algorithm:N", legend=None),
                )
            )
            st.altair_chart((bars + errs).properties(height=380), width="stretch")

            table = pd.DataFrame.from_dict(
                results_mean[metric], orient="index", columns=[metric]
            )
            table["std"] = pd.Series(results_std[metric])
            table = table.sort_values(metric, ascending=asc)
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
                df_metric = df_long[df_long["metric"] == metric]
                if not df_metric.empty:
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
                else:
                    st.info("Sin datos para esta métrica.")

    # Sección de tiempos
    if include_time:
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

                bars_t = (
                    alt.Chart(df_t)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("algorithm:N", sort=list(df_t["algorithm"])),
                        y=alt.Y("mean:Q", title=tmetric),
                        color=alt.Color("algorithm:N", legend=None),
                    )
                )
                errs_t = (
                    alt.Chart(df_t)
                    .mark_errorbar()
                    .encode(
                        x=alt.X("algorithm:N", sort=list(df_t["algorithm"])),
                        y=alt.Y("ymin:Q", title=tmetric),
                        y2="ymax:Q",
                        color=alt.Color("algorithm:N", legend=None),
                    )
                )
                st.altair_chart(
                    (bars_t + errs_t).properties(height=320), width="stretch"
                )


# Punto de entrada del módulo
if __name__ == "__main__":
    run_app()
