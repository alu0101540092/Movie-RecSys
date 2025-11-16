import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import altair as alt  # type: ignore


# Configura la página de Streamlit (título, icono, layout)
def configure_streamlit():
    st.set_page_config(page_title="Movie-RecSys", page_icon="🎬", layout="wide")
    st.title("Comparación de Algoritmos de Recomendación")


# Dibuja los controles en la barra lateral y devuelve los parámetros
def sidebar_controls(all_measures: tuple[str, ...]):
    with st.sidebar:
        st.header("Parámetros")
        cv = st.slider("Número de particiones (CV)", 3, 10, 5)
        chosen_measures = st.multiselect(
            "Métricas",
            all_measures,
            default=["RMSE", "MSE", "MAE", "FCP"],
            help="Selecciona qué métricas evaluar y visualizar.",
        )
        include_time = st.toggle("Incluir tiempos (fit/test)", value=True)
        verbose = st.checkbox("Verbose", value=True)
    return cv, chosen_measures, include_time, verbose


# Construye un DataFrame resumen para graficar barras + error
def build_metric_summary_df(metric: str, results_mean: dict, results_std: dict):
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


# Crea un gráfico de barras con barras de error para un DataFrame
def chart_bar_with_error(df: pd.DataFrame, y_title: str, add_tooltip: bool = False):
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("algorithm:N", sort=list(df["algorithm"]), title="Algoritmo"),
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


# Construye la tabla de resultados para una métrica (media y std)
def build_metric_table(metric: str, results_mean: dict, results_std: dict):
    table = pd.DataFrame.from_dict(
        results_mean[metric], orient="index", columns=[metric]
    )
    table["std"] = pd.Series(results_std[metric])
    table.index.name = "Algoritmo"
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
            color=alt.Color("algorithm:N", legend=None),
        )
        .properties(height=320)
    )
    st.altair_chart(box, width="stretch")


# Dibuja el contenido de una pestaña de métrica (gráfico, tabla, boxplot)
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


# Dibuja la sección de tiempos (fit y test) con gráficos de barras
def render_time_section(results_mean: dict, results_std: dict):
    st.markdown("### Tiempos de Ejecución")
    time_tabs = st.tabs(["Tiempo de Entrenamiento (fit)", "Tiempo de Test"])
    for tab, tmetric in zip(time_tabs, ("fit_time", "test_time")):
        with tab:
            df_t = build_metric_summary_df(tmetric, results_mean, results_std)
            chart = chart_bar_with_error(df_t, tmetric, add_tooltip=True).properties(
                height=320
            )
            st.altair_chart(chart, width="stretch")
