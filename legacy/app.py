import streamlit as st  # type: ignore
from src.components import (
    configure_streamlit,
    sidebar_controls,
    draw_metric_tab,
    render_time_section,
)
from src.services import run_evaluation, get_static_results


def render_dynamic_results(
    cv: int, chosen_measures: list[str], include_time: bool, verbose: bool
) -> None:
    """
    Renders the dynamic results section (ML-100k).

    Args:
        cv (int): Number of cross-validation folds.
        chosen_measures (list[str]): List of selected metrics.
        include_time (bool): Whether to include execution time.
        verbose (bool): Whether to show verbose output.
    """
    st.header("Resultados ML-100k (Dinámico)")

    _, results_mean, results_std, df_long = run_evaluation(
        cv=cv,
        measures=tuple(chosen_measures),
        verbose=verbose,
        include_time=include_time,
    )

    tabs = st.tabs(chosen_measures)
    for t, metric in zip(tabs, chosen_measures):
        with t:
            draw_metric_tab(
                metric, results_mean, results_std, df_long, key_prefix="dynamic"
            )

    if include_time:
        render_time_section(results_mean, results_std)


def render_static_results(
    chosen_measures: list[str], include_time: bool
) -> None:
    """
    Renders the static results section (ML-32m).

    Args:
        chosen_measures (list[str]): List of selected metrics.
        include_time (bool): Whether to include execution time.
    """
    st.markdown("---")
    st.header("Resultados ML-32m (Pre-calculados)")
    st.info(
        """
        Estos resultados provienen de una ejecución previa sobre el dataset de 32 millones de valoraciones.

        **Nota:** Los algoritmos `Slope One` y `KNNBasic` se ejecutaron sobre un 10% de este dataset debido a restricciones de memoria.
        """
    )

    static_mean, static_std, static_df_long = get_static_results()

    if static_df_long.empty:
        st.error(
            "No se encontraron resultados pre-calculados en results/ml-32m.txt"
        )
        return

    static_measures = [m for m in chosen_measures if m in static_mean]
    if not static_measures:
        return

    st_tabs = st.tabs(static_measures)
    for t, metric in zip(st_tabs, static_measures):
        with t:
            draw_metric_tab(
                metric,
                static_mean,
                static_std,
                static_df_long,
                key_prefix="static",
            )

    if include_time:
        render_time_section(static_mean, static_std, key_prefix="static_")


def main() -> None:
    """
    Main function that orchestrates the Streamlit application.
    """
    configure_streamlit()

    ALL_MEASURES = ("RMSE", "MSE", "MAE", "FCP")
    cv, chosen_measures, include_time, verbose = sidebar_controls(ALL_MEASURES)

    if not chosen_measures:
        st.warning(
            "Por favor, selecciona al menos una métrica en la barra lateral."
        )
        return

    render_dynamic_results(cv, chosen_measures, include_time, verbose)
    render_static_results(chosen_measures, include_time)


if __name__ == "__main__":
    main()
