import streamlit as st  # type: ignore
from src.components import (
    configure_streamlit,
    sidebar_controls,
    draw_metric_tab,
    render_time_section,
)
from src.services import run_evaluation


# Función principal que orquesta la aplicación de Streamlit
def main():
    configure_streamlit()

    ALL_MEASURES = ("RMSE", "MSE", "MAE", "FCP")
    cv, chosen_measures, include_time, verbose = sidebar_controls(ALL_MEASURES)

    if not chosen_measures:
        st.warning("Por favor, selecciona al menos una métrica en la barra lateral.")
        return

    _, results_mean, results_std, df_long = run_evaluation(
        cv=cv,
        measures=tuple(chosen_measures),
        verbose=verbose,
        include_time=include_time,
    )

    tabs = st.tabs(chosen_measures)
    for t, metric in zip(tabs, chosen_measures):
        with t:
            draw_metric_tab(metric, results_mean, results_std, df_long)

    if include_time:
        render_time_section(results_mean, results_std)


if __name__ == "__main__":
    main()
