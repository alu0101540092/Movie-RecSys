import streamlit as st
import pandas as pd
from src.data_loader import load_movies, search_movies
from src.database import add_rating, get_user_ratings, delete_user
from src.model import get_recommendations


def render_search_tab():
    st.header("Buscar y Valorar Películas")

    query = st.text_input("Buscar por título o género")

    # Load movies (cached)
    movies_df = load_movies()

    results = search_movies(query, movies_df)

    if results.empty:
        st.info("No se encontraron películas.")
    else:
        # Pagination or limit
        st.write(f"Mostrando {len(results)} resultados:")

        # Display as a list/grid
        for index, row in results.iterrows():
            with st.expander(f"{row['title']} ({row['genres']})"):
                # Rating slider
                rating = st.slider(
                    "Tu valoración",
                    1.0,
                    5.0,
                    3.0,
                    0.5,
                    key=f"rate_{row['movieId']}",
                )
                if st.button("Enviar Valoración", key=f"btn_{row['movieId']}"):
                    add_rating(
                        st.session_state["user_id"], row["movieId"], rating
                    )
                    st.success(
                        f"Valoraste '{row['title']}' con {rating} estrellas."
                    )


def render_recommendations_tab():
    st.header("Recomendaciones para Ti")

    if st.button("Generar Recomendaciones"):
        with st.spinner("Calculando recomendaciones..."):
            recs = get_recommendations(st.session_state["user_id"])

        if not recs:
            st.info(
                "No hay suficientes datos para generar recomendaciones. ¡Valora algunas películas primero!"
            )
        else:
            for rec in recs:
                st.subheader(f"{rec['title']}")
                st.caption(f"Géneros: {rec['genres']}")
                st.write(f"Predicción: {rec['score']:.2f}/5.0")
                st.markdown("---")


def render_profile_tab():
    st.header("Mi Perfil")
    st.write(f"Usuario: {st.session_state['username']}")

    st.subheader("Mis Valoraciones")
    user_ratings = get_user_ratings(st.session_state["user_id"])
    if not user_ratings.empty:
        movies_df = load_movies()
        # Merge to get titles
        merged = user_ratings.merge(
            movies_df, left_on="movie_id", right_on="movieId"
        )
        st.dataframe(merged[["title", "rating", "genres"]])
    else:
        st.info("Aún no has valorado ninguna película.")

    st.markdown("---")
    st.subheader("Zona de Peligro")
    if st.button("Eliminar Cuenta", type="primary"):
        delete_user(st.session_state["user_id"])
        st.session_state.clear()
        st.rerun()


def dashboard_page():
    tab1, tab2, tab3 = st.tabs(["Buscar", "Recomendaciones", "Perfil"])

    with tab1:
        render_search_tab()
    with tab2:
        render_recommendations_tab()
    with tab3:
        render_profile_tab()
