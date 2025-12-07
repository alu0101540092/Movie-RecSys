import streamlit as st
import pandas as pd
from src.data_loader import load_movies, search_movies
from src.database import add_rating, get_user_ratings, delete_user
from src.model import get_recommendations


def render_search_tab():
    st.header("Buscar y Valorar Películas")

    query = st.text_input("Buscar por título o género")

    # Initialize session state for pagination
    if "search_page" not in st.session_state:
        st.session_state["search_page"] = 0
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    # Reset pagination if query changes
    if query != st.session_state["last_query"]:
        st.session_state["search_page"] = 0
        st.session_state["last_query"] = query

    # Load movies (cached)
    movies_df = load_movies()

    results = search_movies(query, movies_df)

    if results.empty:
        st.info("No se encontraron películas.")
    else:
        # Pagination settings
        ITEMS_PER_PAGE = 10
        total_results = len(results)
        total_pages = (total_results + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
        # Ensure current page is valid
        if st.session_state["search_page"] >= total_pages:
             st.session_state["search_page"] = max(0, total_pages - 1)
        
        current_page = st.session_state["search_page"]
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_results)
        
        st.write(f"Mostrando {start_idx + 1}-{end_idx} de {total_results} resultados:")
        
        # Slice results
        page_results = results.iloc[start_idx:end_idx]

        # Display as a list/grid
        for index, row in page_results.iterrows():
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
        
        # Pagination Controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if current_page > 0:
                if st.button("Anterior"):
                    st.session_state["search_page"] -= 1
                    st.rerun()
        with col3:
            if current_page < total_pages - 1:
                if st.button("Siguiente"):
                    st.session_state["search_page"] += 1
                    st.rerun()


def render_recommendations_tab():
    st.header("Recomendaciones para Ti")

    # --- UI Controls for Hybrid Recommendations ---
    st.subheader("Preferencias de Recomendación")
    col1, col2 = st.columns(2)
    
    with col1:
        # Load unique genres for selector
        movies_df = load_movies()
        # Collect all unique genres
        unique_genres = set()
        for g_str in movies_df['genres'].dropna():
            if g_str != "(no genres listed)":
                unique_genres.update(g_str.split('|'))
        sorted_genres = sorted(list(unique_genres))
        
        selected_genres = st.multiselect(
            "¿Qué te apetece ver hoy? (Opcional)",
            sorted_genres,
            default=[]
        )
        
    with col2:
        alpha = st.slider(
            "Balance: Género vs. Calidad",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 = Solo importa el género. 1.0 = Solo importa la calidad (SVD)."
        )
        st.caption(f"Peso Calidad: {alpha*100:.0f}% | Peso Género: {(1-alpha)*100:.0f}%")

    if st.button("Generar Recomendaciones", type="primary"):
        with st.spinner("Calculando recomendaciones..."):
            recs = get_recommendations(
                st.session_state["user_id"], 
                n=10, 
                selected_genres=selected_genres, 
                alpha=alpha
            )

        if not recs:
            st.info(
                "No hay suficientes datos para generar recomendaciones. ¡Valora algunas películas primero!"
            )
        else:
            for rec in recs:
                st.subheader(f"{rec['title']}")
                st.caption(f"Géneros: {rec['genres']}")
                
                # Display Score
                col_score, col_hybrid = st.columns(2)
                with col_score:
                     st.metric("Predicción", f"{rec['score']:.2f}/5.0")
                     
                with col_hybrid:
                     # Only show hybrid score if relevant (genres selected)
                     if selected_genres:
                         st.metric("Score Híbrido", f"{rec['hybrid_score']:.2f}")
                
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
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = False

    if not st.session_state["confirm_delete"]:
        if st.button("Eliminar Cuenta", type="primary"):
            st.session_state["confirm_delete"] = True
            st.rerun()
    else:
        st.warning("¿Estás seguro de que quieres eliminar tu cuenta? Esta acción no se puede deshacer.")
        # Adjust column weights to keep buttons closer without being stuck
        col_conf1, col_conf2, _ = st.columns([1, 1, 4])
        with col_conf1:
            if st.button("Sí, eliminar", type="primary"):
                delete_user(st.session_state["user_id"])
                st.session_state.clear()
                st.rerun()
        with col_conf2:
            if st.button("Cancelar"):
                st.session_state["confirm_delete"] = False
                st.rerun()


def dashboard_page():
    tab1, tab2, tab3 = st.tabs(["Buscar", "Recomendaciones", "Perfil"])

    with tab1:
        render_search_tab()
    with tab2:
        render_recommendations_tab()
    with tab3:
        render_profile_tab()
