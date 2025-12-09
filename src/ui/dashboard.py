import streamlit as st
import pandas as pd
from src.data_loader import load_movies, search_movies
from src.database import add_rating, get_user_ratings, delete_user, get_user_genres, update_user_genres
from src.model import get_recommendations
from src.utils import translate_genres, get_spanish_genres_list, get_english_genre


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
            with st.expander(f"{row['title']} ({translate_genres(row['genres'])})"):
                # Rating slider
                # Integer rating (1-5 stars)
                # st.feedback returns 0-4
                rating_idx = st.feedback("stars", key=f"rate_{row['movieId']}")
                
                if st.button("Enviar Valoración", key=f"btn_{row['movieId']}"):
                    if rating_idx is not None:
                        final_rating = rating_idx + 1
                        add_rating(
                            st.session_state["user_id"], row["movieId"], final_rating
                        )
                        st.success(
                            f"Valoraste '{row['title']}' con {final_rating} estrellas."
                        )
                    else:
                        st.warning("Por favor, selecciona una puntuación antes de enviar.")
        
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
    
    # Load unique genres for selector
    sorted_genres = get_spanish_genres_list()
    
    selected_genres_es = st.multiselect(
        "¿Qué te apetece ver hoy? (Opcional)",
        sorted_genres,
        default=[],
        placeholder="Elige una opción"
    )
    
    # Convert back to English for model query
    selected_genres = [get_english_genre(g) for g in selected_genres_es]

    if st.button("Generar Recomendaciones", type="primary"):
        with st.spinner("Calculando recomendaciones..."):
            recs = get_recommendations(
                st.session_state["user_id"], 
                n=10, 
                selected_genres=selected_genres, 
                alpha=0.5 # Fixed alpha
            )

        if not recs:
            st.info(
                "No hay suficientes datos para generar recomendaciones. ¡Valora algunas películas primero!"
            )
        else:
            for rec in recs:
                st.subheader(f"{rec['title']}")
                st.caption(f"Géneros: {translate_genres(rec['genres'])}")
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
        
        # Sort by timestamp descending (newest first)
        if "timestamp" in merged.columns:
            merged = merged.sort_values(by="timestamp", ascending=False)
        
        # Rename columns
        merged = merged.rename(columns={
            "title": "Título",
            "rating": "Puntuación", 
            "genres": "Géneros"
        })
        
        merged["Géneros"] = merged["Géneros"].apply(translate_genres)
        
        # Reset index to start at 1 and set name
        merged.index = range(1, len(merged) + 1)
        merged.index.name = "Nº"
        
        # Select and display with styling
        st.dataframe(
            merged[["Título", "Puntuación", "Géneros"]].style
            .format({"Puntuación": "{:.0f}"})
            .set_properties(subset=["Título", "Puntuación"], **{'text-align': 'center'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ]),
            use_container_width=True
        )
    else:
        st.info("Aún no has valorado ninguna película.")

    st.markdown("---")
    st.subheader("Géneros Favoritos")
    
    # Load available genres and user's current genres
    available_genres = get_spanish_genres_list()
    current_genres_en = get_user_genres(st.session_state["user_id"])
    
    # Convert current English genres to Spanish for display/selection
    current_genres_es = [translate_genres(g) for g in current_genres_en]
    # Handle potentially malformed or empty translations that might not be in available_genres
    # filter just in case
    current_genres_es = [g for g in current_genres_es if g in available_genres]

    selected_genres_es = st.multiselect(
        "Edita tus géneros preferidos:",
        available_genres,
        default=current_genres_es,
        placeholder="Elige una opción"
    )

    if st.button("Guardar Géneros"):
        # Convert back to English for storage
        new_genres_en = [get_english_genre(g) for g in selected_genres_es]
        update_user_genres(st.session_state["user_id"], new_genres_en)
        st.success("¡Géneros actualizados correctamente!")
        # Optional: rerun to ensure state is consistent if we use this elsewhere immediately
        # st.rerun()

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
