import streamlit as st
from src.database import init_db
from src.ui.auth import auth_page
from src.ui.dashboard import dashboard_page


def main():
    """
    Main entry point for the Streamlit application.

    This function sets up the page configuration, initializes the database,
    manages the user session state, and routes the user to the appropriate
    page (authentication or dashboard) based on their login status.
    """
    st.set_page_config(page_title="Movie Recommender", layout="wide")

    # Initialize the database (create tables if they don't exist)
    init_db()

    # Initialize Session State for login management
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    st.title("🎬 Sistema Recomendador de Películas")

    # Routing logic: Show Auth page if not logged in, otherwise show Dashboard
    if not st.session_state["logged_in"]:
        auth_page()
    else:
        # Sidebar with user info and logout button
        with st.sidebar:
            st.write(f"Hola, **{st.session_state['username']}**")
            if st.button("Cerrar Sesión"):
                st.session_state.clear()
                st.rerun()

        dashboard_page()


if __name__ == "__main__":
    main()
