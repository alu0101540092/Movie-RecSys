import streamlit as st
from src.database import create_user, authenticate_user
from src.utils import get_spanish_genres_list, get_english_genre


def render_login():
    st.header("Iniciar Sesión")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Entrar")

        if submit:
            user = authenticate_user(username, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user["id"]
                st.session_state["username"] = user["username"]
                st.success(f"Bienvenido {user['username']}")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos")


def render_register():
    st.header("Registrarse")
    with st.form("register_form"):
        username = st.text_input("Usuario")
        email = st.text_input("Email")
        password = st.text_input("Contraseña", type="password")

        # Genre selection
        genres = get_spanish_genres_list()
        favorite_genres_es = st.multiselect("Géneros Favoritos", genres)
        # Map back to English for storage
        favorite_genres = [get_english_genre(g) for g in favorite_genres_es]

        submit = st.form_submit_button("Crear Cuenta")

        if submit:
            if username and email and password:
                if create_user(username, email, password, favorite_genres):
                    st.success(
                        "Cuenta creada exitosamente. Por favor inicia sesión."
                    )
                else:
                    st.error("El usuario o email ya existe.")
            else:
                st.error("Por favor, completa todos los campos.")


def auth_page():
    tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])
    with tab1:
        render_login()
    with tab2:
        render_register()
