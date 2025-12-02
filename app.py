import streamlit as st
from src.database import init_db
from src.ui.auth import auth_page
from src.ui.dashboard import dashboard_page

def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    
    # Initialize DB
    init_db()
    
    # Session State
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        
    st.title("🎬 Sistema Recomendador de Películas")
    
    if not st.session_state['logged_in']:
        auth_page()
    else:
        with st.sidebar:
            st.write(f"Hola, **{st.session_state['username']}**")
            if st.button("Cerrar Sesión"):
                st.session_state.clear()
                st.rerun()
        
        dashboard_page()

if __name__ == "__main__":
    main()
