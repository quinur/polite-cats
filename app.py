"""Streamlit entry point for WHL performance prediction engine."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
from main import get_data, get_model
from src.dashboard import run_dashboard

@st.cache_resource
def load_all_data():
    """Load data and model once and cache them."""
    game_df, team_df = get_data()
    model, _ = get_model(game_df)
    
    return game_df, team_df, model

if __name__ == "__main__":
    st.set_page_config(
        page_title="Polite Cats | WHL Analytics",
        page_icon="🏒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    game_df, team_df, model = load_all_data()
    run_dashboard(game_df, team_df, model)
