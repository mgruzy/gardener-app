"""Garden app — Streamlit entry point."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

GARDEN_ROOT = Path(__file__).parent.parent
DATA_DIR = GARDEN_ROOT / "data"
sys.path.insert(0, str(GARDEN_ROOT))

load_dotenv(GARDEN_ROOT.parent / ".env")

from app.db.config import AppConfig, load_config
from app.db.connections import open_connections
from app.db.schemas import initialize_all_schemas
import app.internal_objects.garden_types as garden_enum

# --- Page config ---
st.set_page_config(
    page_title="Garden",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Bootstrap DB connections once per session ---
if "db" not in st.session_state:
    config = AppConfig(
        mode=garden_enum.AppMode.LIVE,
        garden_db_path=str(DATA_DIR / "garden.duckdb"),
        weather_db_path=str(DATA_DIR / "weather.duckdb"),
        chat_memory_db_path=str(DATA_DIR / "chat_memory.duckdb"),
        plant_db_path=str(DATA_DIR / "plant.duckdb"),
    )
    db = open_connections(config)
    initialize_all_schemas(db)
    st.session_state.db = db
    st.session_state.data_dir = DATA_DIR

# --- App header ---
st.title("🌱 Garden")

# --- Tabs ---
tab_map, tab_plants, tab_weather = st.tabs(["🗺️ Map", "🌿 Plants", "🌤️ Weather"])

with tab_map:
    from app.pages.map_page import render
    render(st.session_state.db, st.session_state.data_dir)

with tab_plants:
    from app.pages.plants_page import render
    render(st.session_state.db)

with tab_weather:
    from app.pages.weather_page import render
    render(st.session_state.db)
