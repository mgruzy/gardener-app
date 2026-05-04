#!/usr/bin/env python
"""Entry point — fetch 4-day actual + 3-day forecast for the garden's location.

Run from this directory:
    cd garden/build/weather_forecast
    python main.py

Override zipcode via environment:
    GARDEN_ZIPCODE=10001 python main.py
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

import forecast_agent


# ── Parameters ────────────────────────────────────────────────────────────────

GARDEN_DATA_DIR = Path(__file__).parents[2] / "data"
WEATHER_DB_PATH = str(GARDEN_DATA_DIR / "weather.duckdb")
GARDEN_DB_PATH  = str(GARDEN_DATA_DIR / "garden.duckdb")
DEFAULT_ZIPCODE = "98115"
PHZM_URL        = "https://phzmapi.org/{zipcode}.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _garden_location(garden_db_path: str) -> tuple[str, float | None, float | None]:
    """
    Read zipcode, lat, lon from garden.duckdb.

    Returns (DEFAULT_ZIPCODE, None, None) if the database is missing or empty.
    """
    try:
        conn = duckdb.connect(garden_db_path, read_only=True)
        row = conn.execute("SELECT zipcode, lat, lon FROM garden LIMIT 1").fetchone()
        conn.close()
        if row:
            return row[0], row[1], row[2]
    except Exception:
        pass
    return DEFAULT_ZIPCODE, None, None


def _geocode(zipcode: str) -> tuple[float | None, float | None]:
    """
    Fetch lat/lon for a US zipcode from the PHZM API.

    Returns (None, None) on any network or parse error.
    """
    try:
        url = PHZM_URL.format(zipcode=zipcode)
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        coords = data.get("coordinates", {})
        return coords.get("lat"), coords.get("lon")
    except Exception as e:
        print(f"Geocode failed for {zipcode}: {e}")
        return None, None


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    """Fetch forecast data and generate a briefing for the configured garden location."""
    zipcode, lat, lon = _garden_location(GARDEN_DB_PATH)
    zipcode = os.environ.get("GARDEN_ZIPCODE", zipcode)

    if lat is None or lon is None:
        print(f"No lat/lon in garden.duckdb — geocoding {zipcode} ...")
        lat, lon = _geocode(zipcode)

    if lat is None or lon is None:
        print(f"Could not resolve coordinates for {zipcode}. Aborting.")
        return

    print(f"Fetching forecast for {zipcode} at ({lat:.4f}, {lon:.4f})")
    result = forecast_agent.run(zipcode, lat, lon, WEATHER_DB_PATH)
    print(result)
    print("Done.")


if __name__ == "__main__":
    run()
