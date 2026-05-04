#!/usr/bin/env python
"""Entry point — runs weather history fetch then frost date and sow calendar build."""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

import frost_dates
import weather_history

# --- Parameters ---
GARDEN_DATA_DIR = Path(__file__).parents[2] / "data"
WEATHER_DB_PATH = str(GARDEN_DATA_DIR / "weather.duckdb")
PLANT_DB_PATH = str(GARDEN_DATA_DIR / "plant.duckdb")
DEFAULT_ZIPCODE = "98115"  # seattle
DISAGREEMENT_THRESHOLD_DAYS = 14


def print_disagreements(zipcode: str) -> None:
    """Print plants where researched and estimated sow dates differ by more than 14 days."""
    conn = duckdb.connect(WEATHER_DB_PATH, read_only=True)
    rows = conn.execute(
        """
        SELECT plant_name,
               outdoor_sow_date,
               estimated_outdoor_sow_date,
               sow_source,
               ABS(DATEDIFF('day', outdoor_sow_date, estimated_outdoor_sow_date)) AS gap_days
        FROM sow_calendar
        WHERE zipcode = ?
          AND outdoor_sow_date IS NOT NULL
          AND estimated_outdoor_sow_date IS NOT NULL
          AND ABS(DATEDIFF('day', outdoor_sow_date, estimated_outdoor_sow_date)) > ?
        ORDER BY gap_days DESC
        """,
        [zipcode, DISAGREEMENT_THRESHOLD_DAYS],
    ).fetchall()
    conn.close()

    if not rows:
        print("  No significant disagreements between researched and estimated dates.")
        return

    print(f"  {'Plant':<28} {'Outdoor':<14} {'Estimated':<14} {'Gap':>5}  Source")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*5}  {'-'*10}")
    for plant, outdoor, estimated, source, gap in rows:
        source_label = source or "unknown"
        print(f"  {plant:<28} {str(outdoor):<14} {str(estimated):<14} {gap:>4}d  {source_label}")


def run() -> None:
    """Fetch weather history and build frost/sow calendar for the configured zipcode."""
    zipcode = os.environ.get("GARDEN_ZIPCODE", DEFAULT_ZIPCODE)
    GARDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Step 1/2 — Weather history ({zipcode}, {WEATHER_DB_PATH})")
    result = weather_history.run(zipcode, WEATHER_DB_PATH)
    print(result)

    print("")

    print(f"Step 2/2 — Frost dates & sow calendar ({zipcode})")
    result = frost_dates.run(zipcode, WEATHER_DB_PATH, PLANT_DB_PATH)
    print(result)

    print("\nDate disagreements (researched vs estimated, >14 days apart):")
    print_disagreements(zipcode)

    print("\nDone.")


if __name__ == "__main__":
    run()
