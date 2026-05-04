"""Weather forecast agent — fetches 4 days of actual weather + 3-day forecast and summarizes."""

import asyncio

from agents import Agent, Runner
from dotenv import load_dotenv

from function_tools import fetch_recent_and_forecast, generate_and_store_summary

load_dotenv()


# ── Parameters ────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"


# ── Agent ─────────────────────────────────────────────────────────────────────

_forecast_agent = Agent(
    name="Weather Forecast Agent",
    instructions="""
You are a weather data agent for a garden app. Fetch recent actual weather and the upcoming
3-day forecast for a garden location, then generate a gardener-focused briefing.

Steps:
1. Call fetch_recent_and_forecast with the zipcode, lat, lon, and db_path.
   This fetches 4 days of actual data from the Open-Meteo archive and today + 3 forecast
   days from the Open-Meteo forecast API, writing everything to weather.duckdb.
2. Call generate_and_store_summary with the zipcode and db_path.
   This reads the stored records, calls OpenAI to produce a 2-3 sentence gardener briefing,
   and persists it in weather_forecast_summary.
3. Report how many actual and forecast records were written, plus the summary text.
""",
    tools=[fetch_recent_and_forecast, generate_and_store_summary],
    model=MODEL,
)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(zipcode: str, lat: float, lon: float, db_path: str) -> str:
    """
    Run the forecast agent for the given location and return the final output.

    Args:
        zipcode: US zipcode for the garden.
        lat: Latitude of the location.
        lon: Longitude of the location.
        db_path: Absolute path to weather.duckdb.

    Returns:
        The agent's final text output summarising what was fetched and the briefing.
    """
    result = asyncio.run(
        Runner.run(
            _forecast_agent,
            f"Fetch recent weather and 3-day forecast for zipcode {zipcode} "
            f"at lat={lat}, lon={lon}. Store at {db_path}.",
        )
    )
    return result.final_output
