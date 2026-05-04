"""Weather history agent — fetches 20 years of daily weather into weather.duckdb."""

import asyncio

from agents import Agent, Runner
from dotenv import load_dotenv

from function_tools import fetch_and_write_weather, geocode_zipcode

load_dotenv()

# --- Parameters ---
MODEL = "gpt-4o-mini"

# --- Agent ---

weather_agent = Agent(
    name="Weather History Agent",
    instructions="""
You are a weather data agent. Fetch 20 years of historical daily weather for a zipcode and store it in weather.duckdb.

Steps:
1. Call geocode_zipcode to get the latitude and longitude for the zipcode.
2. Call fetch_and_write_weather with the coordinates, zipcode, and db_path.
3. Report the date range covered and total records written.
""",
    tools=[geocode_zipcode, fetch_and_write_weather],
    model=MODEL,
)


def run(zipcode: str, db_path: str) -> str:
    """Run the weather history agent for a given zipcode and write to db_path."""
    result = asyncio.run(
        Runner.run(
            weather_agent,
            f"Fetch 20 years of weather history for zipcode {zipcode} and store it at {db_path}.",
        )
    )
    return result.final_output
