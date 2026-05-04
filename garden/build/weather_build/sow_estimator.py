"""Two-agent sow date estimator.

DB Reader Agent pulls plant ranges + frost dates + monthly temps from the databases.
Date Researcher Agent interprets the ranges for the specific zone/city and computes
an estimated_outdoor_sow_date per plant.
"""

import asyncio
import json

import duckdb
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# --- Parameters ---
MODEL = "gpt-4o-mini"
MONTHLY_AVG_YEARS = 5
BATCH_SIZE = 15


# --- DB Reader Tool ---

@function_tool
def read_plant_data(zipcode: str, weather_db_path: str, plant_db_path: str) -> str:
    """
    Read plant sow date ranges, frost dates, and monthly avg temps from the databases.

    Returns a formatted summary string for the date researcher to interpret.
    """
    w_conn = duckdb.connect(weather_db_path, read_only=True)
    p_conn = duckdb.connect(plant_db_path, read_only=True)

    frost_row = w_conn.execute(
        "SELECT avg_last_spring_frost, avg_first_fall_frost FROM frost_dates WHERE zipcode = ?",
        [zipcode],
    ).fetchone()

    last_spring_frost = frost_row[0] if frost_row else None
    first_fall_frost = frost_row[1] if frost_row else None

    monthly_avgs = w_conn.execute(
        """
        SELECT MONTH(date) AS month,
               ROUND(AVG(temp_max_f), 1) AS avg_max_f
        FROM weather_records
        WHERE zipcode = ?
          AND YEAR(date) >= YEAR(CURRENT_DATE) - ?
          AND temp_max_f IS NOT NULL
        GROUP BY MONTH(date)
        ORDER BY month
        """,
        [zipcode, MONTHLY_AVG_YEARS],
    ).fetchall()

    plants = p_conn.execute(
        """
        SELECT pt.name, pv.outdoor_sow_date_range
        FROM plant_types pt
        JOIN plant_varieties pv ON pt.plant_type_id = pv.plant_type_id
        ORDER BY pt.name
        """,
    ).fetchall()

    w_conn.close()
    p_conn.close()

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    lines = [
        f"Zipcode: {zipcode}",
        f"Average Last Spring Frost: {last_spring_frost}",
        f"Average First Fall Frost:  {first_fall_frost}",
        "",
        "Monthly Average High Temperatures (°F, last 5 years):",
    ]
    for month_num, avg_temp in monthly_avgs:
        lines.append(f"  {month_names[month_num - 1]}: {avg_temp}°F")

    lines += ["", "Plant Sow Date Ranges (from plant database):"]
    for name, sow_range in plants:
        lines.append(f"  - {name}: {sow_range or 'n/a'}")

    return "\n".join(lines)


# --- Agents ---

db_reader_agent = Agent(
    name="Plant DB Reader",
    instructions="""
You are a database reader agent. Your job is to retrieve plant and weather data
and present it clearly so a date researcher can compute estimated sow dates.

Call read_plant_data with the zipcode, weather_db_path, and plant_db_path provided.
Return the result exactly as given — do not summarize or omit any plant.
""",
    tools=[read_plant_data],
    model=MODEL,
)


class PlantEstimate(BaseModel):
    plant_name: str
    estimated_outdoor_sow_date: str | None
    reasoning: str


class SowEstimates(BaseModel):
    estimates: list[PlantEstimate]


date_researcher_agent = Agent(
    name="Date Researcher",
    instructions="""
You are an expert horticulturalist. You will receive a plant database summary including:
- The average last spring frost and first fall frost dates
- Monthly average high temperatures for the location
- Each plant's general outdoor sow date range from the database

For each plant, compute a specific estimated_outdoor_sow_date (YYYY-MM-DD) using these rules:

Frost-relative ranges:
- "After last frost" / "After danger of frost" → last spring frost + 1 week
- "X-Y weeks after last frost" → last spring frost + average of X and Y weeks
- "X-Y weeks before last frost" → last spring frost - average of X and Y weeks
- "As soon as soil is workable" → last spring frost - 4 weeks

Soil temperature ranges (use monthly avg temps to find first month exceeding threshold):
- "soil is 60°F" / "soil warms" → find first month where avg > 60°F, use the 1st of that month
- "soil is 70°F" / "soil at least 70°F" → find first month where avg > 70°F, use the 1st of that month

Calendar ranges:
- "Month DD - Month DD" → midpoint between the two dates
- "Month DD - DD" → midpoint

Seasonal phrases:
- "Early spring" → last spring frost - 3 weeks
- "Mid-spring" → last spring frost
- "Late spring" → last spring frost + 2 weeks
- "Spring" → last spring frost + 1 week

Fall-specific entries (e.g. "4-6 weeks before first frost in fall"):
- Use first fall frost date - average weeks

If range is exactly "n/a" → set estimated_outdoor_sow_date to null.

Always include a one-sentence reasoning for each plant.
""",
    output_type=SowEstimates,
    model=MODEL,
)


# --- Orchestrator ---

def _chunk_summary(full_summary: str, batch_size: int) -> list[str]:
    """Split a plant summary into batches of batch_size plants, preserving the header."""
    lines = full_summary.splitlines()
    header_end = next(
        (i for i, l in enumerate(lines) if l.startswith("Plant Sow Date Ranges")), len(lines)
    )
    header = "\n".join(lines[:header_end + 1])
    plant_lines = [l for l in lines[header_end + 1:] if l.strip().startswith("-")]

    chunks = []
    for i in range(0, len(plant_lines), batch_size):
        batch_plants = "\n".join(plant_lines[i:i + batch_size])
        chunks.append(f"{header}\n{batch_plants}")
    return chunks if chunks else [full_summary]


async def _run(zipcode: str, weather_db_path: str, plant_db_path: str) -> dict:
    """Run DB reader then date researcher in batches; return {plant_name: estimated_date_or_null}."""
    reader_result = await Runner.run(
        db_reader_agent,
        f"Read plant data for zipcode {zipcode}. "
        f"Weather DB: {weather_db_path}. Plant DB: {plant_db_path}.",
    )

    batches = _chunk_summary(reader_result.final_output, BATCH_SIZE)
    all_estimates: dict[str, str | None] = {}

    for batch in batches:
        researcher_result = await Runner.run(date_researcher_agent, batch)
        for e in researcher_result.final_output.estimates:
            all_estimates[e.plant_name] = e.estimated_outdoor_sow_date

    return all_estimates


async def run_async(zipcode: str, weather_db_path: str, plant_db_path: str) -> dict:
    """Async entry point for use inside an existing event loop."""
    return await _run(zipcode, weather_db_path, plant_db_path)


def run(zipcode: str, weather_db_path: str, plant_db_path: str) -> dict:
    """Synchronous entry point — runs both agents and returns estimated sow dates."""
    return asyncio.run(_run(zipcode, weather_db_path, plant_db_path))
