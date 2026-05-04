"""Frost date agent — calculates frost dates and sow calendar from weather history."""

import asyncio

from agents import Agent, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered, Runner, output_guardrail
from dotenv import load_dotenv
from pydantic import BaseModel

from function_tools import (
    build_sow_calendar,
    calculate_and_write_frost_dates,
    estimate_sow_dates,
    geocode_zipcode,
    judge_sow_dates,
    research_sow_dates,
)

load_dotenv()

# --- Parameters ---
MODEL = "gpt-4o-mini"

class SowDateEvaluation(BaseModel):
    looks_reasonable: bool
    concern: str

sow_date_checker = Agent(
    name="Sow Date Checker",
    instructions="""
You are reviewing a planting calendar summary for a US zipcode.

IMPORTANT: It is completely normal for cool-season crops (arugula, lettuce, spinach, beets,
carrots, peas, kale, chard, broccoli, cabbage, asparagus, etc.) to have outdoor sow dates
BEFORE the last spring frost. These crops tolerate frost and are often planted 2-6 weeks
early. Do NOT flag these as problems.

Set looks_reasonable=False only for genuinely wrong situations:
- The last spring frost date falls in summer (July or later).
- The total plants written is zero.
- A warm-season crop (tomato, pepper, squash, cucumber, melon, basil, eggplant, bean)
  has an outdoor sow date in January or February.
- Any sow date is more than 3 months before the last spring frost.

Otherwise set looks_reasonable=True.
""",
    output_type=SowDateEvaluation,
    model=MODEL,
)


@output_guardrail
async def evaluate_sow_calendar(ctx, _agent, output):
    """Check the agent's sow calendar summary looks plausible before returning it."""
    result = await Runner.run(sow_date_checker, output, context=ctx.context)
    evaluation = result.final_output
    return GuardrailFunctionOutput(
        output_info=evaluation,
        tripwire_triggered=not evaluation.looks_reasonable,
    )


# --- Agent ---

frost_agent = Agent(
    name="Frost Date Agent",
    instructions="""
You are a frost date and planting calendar agent.

Steps:
1. Call geocode_zipcode with the zipcode to get the hardiness zone and coordinates.
2. Call calculate_and_write_frost_dates with the zipcode and weather_db_path.
3. Call research_sow_dates with the zipcode, zone (from step 1), a city name based on the
   location (e.g. "Seattle area" for 98115), and the plant_db_path.
4. Call estimate_sow_dates with the zipcode, weather_db_path, and plant_db_path.
   This runs a two-agent pipeline that reads plant ranges and historical temps from the
   databases and computes a date for each plant. It returns a JSON string.
5. Call build_sow_calendar with the zipcode, weather_db_path, plant_db_path,
   the researched_dates_json from step 3 (serialized as JSON string),
   and the estimated_dates_json from step 4.
6. Call judge_sow_dates with the zipcode and weather_db_path. This runs the Final Judge
   Agent which reviews plants where researched and estimated dates disagree by more than
   14 days and writes a recommended_date to the sow_calendar.
7. Report the frost dates found, how many plants used researched vs fallback dates,
   how many disagreements the judge resolved, and the total plants in the calendar.
""",
    tools=[geocode_zipcode, calculate_and_write_frost_dates, research_sow_dates, estimate_sow_dates, build_sow_calendar, judge_sow_dates],
    output_guardrails=[evaluate_sow_calendar],
    model=MODEL,
)


def run(zipcode: str, weather_db_path: str, plant_db_path: str) -> str:
    """Run the frost date agent and build the sow calendar for a given zipcode."""
    try:
        result = asyncio.run(
            Runner.run(
                frost_agent,
                f"Calculate frost dates and build sow calendar for zipcode {zipcode}. "
                f"Weather DB: {weather_db_path}. Plant DB: {plant_db_path}.",
                max_turns=25,
            )
        )
        return result.final_output
    except OutputGuardrailTripwireTriggered as e:
        evaluation = e.guardrail_result.output.output_info
        return f"Guardrail blocked output. Concern: {evaluation.concern}"
