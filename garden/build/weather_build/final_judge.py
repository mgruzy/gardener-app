"""Final Judge Agent — resolves disagreements between researched and estimated sow dates."""

import asyncio

from agents import Agent, Runner
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# --- Parameters ---
MODEL = "gpt-4o-mini"


# --- Pydantic Models ---

class JudgedDate(BaseModel):
    plant_name: str
    recommended_date: str
    chosen_source: str  # "researched" | "estimated" | "compromise"
    reasoning: str


class JudgedDates(BaseModel):
    judgments: list[JudgedDate]


# --- Agent ---

final_judge_agent = Agent(
    name="Final Judge",
    instructions="""
You are the final judge for plant sow date disagreements. You receive a list of plants
where the researched date (from regional LLM knowledge) and the estimated date (derived
from the plant database range + historical temperature data) differ by more than 14 days.

For each plant, pick the better recommended_date and explain why. Use these rules:

Prefer the RESEARCHED date when:
- sow_source is "researched" — the LLM had high-confidence regional knowledge
- The plant is a warm-season crop (tomato, pepper, squash, cucumber, melon, basil,
  eggplant, bean) and the researched date is later (more conservative, avoids frost)

Prefer the ESTIMATED date when:
- sow_source is "fallback" — the LLM was not confident, estimated is more grounded
- The estimated date was derived from a soil temperature threshold (60°F or 70°F),
  which is more scientifically specific than a generic frost offset
- The general range from the plant database closely matches the estimated date

Use "compromise" and pick a midpoint when both sources have equal merit.

Return a recommended_date in YYYY-MM-DD format and a one-sentence reasoning for each plant.
""",
    output_type=JudgedDates,
    model=MODEL,
)


# --- Runner ---

async def run_async(disagreements_summary: str) -> dict[str, dict]:
    """Run the final judge agent and return {plant_name: {recommended_date, chosen_source, reasoning}}."""
    result = await Runner.run(final_judge_agent, disagreements_summary)
    return {
        j.plant_name: {
            "recommended_date": j.recommended_date,
            "chosen_source": j.chosen_source,
            "reasoning": j.reasoning,
        }
        for j in result.final_output.judgments
    }


def run(disagreements_summary: str) -> dict[str, dict]:
    """Synchronous entry point."""
    return asyncio.run(run_async(disagreements_summary))
