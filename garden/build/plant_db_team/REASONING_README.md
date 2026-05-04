# plant_db_team — Reasoning & Design Notes

## What This Crew Does

One-shot pipeline that researches, formats, validates, and writes growing data for every plant in `plant_list.yaml` into `plant.duckdb`. Run it once to seed the database; subsequent runs skip plants that already exist.

*Note these are my plants I am interested in growing -- so update to whatever you want the crewai agents to go research. Can even add friuts!

---

## Why CrewAI for This

The plant database is a **one-time build job**, not a reactive loop. CrewAI's sequential process is a natural fit: each stage hands its output as context to the next, and each agent has a clearly scoped responsibility. No tool-calling back and forth, no state management needed.

It was also in week 2 of Ed Donner's [The Complete Agentic AI Engineering Course](https://www.udemy.com/course/the-complete-agentic-ai-engineering-course/?couponCode=25BBPMXINACTIVE) and wanted to use it to learn more about it and apply some custom functions.

---

## Inputs (`main.py` → `crew.kickoff`)

| Input | Source | Example |
|---|---|---|
| `plant_categories` | top-level keys of `plant_list.yaml` | `"vegetables, herbs"` |
| `plant_list` | all plants from `plant_list.yaml`, formatted by category | `"VEGETABLES:\n  - kale\n..."` |
| `zipcode` | `GARDEN_ZIPCODE` env var (default `98115`) | `"98115"` |
| `hardiness_zone` | USDA PHZM API lookup from zipcode | `"8b"` |
| `db_path` | `PLANT_DB_PATH` env var or computed from `__file__` | `/...../data/plant.duckdb` |

The `plant_categories` and `plant_list` are dynamic — they're derived at runtime from the yaml keys so the crew config never hardcodes "vegetables" or "herbs". Adding a new category to `plant_list.yaml` automatically propagates to every agent and task prompt.

---

## Agent Chain

```
plant_researcher → data_formatter → data_validator → database_builder
```

### plant_researcher
- Searches the web for growing data on every plant in `{plant_list}` for zone `{hardiness_zone}` / zipcode `{zipcode}`
- Sources: Old Farmer's Almanac, USDA, established gardening sites
- Output: free-text research summary organized by plant name

### data_formatter
- Takes the research summary and converts it into a strict JSON array
- One object per plant with `plant_type`, `plant_variety`, `companions`, `antagonists`, `pests`, `diseases`
- Uses `{plant_categories}` to assign the correct category to each plant
- Writes `output/all_plants_formatted.json`

### data_validator
- Reviews the JSON array for missing fields, invalid enum values, wrong units, and missing relationships
- Allowed enums: `sun_tolerance` (full_sun | partial_shade | full_shade), `water_required` (low | medium | high)
- Temperatures must be °F (20–120), distances must be inches (1–240)
- Each plant must have ≥1 pest, disease, companion, and antagonist
- Fixes all issues and writes `output/all_plants_validated.json`

### database_builder
- Calls `PlantDatabaseWriter` tool with the validated JSON and `{db_path}`
- Creates all tables if they don't exist
- Skips any plant name already in `plant_types` (idempotent)
- Writes `output/db_build_report.txt` with row counts per table

---

## Database Schema (`plant.duckdb`)

Five tables owned entirely by this crew — the main garden app reads them but never writes them.

```
plant_types         plant_type_id (PK), name (UNIQUE), category
plant_varieties     variety_id (PK), plant_type_id, variety_name, plant_category,
                    sun_tolerance, water_required, soil_n/p/k, growth_needs,
                    post_harvest_soil_needs, days_to_harvest,
                    indoor_sow_weeks_before_frost, outdoor_sow_date_range,
                    spacing_inches, harvest_timing, temp_min_air_f,
                    temp_min_ground_f, height_inches_estimate
plant_companions    companion_id (PK), plant_type_id, companion_name,
                    relationship ('companion' | 'antagonist')
plant_pests         pest_id (PK), plant_type_id, pest_name, symptoms, treatment
plant_diseases      disease_id (PK), plant_type_id, disease_name, symptoms, treatment
```

Companions and antagonists share the `plant_companions` table distinguished by the `relationship` column, since both are "plants that interact with this plant."

---

## Custom Tool — `PlantDatabaseWriter`

CrewAI agents can use tools to take actions beyond text generation. `PlantDatabaseWriter` is the only custom tool in this crew — it's the bridge between the LLM pipeline and the actual database.

**Why a tool and not just agent output?**
The `database_builder` agent could write raw SQL in its response, but that's fragile. I think the tool approach gives us a little bit of control because it uses typed interface, validated inputs, and deterministic execution. The agent decides *when* to call it; the tool handles *how* the write happens.

**How it works (`tools/custom_tool.py`):**

1. Extends `BaseTool` from `crewai.tools` with a Pydantic `args_schema` so CrewAI knows exactly what arguments to pass
2. Receives two inputs: `plant_data_json` (the full validated JSON array) and `db_path` (path to `plant.duckdb`)
3. Creates all 5 tables with `CREATE TABLE IF NOT EXISTS` — safe to call on an already-initialized DB
4. Iterates each plant object, checks if `plant_types.name` already exists, and skips it if so
5. Inserts into all 5 tables in one transaction per plant: type → variety → companions/antagonists → pests → diseases
6. Returns a plain-text report with row counts that the agent includes in its final output

**Input schema:**
```python
plant_data_json: str   # the full JSON array from validate_task
db_path: str           # absolute path to plant.duckdb
```

**Why companions and antagonists share one table:**
Both describe "plants that interact with this plant" — the only difference is whether the relationship helps or hurts. A single `plant_companions` table with a `relationship` column (`'companion'` | `'antagonist'`) keeps queries simple and avoids duplicating the schema.

---

## File Map

```
config/plant_list.yaml      source of truth for which plants to research
config/agents.yaml          agent roles, goals, backstories (all use {plant_categories} template)
config/tasks.yaml           task descriptions and expected outputs
crew.py                     wires agents + tasks into a sequential Crew
main.py                     loads yaml, resolves hardiness zone, kicks off crew
tools/custom_tool.py        PlantDatabaseWriter — creates schema and inserts records
output/                     intermediary JSON files + db build report (git-ignored)
```

---

## How to Run

```bash
cd garden/build/plant_db_team

# default (Seattle 98115)
uv run plant_db_team

# custom zipcode and DB path
GARDEN_ZIPCODE=10001 PLANT_DB_PATH=/path/to/plant.duckdb uv run plant_db_team
```

The hardiness zone is resolved automatically from the zipcode via `https://phzmapi.org/{zipcode}.json`. Falls back to `6b` if the API is unreachable.

---

## Pre-built Database

The `plant.duckdb` in `garden/data/` already contains all 74 vegetables and herbs from `plant_list.yaml`. **You do not need to run the crew to get those** — the database is ready to use.

Full run cost: ~$3.60 with GPT-4o (8 batches of 10 plants each).

---

## Adding More Plants

To add new plants, replace the contents of `plant_list.yaml` with only the new plants you want to research, then re-run the crew. The `PlantDatabaseWriter` checks the database before each insert — anything already there is skipped, and only the new plants are appended.

```bash
# example: replace plant_list.yaml with just your new additions, then:
uv run plant_db_team
```
