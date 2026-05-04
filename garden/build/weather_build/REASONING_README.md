# weather_build — Reasoning & Design Notes

## What this does

Fetches 20 years of daily weather history for a US zipcode, calculates frost dates,
and builds a per-plant sow calendar with two date estimates per plant.

Run with:
```bash
cd garden/build/weather_build
python main.py
```

Set a different zipcode with the environment variable:
```bash
GARDEN_ZIPCODE=10001 python main.py
```

---

## Pre-built data — you may not need to run this

**Seattle (98115) data is already in the repo** (`garden/data/weather.duckdb`).
20 years of daily weather, frost dates, and a full sow calendar are pre-populated.

- If you are in Seattle → just use the database as-is, no need to run anything.
- **If you live outside Seattle**, update `DEFAULT_ZIPCODE` in `main.py` (or set the
  `GARDEN_ZIPCODE` env var) and run `python main.py`. Weather records are fetched
  from the free [Open-Meteo archive API](https://open-meteo.com/) — no API key required.

**Cost to run**: ~$0. The LLM calls (research + estimation) use `gpt-4o-mini` with small
prompts. A full run for all 74 plants costs less than a cent.

---

## Step 1 — Weather History (`weather_history.py`)

1. Geocodes the zipcode via the [PHZM API](https://phzmapi.org/) → lat/lon + hardiness zone.
2. Fetches 20 years of daily data (temp min/max, precipitation, wind) from Open-Meteo.
3. Writes to `weather_records` in `weather.duckdb`. Already-written dates are skipped
   so re-runs are safe and free.

---

## Step 2 — Frost Dates & Sow Calendar (`frost_dates.py`)

Runs a single agent (`frost_agent`) with 5 tools and an output guardrail, in order:

### Tool 1 — `geocode_zipcode`
Retrieves hardiness zone (e.g. `9a`) and coordinates. Zone is passed to the
research step so date recommendations are regionally specific.

### Tool 2 — `calculate_and_write_frost_dates`
Queries the last 10 years of `weather_records` to find:
- **Average last spring frost** — latest day in Jan–June where `temp_min_f ≤ 32°F`,
  averaged across years as a day-of-year then converted to a calendar date.
- **Average first fall frost** — earliest day in Jul–Dec where `temp_min_f ≤ 32°F`.
- **Frost-free days** — gap between the two.

For frost-free zones (no frost days found), returns `frost_free_days=365`.

### Tool 3 — `research_sow_dates`
Makes a single `gpt-4o-mini` call with all plant names + zone + city.
The LLM returns per-plant:
- `outdoor_sow_date` — a specific regional date
- `confidence` — `"high"` (use it) or `"low"` (too general, fall back)
- `note` — one-line reasoning

This gives regionally specific dates rather than generic "after last frost" advice.
Example: tomatoes in Seattle are `"high"` confidence at late May because the LLM
knows the maritime climate requires soil to warm past 60°F.

### Tool 4 — `estimate_sow_dates` (two-agent pipeline in `sow_estimator.py`)

This is where the `estimated_outdoor_sow_date` is reasoned. Two agents talk to each other:

**Agent 1 — DB Reader (`db_reader_agent`)**
Calls `read_plant_data` tool which reads from both databases and returns a
formatted summary with:
- Frost dates for the zipcode
- Monthly average high temps (last 5 years) — used for soil temperature reasoning
- Every plant's `outdoor_sow_date_range` from the plant database

**Agent 2 — Date Researcher (`date_researcher_agent`)**
Receives the DB Reader's summary and interprets each plant's range string using
domain rules:

| Range type | Reasoning |
|---|---|
| `"After last frost"` | last spring frost + 1 week |
| `"X–Y weeks before/after last frost"` | frost date ± average of X and Y weeks |
| `"As soon as soil is workable"` | frost date − 4 weeks (cool-season crop) |
| `"Transplant when soil is 70°F"` | find first month where avg high > 70°F |
| `"Transplant when soil is 60°F"` | find first month where avg high > 60°F |
| `"May 1 – May 15"` (calendar range) | midpoint of the two dates |
| `"Early spring"` | frost − 3 weeks |
| `"Late spring"` | frost + 2 weeks |
| `"4–6 weeks before first frost in fall"` | fall frost − average weeks |
| `"n/a"` | null — no estimate possible |

The researcher returns a structured Pydantic `SowEstimates` object with one
`PlantEstimate` per plant, including a one-sentence reasoning for each date.

### Tool 5 — `build_sow_calendar`
Writes to `sow_calendar` in `weather.duckdb`, replacing all rows for the zipcode.
Each plant gets two date columns:

| Column | Source |
|---|---|
| `outdoor_sow_date` | Researched (high-confidence) or frost + 2 weeks fallback |
| `estimated_outdoor_sow_date` | Two-agent DB range analysis (null if n/a) |

`sow_source` column records `"researched"` or `"fallback"` for `outdoor_sow_date`.

### Output Guardrail — `evaluate_sow_calendar`
After `frost_agent` produces its final summary text, a `sow_date_checker` agent
reviews it and confirms:
- The last spring frost date is plausible (not in summer).
- The plant count is non-zero.
- Any specific dates mentioned fall after the frost date.

If the check fails (`looks_reasonable=False`), the guardrail blocks the output and
the concern is printed instead of crashing.

---

## On Disagreements — Turns Out Gardening is Hard

When you ask two different agents the same question — one trained on regional horticultural
knowledge, one reading actual historical temperature data — they don't always agree. For
example, the LLM researcher might say "plant tomatoes May 25" based on Seattle's maritime
climate, while the DB estimator says "June 1" because that's when avg highs first clear 60°F
in the historical record. Both are defensible. Neither is obviously wrong.

Rather than picking a winner arbitrarily or just showing both and shrugging, we brought in a
third agent: `final_judge_agent`. It reviews each disagreement with full context — both dates,
the source of each, and the plant's general sow range from the database — and reasons through
which is more trustworthy. Researched dates win when the LLM had high regional confidence.
Estimated dates win when the LLM fell back to a generic frost offset and the DB range gives
better grounding. The judge writes a `recommended_date` and `recommended_source` back to the
calendar so you always have a final answer.

See `judge decisions` in `database_explorer.ipynb` to review what the judge picked and why.

---

## Cost

| Step | Model | Approx cost |
|---|---|---|
| `research_sow_dates` | gpt-4o-mini | ~$0.010 |
| `estimate_sow_dates` (2 agents, batched) | gpt-4o-mini | ~$0.020 |
| `judge_sow_dates` (final judge) | gpt-4o-mini | ~$0.010 |
| `evaluate_sow_calendar` (guardrail) | gpt-4o-mini | ~$0.005 |
| **Total** | | **~$0.05** |

Weather fetch is free (Open-Meteo, no key). Plant DB was pre-built by the CrewAI
`plant_db_team` pipeline (~$3.60 for 74 plants with gpt-4o). You only need to run
that once — the plant database is already in the repo.
