"""Garden Advisor — LangGraph worker / evaluator / tools graph.

Embedded in the Plants tab as a collapsible section. Analyses plot performance
(what's thriving, what's struggling) then recommends what to plant in open space,
weighted toward the grower's actual history and stated preferences.
"""

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.db.connections import DatabaseConnections
import app.rag.plant_retriever as plant_retriever


# ── Parameters ────────────────────────────────────────────────────────────────

WORKER_MODEL    = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4o-mini"

SUCCESS_CRITERIA = (
    "Lead with a clear performance summary: which plants are thriving (harvests logged), "
    "which are on track, and which are overdue with no harvests — name them specifically. "
    "Then recommend what to plant in the open space, biased toward plants the grower has "
    "grown before and their stated food/interest preferences. "
    "Every open-space pick must be justified by companion rules, frost window, and sun zone. "
    "Be specific and personal — not a generic gardening guide."
)

_ACTIVE_STATUSES   = ("active", "planned")
_OCCUPIED_STATUSES = ("active", "planned", "fully_harvested")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_advice_table(db: DatabaseConnections) -> None:
    db.garden.execute("""
        CREATE TABLE IF NOT EXISTS layout_advice (
            advice_id    VARCHAR PRIMARY KEY,
            plot_id      VARCHAR NOT NULL,
            generated_at TIMESTAMP NOT NULL,
            advice_text  VARCHAR NOT NULL
        )
    """)


def _load_latest_advice(db: DatabaseConnections, plot_id: str) -> dict | None:
    row = db.garden.execute(
        """SELECT advice_id, generated_at, advice_text
           FROM layout_advice WHERE plot_id = ?
           ORDER BY generated_at DESC LIMIT 1""",
        [plot_id],
    ).fetchone()
    if not row:
        return None
    return {"advice_id": row[0], "generated_at": row[1], "advice_text": row[2]}


def _save_advice(db: DatabaseConnections, plot_id: str, advice_text: str) -> None:
    db.garden.execute(
        """INSERT INTO layout_advice (advice_id, plot_id, generated_at, advice_text)
           VALUES (?, ?, ?, ?)""",
        [str(uuid.uuid4()), plot_id, datetime.now(timezone.utc), advice_text],
    )
    db.garden.commit()


# ── Structured evaluator output ───────────────────────────────────────────────

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the worker's response")
    success_criteria_met: bool
    user_input_needed: bool


# ── Graph state ───────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


# ── Tool factory ──────────────────────────────────────────────────────────────

def make_tools(db: DatabaseConnections) -> list[Any]:
    """Build LangChain tools as closures that capture the live DB connections."""

    @tool
    def get_plant_performance(plot_id: str) -> str:
        """
        Analyse how each plant type in the plot is actually performing.

        For every active/planned plant returns: days in ground, expected harvest
        window, harvests logged, weight harvested, and a performance tag:
          ✅ PRODUCTIVE   — harvests logged
          🔔 HARVEST WINDOW — within harvest window, check now
          ⚠️ OVERDUE      — past days_to_harvest with no harvests logged
          ⏳ ON TRACK     — still growing, not yet due
          🆕 JUST PLANTED — less than 14 days in ground
        """
        instances = db.garden.execute("""
            SELECT pi.plant_type_id,
                   pi.status,
                   MIN(COALESCE(pi.planted_date, pi.planned_sow_date)) AS earliest_date,
                   COUNT(pi.instance_id) AS instance_count,
                   SUM(COALESCE(pi.harvest_count, 0)) AS total_harvests,
                   SUM(COALESCE(pi.harvest_weight_lbs, 0.0)) AS total_weight
            FROM plant_instances pi
            WHERE pi.plot_id = ?
              AND pi.status IN ('active', 'planned')
            GROUP BY pi.plant_type_id, pi.status
            ORDER BY pi.plant_type_id
        """, [plot_id]).fetchall()

        if not instances:
            return "No active or planned plants in this plot."

        # Harvest log totals per plant type
        harvest_log = db.garden.execute("""
            SELECT pi.plant_type_id,
                   COUNT(hl.harvest_id) AS log_entries,
                   COALESCE(SUM(hl.weight_lbs), 0.0) AS log_weight
            FROM harvest_log hl
            JOIN plant_instances pi ON pi.instance_id = hl.instance_id
            WHERE pi.plot_id = ?
            GROUP BY pi.plant_type_id
        """, [plot_id]).fetchall()
        log_map: dict[str, tuple[int, float]] = {r[0]: (r[1], r[2]) for r in harvest_log}

        type_ids = list({r[0] for r in instances if r[0]})
        name_map: dict[str, str] = {}
        dth_map: dict[str, float] = {}
        if type_ids:
            ph = ",".join("?" * len(type_ids))
            name_rows = db.plant.execute(
                f"SELECT plant_type_id, name FROM plant_types WHERE plant_type_id IN ({ph})",
                type_ids,
            ).fetchall()
            name_map = {r[0]: r[1] for r in name_rows}

            dth_rows = db.plant.execute(
                f"SELECT plant_type_id, AVG(days_to_harvest) FROM plant_varieties "
                f"WHERE plant_type_id IN ({ph}) GROUP BY plant_type_id",
                type_ids,
            ).fetchall()
            dth_map = {r[0]: float(r[1]) for r in dth_rows if r[1]}

        today = date.today()
        lines = ["PLANT PERFORMANCE:"]

        for type_id, status, earliest_date, count, hc, hw in instances:
            name = name_map.get(type_id, type_id or "Unknown")
            log_entries, log_weight = log_map.get(type_id, (0, 0.0))
            total_harvests = (hc or 0) + log_entries
            total_weight   = (hw or 0.0) + log_weight

            if earliest_date:
                try:
                    ed = earliest_date if isinstance(earliest_date, date) else date.fromisoformat(str(earliest_date))
                    days_in = (today - ed).days
                except Exception:
                    days_in = None
            else:
                days_in = None

            dth = dth_map.get(type_id)

            if total_harvests > 0:
                tag = "✅ PRODUCTIVE"
            elif days_in is None:
                tag = "❓ NO DATE RECORDED"
            elif days_in < 14:
                tag = "🆕 JUST PLANTED"
            elif dth and days_in >= int(dth * 1.1):
                tag = "⚠️ OVERDUE — no harvests logged"
            elif dth and days_in >= int(dth * 0.8):
                tag = "🔔 HARVEST WINDOW — check now"
            else:
                tag = "⏳ ON TRACK"

            timing = ""
            if days_in is not None:
                timing = f"{days_in}d in ground"
                if dth:
                    timing += f", avg harvest at {int(dth)}d"

            harvest_info = ""
            if total_harvests > 0:
                harvest_info = f", {total_harvests} harvest(s)"
                if total_weight > 0:
                    harvest_info += f" / {total_weight:.1f} lbs"

            lines.append(
                f"  {name} ({count} plants, {status}) — {tag}"
                + (f" — {timing}" if timing else "")
                + harvest_info
            )

        return "\n".join(lines)

    @tool
    def get_grower_history() -> str:
        """
        Summarise what this grower has actually planted and harvested across
        the entire garden (all plots, all time).

        Returns plant types ranked by: times planted, total harvests, total weight.
        Use this to bias open-space recommendations toward plants the grower
        has grown before and succeeded with.
        """
        rows = db.garden.execute("""
            SELECT pi.plant_type_id,
                   COUNT(pi.instance_id)                        AS times_planted,
                   SUM(COALESCE(pi.harvest_count, 0))           AS inst_harvests,
                   SUM(COALESCE(pi.harvest_weight_lbs, 0.0))    AS inst_weight
            FROM plant_instances pi
            GROUP BY pi.plant_type_id
            ORDER BY times_planted DESC
        """).fetchall()

        if not rows:
            return "No planting history found."

        log_rows = db.garden.execute("""
            SELECT pi.plant_type_id,
                   COUNT(hl.harvest_id)             AS log_entries,
                   COALESCE(SUM(hl.weight_lbs), 0.0) AS log_weight
            FROM harvest_log hl
            JOIN plant_instances pi ON pi.instance_id = hl.instance_id
            GROUP BY pi.plant_type_id
        """).fetchall()
        log_map: dict[str, tuple[int, float]] = {r[0]: (r[1], r[2]) for r in log_rows}

        type_ids = [r[0] for r in rows if r[0]]
        name_map: dict[str, str] = {}
        if type_ids:
            ph = ",".join("?" * len(type_ids))
            name_rows = db.plant.execute(
                f"SELECT plant_type_id, name FROM plant_types WHERE plant_type_id IN ({ph})",
                type_ids,
            ).fetchall()
            name_map = {r[0]: r[1] for r in name_rows}

        lines = ["GROWER HISTORY (all time, all plots):"]
        for type_id, times_planted, inst_h, inst_w in rows:
            name = name_map.get(type_id, type_id or "Unknown")
            log_h, log_w = log_map.get(type_id, (0, 0.0))
            total_h = (inst_h or 0) + log_h
            total_w = (inst_w or 0.0) + log_w
            harvest_note = f", {total_h} harvests / {total_w:.1f} lbs" if total_h > 0 else ""
            lines.append(f"  {name}: planted {times_planted}x{harvest_note}")

        return "\n".join(lines)

    @tool
    def estimate_open_space(plot_id: str) -> str:
        """
        Estimate available open space in the plot in square feet.

        Cleared (fully_harvested) plants are treated as freed space.
        Falls back to an estimate if plot area is not configured.
        """
        area_row = db.garden.execute(
            "SELECT area_sqft FROM plots WHERE plot_id = ?", [plot_id]
        ).fetchone()
        total_area = float(area_row[0]) if area_row and area_row[0] else None

        ph = ",".join("?" * len(_ACTIVE_STATUSES))
        active_instances = db.garden.execute(
            f"""SELECT COALESCE(spacing_inches, 6.0)
                FROM plant_instances
                WHERE plot_id = ? AND status IN ({ph})""",
            [plot_id, *_ACTIVE_STATUSES],
        ).fetchall()
        active_sqft = sum((r[0] / 12.0) ** 2 for r in active_instances)

        if total_area:
            open_sqft = max(0.0, total_area - active_sqft)
            return (
                f"Plot total: {total_area:.0f} sqft. "
                f"Used by active plants: ~{active_sqft:.0f} sqft. "
                f"Available open space: ~{open_sqft:.0f} sqft."
            )
        return (
            f"Plot area not configured (set scale on Map tab). "
            f"Active plants occupy ~{active_sqft:.0f} sqft. "
            f"Assume at least 50 sqft free and proceed."
        )

    @tool
    def get_plot_zones(plot_id: str) -> str:
        """Return the sun and height zone breakdown for the plot."""
        try:
            db.garden.execute("""
                CREATE TABLE IF NOT EXISTS plot_regions (
                    region_id   VARCHAR PRIMARY KEY,
                    plot_id     VARCHAR NOT NULL,
                    region_type VARCHAR NOT NULL,
                    zone_value  VARCHAR NOT NULL,
                    polygon     JSON NOT NULL,
                    area_sqft   DOUBLE
                )
            """)
            rows = db.garden.execute(
                """SELECT region_type, zone_value, COALESCE(area_sqft, 0)
                   FROM plot_regions WHERE plot_id = ?
                   ORDER BY region_type, area_sqft DESC""",
                [plot_id],
            ).fetchall()
        except Exception as exc:
            return f"Zone data unavailable ({exc}). Assume full sun, no height restrictions."

        if not rows:
            return (
                "No zones defined. "
                "Assume full sun and no height restrictions — all types are candidates."
            )

        sun_zones:    list[str] = []
        height_zones: list[str] = []
        for region_type, zone_value, area in rows:
            entry = f"{zone_value} ({area:.0f} sqft)"
            if region_type == "sun":
                sun_zones.append(entry)
            elif region_type == "height":
                height_zones.append(entry)

        return "\n".join([
            f"Sun zones: {', '.join(sun_zones) if sun_zones else 'not defined (assume full_sun)'}",
            f"Height zones: {', '.join(height_zones) if height_zones else 'not defined'}",
        ])

    @tool
    def get_frost_context() -> str:
        """
        Return frost timing: days since last spring frost, days until fall frost,
        and planting window guidance. Falls back to seasonal heuristics.
        """
        today = date.today()
        lines = [f"Today: {today}"]

        try:
            zipcode_row = db.garden.execute("SELECT zipcode FROM garden LIMIT 1").fetchone()
            zipcode = zipcode_row[0] if zipcode_row else None
            frost = None
            if zipcode:
                frost = db.weather.execute(
                    """SELECT avg_last_spring_frost, avg_first_fall_frost, frost_free_days
                       FROM frost_dates WHERE zipcode = ?""",
                    [zipcode],
                ).fetchone()
        except Exception:
            frost = None

        if not frost:
            month = today.month
            if 3 <= month <= 5:
                lines += ["Season: spring — frost-safe after mid-April.", "days_until_fall_frost: 200"]
            elif 6 <= month <= 8:
                lines += ["Season: summer — full growing season.", "days_until_fall_frost: 130"]
            elif 9 <= month <= 10:
                lines += ["Season: fall — fast-maturing/cold-hardy only (<60d).", "days_until_fall_frost: 45"]
            else:
                lines += ["Season: winter — plan for spring.", "days_until_fall_frost: 0"]
            return "\n".join(lines)

        if frost[0]:
            sd = frost[0] if isinstance(frost[0], date) else date.fromisoformat(str(frost[0]))
            sd = sd.replace(year=today.year)
            delta = (today - sd).days
            lines.append(
                f"Last spring frost: {sd} ({delta}d ago — frost-safe)"
                if delta >= 0 else
                f"Last spring frost: {sd} ({abs(delta)}d away — still possible)"
            )

        if frost[1]:
            fd = frost[1] if isinstance(frost[1], date) else date.fromisoformat(str(frost[1]))
            fd = fd.replace(year=today.year)
            delta_fall = (fd - today).days
            if delta_fall > 0:
                lines.append(
                    f"First fall frost: {fd} ({delta_fall}d away). "
                    f"Plants needing >{delta_fall}d may not finish."
                )
                lines.append(f"days_until_fall_frost: {delta_fall}")
            else:
                lines.append("First fall frost already past — cold-hardy crops only.")
                lines.append("days_until_fall_frost: 0")

        if frost[2]:
            lines.append(f"Frost-free days/year: {frost[2]}")

        return "\n".join(lines)

    @tool
    def get_companion_analysis(plot_id: str) -> str:
        """
        Analyse companion and antagonist relationships among plants in the plot.

        Returns existing conflicts, positive pairings, and the JSON name list
        needed for find_open_space_candidates.
        """
        type_rows = db.garden.execute(
            "SELECT DISTINCT plant_type_id FROM plant_instances WHERE plot_id = ?",
            [plot_id],
        ).fetchall()
        type_ids = [r[0] for r in type_rows]

        if not type_ids:
            return "No plants found in this plot."

        ph = ",".join("?" * len(type_ids))
        name_rows = db.plant.execute(
            f"SELECT plant_type_id, name FROM plant_types WHERE plant_type_id IN ({ph})",
            type_ids,
        ).fetchall()
        name_map      = {r[0]: r[1] for r in name_rows}
        current_names = set(name_map.values())

        companion_rows = db.plant.execute(
            f"""SELECT pt.name, pc.companion_name, pc.relationship
                FROM plant_companions pc
                JOIN plant_types pt ON pt.plant_type_id = pc.plant_type_id
                WHERE pc.plant_type_id IN ({ph})""",
            type_ids,
        ).fetchall()

        conflicts: list[str] = []
        positive:  list[str] = []
        for plant_name, companion_name, rel in companion_rows:
            if companion_name in current_names:
                if rel == "antagonist":
                    conflicts.append(f"⚠️ {plant_name} ↔ {companion_name} (antagonists)")
                else:
                    positive.append(f"✅ {plant_name} ↔ {companion_name} (companions)")

        lines = [f"Current plants: {', '.join(sorted(current_names))}", ""]
        if conflicts:
            lines += ["EXISTING CONFLICTS:"] + conflicts + [""]
        if positive:
            lines += ["POSITIVE PAIRINGS:"] + positive + [""]
        if not conflicts and not positive:
            lines.append("No companion/antagonist relationships among current plants.")
        lines.append(f"Pass to find_open_space_candidates: {json.dumps(sorted(current_names))}")
        return "\n".join(lines)

    @tool
    def find_open_space_candidates(
        current_plant_names_json: str,
        preferred_plant_names_json: str,
        available_sqft: float,
        dominant_sun_zone: str,
        days_until_fall_frost: int,
    ) -> str:
        """
        Find plants that fit the open space, ranked by personal history + companion fit.

        Plants the grower has grown before are ranked higher. Antagonists excluded.
        Frost-risky plants are flagged but not hidden.

        Args:
            current_plant_names_json:   JSON array of plants currently in the plot.
            preferred_plant_names_json: JSON array of plants the grower has grown before
                                        (from get_grower_history) — these get a ranking boost.
            available_sqft:             Open space in square feet.
            dominant_sun_zone:          e.g. 'full_sun', 'partial_shade', 'full_shade'.
            days_until_fall_frost:      Days until first fall frost (0 = past frost).
        """
        try:
            current_names: list[str]   = json.loads(current_plant_names_json)
            preferred_names: list[str] = json.loads(preferred_plant_names_json)
        except Exception:
            return "Invalid JSON — pass JSON arrays of strings."

        ph_cur = ",".join("?" * len(current_names)) if current_names else "''"

        antagonists: set[str] = set()
        companions:  set[str] = set()
        if current_names:
            ant_rows = db.plant.execute(
                f"""SELECT DISTINCT pc.companion_name FROM plant_companions pc
                    JOIN plant_types pt ON pt.plant_type_id = pc.plant_type_id
                    WHERE pt.name IN ({ph_cur}) AND pc.relationship = 'antagonist'""",
                current_names,
            ).fetchall()
            antagonists = {r[0] for r in ant_rows}

            comp_rows = db.plant.execute(
                f"""SELECT DISTINCT pc.companion_name FROM plant_companions pc
                    JOIN plant_types pt ON pt.plant_type_id = pc.plant_type_id
                    WHERE pt.name IN ({ph_cur}) AND pc.relationship = 'companion'""",
                current_names,
            ).fetchall()
            companions = {r[0] for r in comp_rows}

        all_plants = db.plant.execute(
            """SELECT pt.name, pt.category,
                      pv.spacing_inches, pv.sun_tolerance,
                      pv.days_to_harvest, pv.height_inches_estimate
               FROM plant_types pt
               LEFT JOIN plant_varieties pv ON pv.plant_type_id = pt.plant_type_id
               ORDER BY pt.name"""
        ).fetchall()

        preferred_set = set(preferred_names)
        results: list[dict] = []
        seen:    set[str]   = set()

        effective_sqft = available_sqft if available_sqft > 0 else 80.0

        for name, category, spacing, sun, days, height in all_plants:
            if name in current_names or name in seen or name in antagonists:
                continue
            seen.add(name)

            spacing_val = float(spacing) if spacing else 6.0
            footprint   = (spacing_val / 12.0) ** 2
            max_plants  = int(effective_sqft / footprint) if footprint > 0 else 1
            if max_plants < 1:
                continue

            days_val  = int(days) if days else None
            frost_ok  = days_val is None or days_until_fall_frost <= 0 or days_val <= days_until_fall_frost
            frost_note = f" ⚠️ needs {days_val}d, only {days_until_fall_frost}d left" if not frost_ok else ""

            sun_ok   = (sun is None or dominant_sun_zone == "full_sun"
                        or sun == dominant_sun_zone or sun == "partial_shade")
            sun_note = f" (prefers {sun})" if not sun_ok else ""

            # Score: grower history (+3) > companion (+2) > frost fit (+1) > sun fit (+1)
            score = (
                (3 if name in preferred_set else 0)
                + (2 if name in companions else 0)
                + (1 if frost_ok else 0)
                + (1 if sun_ok else 0)
            )

            results.append({
                "name": name, "category": category, "score": score,
                "spacing_in": spacing_val, "max_plants": max_plants,
                "sun_tolerance": sun, "days_to_harvest": days_val,
                "is_companion": name in companions,
                "is_familiar": name in preferred_set,
                "frost_note": frost_note, "sun_note": sun_note,
            })

        results.sort(key=lambda x: -x["score"])

        if not results:
            return "No compatible plants found for the available space."

        lines = [f"Top candidates for ~{effective_sqft:.0f} sqft:", ""]
        for r in results[:12]:
            tags = []
            if r["is_familiar"]:
                tags.append("🔁 GROWN BEFORE")
            if r["is_companion"]:
                tags.append("🌟 COMPANION")
            tag_str = " ".join(tags) if tags else "○ neutral"
            lines.append(
                f"- {r['name']} ({r['category']}) [{tag_str}] — "
                f"spacing {r['spacing_in']:.0f}in, fits ~{r['max_plants']} plants, "
                f"harvest: {r['days_to_harvest'] or '?'}d"
                f"{r['frost_note']}{r['sun_note']}"
            )
        return "\n".join(lines)

    @tool
    def search_plant_knowledge(query: str) -> str:
        """Search the plant knowledge base for care, companions, pests, and growing tips."""
        if not plant_retriever.is_ready():
            return "Plant knowledge index not built yet."
        try:
            results = plant_retriever.search(query, k=3)
            return "\n\n---\n\n".join(results)
        except Exception as exc:
            return f"Knowledge search failed: {exc}"

    return [
        get_plant_performance,
        get_grower_history,
        estimate_open_space,
        get_plot_zones,
        get_frost_context,
        get_companion_analysis,
        find_open_space_candidates,
        search_plant_knowledge,
    ]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    db: DatabaseConnections,
    food_prefs: str = "",
    interests: str = "",
) -> Any:
    """Compile the garden advisor LangGraph."""
    tools = make_tools(db)

    worker_llm            = ChatOpenAI(model=WORKER_MODEL, temperature=0)
    worker_llm_with_tools = worker_llm.bind_tools(tools)

    evaluator_llm         = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    evaluator_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

    def worker(state: State) -> Dict[str, Any]:
        pref_ctx = f"\nThe grower enjoys eating: {food_prefs}." if food_prefs else ""
        int_ctx  = f"\nThe grower is interested in: {interests}." if interests else ""

        system_message = (
            f"You are a personal garden advisor who knows this specific grower's history. "
            f"Today is {date.today()}.{pref_ctx}{int_ctx}\n\n"
            "WORKFLOW — follow in this exact order:\n"
            "1. get_plant_performance(plot_id) — tag each plant: thriving, on track, overdue\n"
            "2. get_grower_history() — what they have grown and harvested across all time\n"
            "3. estimate_open_space(plot_id) — how much space is free\n"
            "4. get_plot_zones(plot_id) — sun and height zones\n"
            "5. get_frost_context() — days until fall frost\n"
            "6. get_companion_analysis(plot_id) — conflicts, pairings, current name list\n"
            "7. find_open_space_candidates(current_plant_names_json, preferred_plant_names_json, "
            "available_sqft, dominant_sun_zone, days_until_fall_frost) — pass the grower history "
            "plant names as preferred_plant_names_json\n"
            "8. Optionally search_plant_knowledge for specific tips on flagged or recommended plants\n\n"
            "OUTPUT FORMAT:\n"
            "**What's happening in the plot:**\n"
            "- Name each productive plant and say what it's produced\n"
            "- Name each struggling/overdue plant specifically — don't be vague\n"
            "- Flag any companion conflicts\n\n"
            "**What to plant in the open space:**\n"
            "- Lead with plants the grower has grown before (🔁) and companions (🌟)\n"
            "- Factor in their food preferences and gardening interests\n"
            "- Give estimated plant counts\n"
            "- Flag frost-risky picks\n\n"
            "RULES:\n"
            "- Never suggest removing or changing existing plants\n"
            "- Be specific — name the actual plants, actual counts, actual days\n"
            "- Do NOT give generic gardening advice — speak to this person's actual situation\n\n"
            f"Success criteria:\n{state['success_criteria']}"
        )

        if state.get("feedback_on_work"):
            system_message += (
                f"\n\nFeedback on previous response:\n{state['feedback_on_work']}"
                "\nAddress this directly."
            )

        messages = state["messages"]
        found = False
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg.content = system_message
                found = True
        if not found:
            messages = [SystemMessage(content=system_message)] + messages

        return {"messages": [worker_llm_with_tools.invoke(messages)]}

    def worker_router(state: State) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "evaluator"

    def evaluator(state: State) -> Dict[str, Any]:
        last_response = state["messages"][-1].content

        system_message = (
            "You are an evaluator for a personal garden advisor. "
            "Accept if the response: names specific plants with their performance status, "
            "calls out struggling plants by name, biases recommendations toward the grower's "
            "history and preferences, and gives specific plant counts. "
            "Reject only if it gives generic advice with no reference to actual plant performance."
        )
        user_message = (
            f"Success criteria: {state['success_criteria']}\n\n"
            f"Response:\n{last_response}\n\n"
            "Does it name specific plants and their performance? "
            "Does it reference the grower's history? Is it personal and specific?"
        )
        if state.get("feedback_on_work"):
            user_message += f"\n\nPrevious feedback: {state['feedback_on_work']}"

        result = evaluator_with_output.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ])

        return {
            "messages": [{"role": "assistant", "content": f"[Evaluator: {result.feedback}]"}],
            "feedback_on_work": result.feedback,
            "success_criteria_met": result.success_criteria_met,
            "user_input_needed": result.user_input_needed,
        }

    def route_based_on_evaluation(state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        return "worker"

    graph_builder = StateGraph(State)
    graph_builder.add_node("worker", worker)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_node("evaluator", evaluator)
    graph_builder.add_conditional_edges(
        "worker", worker_router, {"tools": "tools", "evaluator": "evaluator"}
    )
    graph_builder.add_edge("tools", "worker")
    graph_builder.add_conditional_edges(
        "evaluator", route_based_on_evaluation, {"worker": "worker", "END": END}
    )
    graph_builder.add_edge(START, "worker")

    return graph_builder.compile(checkpointer=MemorySaver())


# ── Section renderer (called from plants_page) ────────────────────────────────

def render_section(db: DatabaseConnections, plot_id: str, plot_name: str) -> None:
    """Render the Garden Advisor as a collapsible section in the Plants tab."""
    _ensure_advice_table(db)

    food_prefs = st.session_state.get("garden_bot_food_prefs", "")
    interests  = st.session_state.get("garden_bot_interests", "")

    st.divider()
    with st.expander("🌿 Garden Advisor", expanded=False):
        prior = _load_latest_advice(db, plot_id)
        if prior:
            st.caption(f"Last analysis: {prior['generated_at']}")
            st.markdown(prior["advice_text"])
            btn_label = "🔄 Re-analyze"
        else:
            st.caption("No analysis yet for this plot.")
            btn_label = "🌿 Analyze plot"

        if food_prefs or interests:
            st.caption(
                f"Using your preferences from the Assistant tab — "
                f"{'foods: ' + food_prefs if food_prefs else ''}"
                f"{' · ' if food_prefs and interests else ''}"
                f"{'interests: ' + interests if interests else ''}"
            )
        else:
            st.caption("Set food preferences in the Assistant tab to get personalised recommendations.")

        if st.button(btn_label, key=f"advisor_run_{plot_id}"):
            with st.spinner("Analysing your plot…"):
                try:
                    graph  = build_graph(db, food_prefs, interests)
                    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                    task = (
                        f"Analyse plot '{plot_name}' (plot_id: {plot_id}). "
                        "Check performance of every plant, identify what's thriving and "
                        "what's struggling, then recommend what to plant in the open space "
                        "based on my history and preferences."
                    )
                    result = graph.invoke(
                        {
                            "messages": [{"role": "user", "content": task}],
                            "success_criteria": SUCCESS_CRITERIA,
                            "feedback_on_work": None,
                            "success_criteria_met": False,
                            "user_input_needed": False,
                        },
                        config=config,
                    )
                    ai_reply = next(
                        (
                            m for m in reversed(result["messages"])
                            if isinstance(m, AIMessage)
                            and m.content
                            and not m.content.startswith("[Evaluator:")
                        ),
                        None,
                    )
                    advice = ai_reply.content if ai_reply else "No recommendation generated."
                    _save_advice(db, plot_id, advice)
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Advisor error: {exc}")
