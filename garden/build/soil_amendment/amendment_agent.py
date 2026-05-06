"""Soil Amendment Advisor — AutoGen AgentChat.

Analyzes a plot's harvest history and plant soil needs, then recommends
soil amendments for the next planting season.

Importable by the Streamlit app or run standalone via main.py.

ASSUMPTIONS:
  Method: AutoGen AssistantAgent with reflect_on_tool_use=True
  Ref: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
  Requires:
    - OPENAI_API_KEY in environment
    - plant.duckdb and garden.duckdb at garden/data/
    - plant_rag index built (garden/data/plant_rag/) for get_amendment_knowledge
  Violated by: missing harvest data (returns sparse advice), missing RAG index (skips knowledge retrieval)
  Validated: no
"""

import json
from pathlib import Path
from typing import Any

import duckdb
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

try:
    import app.rag.plant_retriever as plant_retriever
    _RAG_AVAILABLE = True
except ImportError:
    plant_retriever = None  # type: ignore[assignment]
    _RAG_AVAILABLE = False


# ── Parameters ────────────────────────────────────────────────────────────────

_GARDEN_ROOT    = Path(__file__).parents[2]
_DATA_DIR       = _GARDEN_ROOT / "data"
_GARDEN_DB_PATH = str(_DATA_DIR / "garden.duckdb")
_PLANT_DB_PATH  = str(_DATA_DIR / "plant.duckdb")
_ENV_PATH       = _GARDEN_ROOT.parents[2] / ".env"

MODEL = "gpt-4o-mini"

SYSTEM_MESSAGE = """\
You are a soil amendment advisor for a home garden.

Your job:
1. Use get_plot_harvest_history to see what was grown and harvested in the plot.
2. Use get_plant_soil_needs for each plant type returned to check NPK requirements and notes.
3. Use get_amendment_knowledge to look up best practices for soil amendments.
4. Synthesize the findings into a practical, actionable amendment plan.

Your advice should:
- Identify nutrients likely depleted based on what was grown.
- Suggest specific organic amendments (compost, bone meal, kelp meal, etc.) with approximate quantities per 100 sq ft where possible.
- Note any pH adjustments needed.
- Mention timing (pre-season vs post-harvest).
- Be grounded in the actual harvest data — if there was little or no harvesting logged, say so.
- Be friendly and practical for a home gardener.
"""

load_dotenv(_ENV_PATH)


# ── Tool factory ──────────────────────────────────────────────────────────────

def _make_tools(
    garden_conn: duckdb.DuckDBPyConnection,
    plant_conn: duckdb.DuckDBPyConnection,
) -> list:
    """
    Return tool functions bound to the provided database connections.

    Accepts existing open connections to avoid DuckDB file-lock conflicts
    when called from a Streamlit app that already holds the connections.

    Args:
        garden_conn: Open DuckDB connection to garden.duckdb.
        plant_conn:  Open DuckDB connection to plant.duckdb.

    Returns:
        List of plain callables suitable for use as AutoGen tools.
    """

    def get_plot_harvest_history(plot_id: str) -> str:
        """
        Return a summary of what was grown and harvested in a plot, grouped by plant type.

        Args:
            plot_id: The plot UUID to look up.

        Returns:
            JSON-formatted string with plant names, instance count, total harvest
            sessions, total quantity, and total weight per plant type.
        """
        try:
            rows = garden_conn.execute(
                """
                SELECT
                    pi.plant_type_id,
                    COUNT(DISTINCT pi.instance_id)                              AS plant_count,
                    COUNT(DISTINCT CASE WHEN pi.status IN ('fully_harvested', 'harvested')
                                        THEN pi.instance_id END)                AS harvested_count,
                    COALESCE(SUM(hl.quantity), 0)                               AS total_qty,
                    COALESCE(SUM(hl.weight_lbs), 0.0)                           AS total_weight_lbs,
                    COUNT(DISTINCT hl.harvest_id)                               AS harvest_log_entries,
                    MAX(hl.harvest_date)                                         AS last_harvest_date
                FROM plant_instances pi
                LEFT JOIN harvest_log hl ON hl.instance_id = pi.instance_id
                WHERE pi.plot_id = ?
                  AND pi.status IN ('fully_harvested', 'harvested')
                GROUP BY pi.plant_type_id
                ORDER BY harvested_count DESC, total_weight_lbs DESC
                """,
                [plot_id],
            ).fetchall()

            if not rows:
                return json.dumps({"error": "No harvested plants found for this plot. Mark plants as 'fully_harvested' or 'harvested' to get amendment advice."})

            result: list[dict[str, Any]] = []
            for plant_type_id, count, harvested, qty, weight, log_entries, last_h in rows:
                name_row = plant_conn.execute(
                    "SELECT name FROM plant_types WHERE plant_type_id = ?",
                    [plant_type_id],
                ).fetchone()
                name = name_row[0] if name_row else plant_type_id
                result.append({
                    "plant": name,
                    "plant_type_id": plant_type_id,
                    "instances": count,
                    "instances_marked_harvested": harvested,
                    "harvest_log_entries": log_entries,
                    "total_quantity_logged": int(qty),
                    "total_weight_lbs_logged": round(float(weight), 2),
                    "last_harvest_date": str(last_h) if last_h else None,
                    "note": (
                        "Marked as harvested in the app but no weight/quantity was logged."
                        if harvested > 0 and log_entries == 0
                        else None
                    ),
                })
            return json.dumps(result, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def get_plant_soil_needs(plant_type_id: str) -> str:
        """
        Return soil NPK requirements and growing notes for a plant type.

        Args:
            plant_type_id: The plant type UUID to look up.

        Returns:
            JSON-formatted string with NPK values, growth needs, and post-harvest
            soil notes from plant_varieties.
        """
        try:
            name_row = plant_conn.execute(
                "SELECT name FROM plant_types WHERE plant_type_id = ?",
                [plant_type_id],
            ).fetchone()
            name = name_row[0] if name_row else plant_type_id

            variety = plant_conn.execute(
                """SELECT soil_n, soil_p, soil_k, growth_needs, post_harvest_soil_needs
                   FROM plant_varieties WHERE plant_type_id = ? LIMIT 1""",
                [plant_type_id],
            ).fetchone()

            if not variety:
                return json.dumps({"plant": name, "note": "No soil data found."})

            soil_n, soil_p, soil_k, growth_needs, post_harvest = variety
            return json.dumps({
                "plant": name,
                "soil_N": soil_n,
                "soil_P": soil_p,
                "soil_K": soil_k,
                "growth_needs": growth_needs,
                "post_harvest_soil_needs": post_harvest,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def get_amendment_knowledge(query: str) -> str:
        """
        Search the plant knowledge base for soil amendment guidance.

        Args:
            query: A natural-language question about soil amendments, composting,
                   or soil health.

        Returns:
            Relevant text passages from the plant knowledge index.
        """
        if not _RAG_AVAILABLE or plant_retriever is None:
            return "Plant knowledge module not available — check sys.path includes garden/."
        try:
            if not plant_retriever.is_ready():
                return "Plant knowledge index not built yet. Run build/plant_rag/main.py first."
            results = plant_retriever.search(query, k=3)
            return "\n\n---\n\n".join(results)
        except Exception as exc:
            return f"Knowledge search failed: {exc}"

    return [get_plot_harvest_history, get_plant_soil_needs, get_amendment_knowledge]


# ── Agent runner ──────────────────────────────────────────────────────────────

async def run(
    plot_id: str,
    plot_name: str,
    garden_conn: duckdb.DuckDBPyConnection | None = None,
    plant_conn: duckdb.DuckDBPyConnection | None = None,
) -> str:
    """
    Run the soil amendment advisor for a plot and return the advice text.

    Accepts optional existing database connections to avoid DuckDB file-lock
    conflicts when called from a Streamlit app. Opens fresh connections when
    called standalone (garden_conn and plant_conn are None).

    Args:
        plot_id:     UUID of the plot to analyze.
        plot_name:   Human-readable name used in the task prompt.
        garden_conn: Existing garden.duckdb connection, or None to open one.
        plant_conn:  Existing plant.duckdb connection, or None to open one.

    Returns:
        Markdown-formatted amendment recommendation from the agent.
    """
    own_garden = garden_conn is None
    own_plant  = plant_conn is None

    if own_garden:
        garden_conn = duckdb.connect(_GARDEN_DB_PATH, read_only=True)
    if own_plant:
        plant_conn = duckdb.connect(_PLANT_DB_PATH, read_only=True)

    tools = _make_tools(garden_conn, plant_conn)

    model_client = OpenAIChatCompletionClient(model=MODEL)
    agent = AssistantAgent(
        name="soil_amendment_advisor",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=tools,
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    task = (
        f"Analyze the soil amendment needs for plot '{plot_name}' (plot_id: {plot_id}). "
        "Review what was grown and harvested there, check the soil requirements for those plants, "
        "look up relevant amendment best practices, and provide a specific amendment plan."
    )

    response = await agent.on_messages(
        [TextMessage(content=task, source="user")],
        cancellation_token=CancellationToken(),
    )

    await model_client.close()

    if own_garden:
        garden_conn.close()
    if own_plant:
        plant_conn.close()

    return response.chat_message.content
