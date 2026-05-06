"""General Garden Bot — LangGraph worker / evaluator / tools graph."""

import sqlite3
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
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
SOW_LOOKAHEAD_DAYS = 60

_GARDEN_ROOT    = Path(__file__).parents[2]
_MEMORY_DB_PATH = str(_GARDEN_ROOT / "data" / "assistant_memory.db")
_THREAD_ID_PATH = _GARDEN_ROOT / "data" / "assistant_thread_id.txt"

SUCCESS_CRITERIA = (
    "Provide a specific, actionable response to the user's question. "
    "Call the relevant tools to check garden state, weather, frost dates, sow calendar, and harvests. "
    "If a tool returns no data, acknowledge it briefly and move on — do not loop trying to get it. "
    "Give concrete, useful advice based on whatever data is available. "
    "Tailor suggestions to the user's food preferences and gardening interests."
)


# ── Persistent checkpointer ───────────────────────────────────────────────────

_conn         = sqlite3.connect(_MEMORY_DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


# ── Thread-ID helpers ─────────────────────────────────────────────────────────

def _load_thread_id() -> str:
    """Return the saved thread_id, or create and persist a new one."""
    if _THREAD_ID_PATH.exists():
        tid = _THREAD_ID_PATH.read_text().strip()
        if tid:
            return tid
    return _new_thread_id()


def _new_thread_id() -> str:
    """Generate a fresh thread_id and save it so the next page load continues it."""
    tid = str(uuid.uuid4())
    _THREAD_ID_PATH.write_text(tid)
    return tid


def _restore_messages(graph: Any, thread_id: str) -> list[dict]:
    """Rebuild the display message list from the persisted graph checkpoint."""
    try:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", [])
    except Exception:
        return []
    result = []
    for msg in msgs:
        if isinstance(msg, HumanMessage) and msg.content:
            result.append({"role": "user", "content": msg.content})
        elif (
            isinstance(msg, AIMessage)
            and msg.content
            and not msg.content.startswith("[Evaluator:")
        ):
            result.append({"role": "assistant", "content": msg.content})
    return result


# ── Structured evaluator output ───────────────────────────────────────────────

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the worker's response")
    success_criteria_met: bool = Field(
        description="Whether the response meets the success criteria"
    )
    user_input_needed: bool = Field(
        description="True if the worker needs clarification or is stuck"
    )


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
    def get_garden_overview() -> str:
        """Get the garden name, location, and a summary of all plots with plant counts."""
        garden_row = db.garden.execute(
            "SELECT name, zipcode FROM garden LIMIT 1"
        ).fetchone()

        plots = db.garden.execute(
            """SELECT pl.name, ROUND(pl.area_sqft, 0),
                      COUNT(pi.instance_id) AS plant_count
               FROM plots pl
               LEFT JOIN plant_instances pi ON pi.plot_id = pl.plot_id
               GROUP BY pl.plot_id, pl.name, pl.area_sqft
               ORDER BY pl.name"""
        ).fetchall()

        if not garden_row and not plots:
            return "No garden data found yet. The garden may not be saved in the database."

        parts: list[str] = []
        if garden_row:
            parts.append(f"Garden: {garden_row[0]} (zipcode {garden_row[1]})")
        else:
            parts.append("Garden metadata not set (no name/zipcode row), but plots exist.")

        if plots:
            plot_lines = [
                f"  - {p[0]}: {int(p[1] or 0)} sqft, {p[2]} plants" for p in plots
            ]
            total_plants = sum(p[2] for p in plots)
            parts.append(f"Plots ({len(plots)} total, {total_plants} plants):")
            parts.extend(plot_lines)
        else:
            parts.append("No plots found.")

        return "\n".join(parts)

    @tool
    def get_planted_plants() -> str:
        """Get all plant instances in the garden with their plot, status, and dates."""
        instances = db.garden.execute("""
            SELECT pi.plant_type_id, pl.name AS plot_name, pi.status,
                   pi.planted_date, pi.planned_sow_date, pi.harvest_count
            FROM plant_instances pi
            JOIN plots pl ON pl.plot_id = pi.plot_id
            ORDER BY pi.status, pl.name
        """).fetchall()
        if not instances:
            return "No plants in the garden yet."

        type_ids = list({r[0] for r in instances if r[0]})
        names_map: dict[str, str] = {}
        if type_ids:
            placeholders = ",".join("?" * len(type_ids))
            try:
                name_rows = db.plant.execute(
                    f"SELECT plant_type_id, name FROM plant_types "
                    f"WHERE plant_type_id IN ({placeholders})",
                    type_ids,
                ).fetchall()
                names_map = {r[0]: r[1] for r in name_rows}
            except Exception:
                pass

        lines = []
        for r in instances:
            plant_name = names_map.get(r[0], r[0] or "Unknown plant")
            planted = r[3] or r[4] or "no date"
            harvests = r[5] or 0
            lines.append(
                f"- {plant_name} in {r[1]}: {r[2]}, date={planted}, harvests={harvests}"
            )
        return "\n".join(lines)

    @tool
    def get_weather_conditions() -> str:
        """Get the current weather briefing and 7-day window (actual + forecast) for the garden."""
        row = db.garden.execute("SELECT zipcode FROM garden LIMIT 1").fetchone()
        if not row:
            return "No garden configured."
        zipcode = row[0]

        summary_row = db.weather.execute(
            "SELECT summary_text FROM weather_forecast_summary WHERE zipcode = ?",
            [zipcode],
        ).fetchone()
        summary = summary_row[0] if summary_row else None

        today = date.today()
        fc_rows = db.weather.execute(
            """SELECT date, record_type, temp_min_f, temp_max_f, precipitation_in, wind_max_mph
               FROM weather_forecast
               WHERE zipcode = ? AND date BETWEEN ? AND ?
               ORDER BY date""",
            [zipcode, str(today - timedelta(days=4)), str(today + timedelta(days=3))],
        ).fetchall()

        if not fc_rows and not summary:
            return "No weather data found. Click 'Get Forecast' on the Weather tab first."

        lines: list[str] = []
        if summary:
            lines.append(f"AI briefing: {summary}")
        for r in fc_rows:
            lines.append(
                f"{r[0]} ({r[1]}): high={r[3]}°F low={r[2]}°F "
                f"precip={r[4]}in wind={r[5]}mph"
            )
        return "\n".join(lines)

    @tool
    def get_frost_dates() -> str:
        """Get the average last spring frost, first fall frost, and frost-free days."""
        row = db.garden.execute("SELECT zipcode FROM garden LIMIT 1").fetchone()
        if not row:
            return "No garden configured."
        zipcode = row[0]

        frost = db.weather.execute(
            """SELECT avg_last_spring_frost, avg_first_fall_frost,
                      frost_free_days, years_analyzed
               FROM frost_dates WHERE zipcode = ?""",
            [zipcode],
        ).fetchone()
        if not frost:
            return "No frost data yet. Run 'build/weather_build/main.py' first."
        return (
            f"Avg last spring frost: {frost[0]}, avg first fall frost: {frost[1]}, "
            f"frost-free days: {frost[2]}, based on {frost[3]} years of records."
        )

    @tool
    def get_upcoming_sow_tasks() -> str:
        """Get plants with sow or transplant dates coming up in the next 60 days."""
        row = db.garden.execute("SELECT zipcode FROM garden LIMIT 1").fetchone()
        if not row:
            return "No garden configured."
        zipcode = row[0]

        today = date.today()
        cutoff = today + timedelta(days=SOW_LOOKAHEAD_DAYS)
        sow_rows = db.weather.execute(
            """SELECT plant_name, indoor_start_date, outdoor_sow_date,
                      recommended_date, notes
               FROM sow_calendar
               WHERE zipcode = ?
                 AND COALESCE(recommended_date, outdoor_sow_date) BETWEEN ? AND ?
               ORDER BY COALESCE(recommended_date, outdoor_sow_date)""",
            [zipcode, str(today), str(cutoff)],
        ).fetchall()
        if not sow_rows:
            return f"No sow tasks in the next {SOW_LOOKAHEAD_DAYS} days."

        lines = []
        for r in sow_rows:
            rec = r[3] or r[2] or "unknown date"
            note = f" — {r[4]}" if r[4] else ""
            lines.append(f"- {r[0]}: sow by {rec}{note}")
        return "\n".join(lines)

    @tool
    def get_harvest_history() -> str:
        """Get the 20 most recent harvest log entries across all plants."""
        rows = db.garden.execute("""
            SELECT hl.harvest_date, pi.plant_type_id, pl.name AS plot_name,
                   hl.quantity, hl.weight_lbs, hl.notes
            FROM harvest_log hl
            JOIN plant_instances pi ON pi.instance_id = hl.instance_id
            JOIN plots pl ON pl.plot_id = pi.plot_id
            ORDER BY hl.harvest_date DESC
            LIMIT 20
        """).fetchall()
        if not rows:
            return "No harvests recorded yet."

        type_ids = list({r[1] for r in rows if r[1]})
        names_map: dict[str, str] = {}
        if type_ids:
            placeholders = ",".join("?" * len(type_ids))
            try:
                name_rows = db.plant.execute(
                    f"SELECT plant_type_id, name FROM plant_types "
                    f"WHERE plant_type_id IN ({placeholders})",
                    type_ids,
                ).fetchall()
                names_map = {r[0]: r[1] for r in name_rows}
            except Exception:
                pass

        lines = []
        for r in rows:
            plant_name = names_map.get(r[1], "Unknown")
            qty = f"{r[3]} items" if r[3] else ""
            wt = f"{r[4]:.1f} lbs" if r[4] else ""
            amount = ", ".join(filter(None, [qty, wt])) or "recorded"
            note = f" — {r[5]}" if r[5] else ""
            lines.append(f"- {r[0]}: {plant_name} from {r[2]}: {amount}{note}")
        return "\n".join(lines)

    @tool
    def search_plant_knowledge(query: str) -> str:
        """Search the plant knowledge base for care guides, companion planting, pests, diseases, and growing tips."""
        if not plant_retriever.is_ready():
            return "Plant knowledge index not built yet. Run build/plant_rag/main.py first."
        try:
            results = plant_retriever.search(query, k=4)
            return "\n\n---\n\n".join(results)
        except Exception as exc:
            return f"Knowledge search failed: {exc}"

    return [
        get_garden_overview,
        get_planted_plants,
        get_weather_conditions,
        get_frost_dates,
        get_upcoming_sow_tasks,
        get_harvest_history,
        search_plant_knowledge,
    ]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(
    db: DatabaseConnections,
    food_prefs: str = "",
    interests: str = "",
) -> Any:
    """Compile the worker / evaluator / tools LangGraph with garden-aware context."""
    tools = make_tools(db)

    worker_llm = ChatOpenAI(model=WORKER_MODEL, temperature=0)
    worker_llm_with_tools = worker_llm.bind_tools(tools)

    evaluator_llm = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

    def worker(state: State) -> Dict[str, Any]:
        pref_ctx = f"\nThe gardener enjoys eating: {food_prefs}." if food_prefs else ""
        int_ctx  = f"\nThe gardener is interested in: {interests}." if interests else ""

        system_message = (
            f"You are a friendly, knowledgeable garden health assistant and planner. "
            f"Today is {date.today()}.{pref_ctx}{int_ctx}\n\n"
            "Your role:\n"
            "- Understand what the user wants to do in their garden and respond directly to their question\n"
            "- Call tools to get current data — plants, weather, frost, sow calendar, harvests\n"
            "- Identify specific problems and opportunities based on actual tool results\n"
            "- Be warm, conversational, and direct — get to the point\n\n"
            "STRICT RULES:\n"
            "- NEVER say you 'couldn't get' or 'don't have' data without actually calling the tool first. "
            "If you need frost dates, call get_frost_dates. If you need weather, call get_weather_conditions. "
            "Do NOT assume data is unavailable — call the tool and find out.\n"
            "- Do NOT open every reply with a garden summary ('you have 244 plants in New Bed...'). "
            "The user knows their garden. Respond to what they actually asked.\n"
            "- Do NOT repeat information you already gave in a previous turn unless the user asks.\n"
            "- If a tool returns no data, say so in one sentence and move on with advice.\n\n"
            "IMPORTANT — if tools return no garden data yet:\n"
            "- Stay engaged. Ask what they hope to grow or what they enjoy eating.\n"
            "- Be helpful even without data — discuss what grows well right now, timing, soil prep.\n\n"
            f"Success criteria for this task:\n{state['success_criteria']}\n\n"
            "Reply conversationally and concisely."
        )

        if state.get("feedback_on_work"):
            system_message += (
                f"\n\nYour previous response was rejected. Feedback:\n{state['feedback_on_work']}"
                "\nPlease address this and try again."
            )

        found_system = False
        messages = state["messages"]
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg.content = system_message
                found_system = True

        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def worker_router(state: State) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    def format_conversation(messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tool use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(state: State) -> Dict[str, Any]:
        last_response = state["messages"][-1].content

        system_message = (
            "You are an evaluator for a garden health assistant. "
            "Assess whether the assistant's response adequately addresses the user's question. "
            "A good response: attempted to use tools (even if some returned no data), "
            "gave specific advice based on whatever data was available, and was helpful. "
            "IMPORTANT: if a tool returned 'no data available' or similar, that counts as valid tool use — "
            "do NOT fail the response just because some data was missing. "
            "Only reject if the assistant ignored tools entirely or gave a completely unhelpful response."
        )

        user_message = (
            f"Evaluate this garden assistant conversation:\n\n"
            f"{format_conversation(state['messages'])}\n\n"
            f"Success criteria: {state['success_criteria']}\n\n"
            f"Final response from the assistant:\n{last_response}\n\n"
            "Did the assistant attempt to use tools? "
            "Did it give useful, specific advice based on whatever data was available? "
            "Remember: missing data is not a failure if the assistant acknowledged it and still helped."
        )

        if state["feedback_on_work"]:
            user_message += (
                f"\n\nPrevious feedback given: {state['feedback_on_work']}"
                "\nIf the same mistakes are repeated, set user_input_needed=True."
            )

        eval_result = evaluator_llm_with_output.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ])

        return {
            "messages": [{"role": "assistant", "content": f"[Evaluator: {eval_result.feedback}]"}],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
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

    return graph_builder.compile(checkpointer=_checkpointer)


# ── Entry point ───────────────────────────────────────────────────────────────

def render(db: DatabaseConnections) -> None:
    """Render the General Garden Bot chat tab."""
    st.subheader("🌱 General Garden Bot")
    st.caption("Ask anything about your garden — planning, health, weather, harvests.")

    with st.expander("⚙️ About you", expanded=False):
        food_prefs = st.text_input(
            "Foods you like to eat",
            placeholder="e.g. tomatoes, hot peppers, fresh herbs, salad greens",
            key="garden_bot_food_prefs",
        )
        interests = st.text_input(
            "Gardening interests",
            placeholder="e.g. organic growing, succession planting, companion planting",
            key="garden_bot_interests",
        )

    if "assistant_graph" not in st.session_state:
        st.session_state.assistant_graph = build_graph(db, food_prefs, interests)
        st.session_state.assistant_thread_id = _load_thread_id()
        # Restore display messages from the persisted graph state
        st.session_state.assistant_messages = _restore_messages(
            st.session_state.assistant_graph,
            st.session_state.assistant_thread_id,
        )

    graph = st.session_state.assistant_graph
    config = {"configurable": {"thread_id": st.session_state.assistant_thread_id}}

    _, col_ctrl = st.columns([6, 1])
    with col_ctrl:
        if st.button("🗑️ Clear", help="Start a new conversation (history is preserved)"):
            st.session_state.assistant_messages = []
            st.session_state.assistant_thread_id = _new_thread_id()
            st.rerun()

    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your garden…")
    if user_input:
        st.session_state.assistant_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking…"):
            try:
                result = graph.invoke(
                    {
                        "messages": [{"role": "user", "content": user_input}],
                        "success_criteria": SUCCESS_CRITERIA,
                        "feedback_on_work": None,
                        "success_criteria_met": False,
                        "user_input_needed": False,
                    },
                    config=config,
                )
                # messages[-1] is the evaluator note; messages[-2] is the worker's final response
                user_reply = result["messages"][-2]
                ai_content = user_reply.content if isinstance(user_reply, AIMessage) else str(user_reply)
            except Exception as exc:
                ai_content = f"Sorry, something went wrong: {exc}"

        with st.chat_message("assistant"):
            st.markdown(ai_content)
        st.session_state.assistant_messages.append({"role": "assistant", "content": ai_content})
