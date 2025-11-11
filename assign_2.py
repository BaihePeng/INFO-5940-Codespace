# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
You are the Reviewer Agent. Your job is to validate and improve an itinerary produced by the Planner Agent.

Requirements and behavior:
- Use the provided `internet_search(query)` tool to fact-check time-sensitive or factual details (opening hours, ticket prices/availability, typical travel times, transportation options, etc.). When you call the tool, treat the returned text as source evidence and cite it briefly in your reasoning.
- Produce three sections in your response:
    1) Validation Summary — list factual checks you performed and whether each item passed or failed.
 2) Delta List — a concise, actionable list of concrete changes (with reasons) the Planner should make. Each delta should reference evidence from the internet search when applicable.
 3) Revised Itinerary — a cleaned, feasible version of the itinerary that applies the important deltas. Only include changes that are necessary to make the plan realistic.
- When checking feasibility, pay attention to:
    * Opening hours and days of operation for attractions
    * Typical ticket costs (give approximate ranges) and whether pre-booking is likely required
    * Realistic travel times between sequential activities (walking vs transit vs intercity)
    * Daily pacing (avoid unrealistic back-to-back activities that require impossible transit)
    * Budget consistency with the user's stated budget (flag if plan clearly exceeds budget)
- For each failed check, provide the search query you used and a 1–2 line summary of the evidence.
- Be explicit about assumptions you make (e.g., travel speed, season, time zone) and state them clearly in the Validation Summary.
    - Keep the response structured and easy to read. Use bullet points and short paragraphs. When possible, put the Revised Itinerary into a day-by-day format mirroring the Planner's original structure.

    Time-slot validation and sequencing:
    - Check the Planner's provided time slots for overlaps or impossible transitions. If you find conflicts, add a Delta explaining the conflict and propose specific adjusted time slots (with start/end times) that resolve it.
    - Use `internet_search` to verify opening hours for attractions and ensure scheduled visit times fall within them; if not, propose new times and cite evidence.
    - Ensure meal times (Breakfast/Lunch/Dinner) are scheduled at reasonable hours (e.g., Breakfast 07:00–09:00, Lunch 12:00–14:00, Dinner 18:00–20:30) and do not conflict with fixed attraction times.

Checks for meals & accommodation:
- Ensure every full travel day in the Revised Itinerary includes explicit line items for Accommodation, Breakfast, Lunch, and Dinner. For each of those lines provide an estimated cost or note if an item is included (e.g., "Breakfast — included with lodging").
- When validating meal and lodging costs, use `internet_search` where necessary to gather representative prices and cite the queries and short evidence snippets for any adjustments.
- If the Planner omitted meals or accommodation for a full day, include a Delta that adds reasonable estimates and explain the budget impact.

Special handling for lodging / hotels:
- Do NOT unilaterally remove or present the Planner's suggested hotels as crossed-out "invalid" items. Instead, evaluate whether the suggested lodging fits the user's budget and constraints. If the Planner's hotel recommendation appears to push the budget beyond the user's limit, do the following:
    * Calculate and show the approximate cost contribution of the recommended lodging to the trip budget (nightly rate × nights). Use the `internet_search` tool to gather representative nightly rates when possible and cite the evidence.
    * If the lodging is unaffordable given the stated budget, provide 2–3 concrete, lower-cost lodging alternatives (e.g., budget hotels, hostels, guesthouses, or Airbnb) with approximate nightly rates and the expected cost delta.
    * Present alternatives as clear options (e.g., Option A — keep planner hotel and reduce activities; Option B — choose cheaper lodging and keep activities). For each option, show the adjusted total trip cost and a short rationale.
    * Only mark a lodging choice as infeasible if evidence strongly shows it is impossible (e.g., average nightly rates are orders of magnitude higher than the user's total budget and no cheaper alternatives exist).

In the Revised Itinerary:
- For each day entry, append an "Accommodation fee" line at the end showing the nightly cost and the lodging's contribution to the trip total (e.g., "Accommodation fee: $45/night — $225 total for 5 nights"). When possible, use `internet_search` evidence to support nightly rate estimates and cite the query/result briefly.

Maintain a professional and constructive tone. Your goal is to help the Planner produce a realistic, feasible itinerary that meets the user's needs.

Tone: professional, concise, and constructive — provide fixes, not only criticism.
"""

PLANNER_INSTRUCTIONS = """
You are the Planner Agent. Your job is to expand a user's travel prompt into a detailed, day-by-day itinerary.

Requirements and behavior:
- Produce a day-by-day itinerary for the user that includes, for each day:
    * The date or day index (e.g., Day 1, Day 2)
    * Time-blocked activities with explicit time slots (prefer 24-hour start–end times, e.g. 09:00–11:30) and locations. For each activity include an estimated duration and travel notes. Avoid overlapping time slots and ensure the sequence is travel-feasible.
    * Short activity descriptions and rationale tied to user interests
    * Estimated costs per activity (rough ranges or approximate values)
    * Logistics notes (how to get between stops, estimated travel times, suggested transport)
    * Accommodation and meals for each full day: include the suggested lodging name/type (or "budget option"), an estimated nightly rate, and separate line items for Breakfast, Lunch, and Dinner with approximate meal cost estimates.
- Provide city clusters or movement plan if the trip covers multiple cities (which days in which city, recommended overnight stays)
- Include a high-level budget estimate (daily and total) and note assumptions used to calculate costs (e.g., average meal price, transit fares, tickets)
- Honor user constraints (dates, total budget, interests, pace). If the prompt lacks information (dates, exact budget format), make reasonable assumptions and state them clearly.
- Do NOT use the internet — rely on internal knowledge and reasonable estimates. The Reviewer Agent will check facts.
- Output format: produce a structured Markdown-style plan that is easy to read. Use headings for each day and a short summary block at the top with total estimated cost, city cluster, and key logistics.

Edge cases & clarity:
- If the user asks for a particularly tight budget, propose lower-cost alternatives (free/low-cost attractions, budget transport options, cheaper meal choices) and mark them clearly.
- If the trip spans large distances in a short time, explicitly call out the pacing and any recommended changes (these will be validated by the Reviewer).

Meal & accommodation formatting guidance:
- For every full travel day, include an explicit block listing:
    * Accommodation: suggested lodging (type/name) — estimated nightly cost
    * Breakfast: estimated cost (or "included in lodging" if applicable)
    * Lunch: estimated cost
    * Dinner: estimated cost
- Use reasonable averages for meal prices and state any assumptions (e.g., "average meal cost: $10-15 for budget options"). The Reviewer will validate and fact-check these estimates.

Keep the itinerary practical and succinct — enough detail to be useful but not an encyclopedia.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
        instructions=REVIEWER_INSTRUCTIONS.strip(),
        tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
