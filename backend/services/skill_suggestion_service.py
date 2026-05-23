"""
skill_suggestion_service — Proactive skill suggestion after complex ACT loops.

Analyses completed ACT loops with 4+ tool-calling iterations and asks the LLM
whether the workflow is reusable enough to save as a skill. When yes, dispatches
a natural-language suggestion to the user via a fresh UMP turn.

Entry point: ``maybe_suggest_skill(act_trail, raw_input)`` — non-blocking,
never raises.
"""

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[SKILL_SUGGEST]"

_SKILLS_DB_PATH = (
    Path(__file__).resolve().parent.parent / "abilities" / "assets" / "skills.sqlite"
)

# Module-level cooldown: seconds between successive suggestions.
_COOLDOWN_SECONDS = 300

# Shared state — protected by GIL (single float write/read is atomic in CPython).
_last_suggested_at: float = 0.0

# Maximum characters taken from a single act_trail entry result for summarization.
_SNIPPET_MAX_CHARS = 200

_ANALYSIS_SYSTEM_PROMPT = """\
You analyse completed AI assistant workflows to determine whether they represent
a reusable, repeatable pattern worth saving as a skill playbook.

Default verdict: NOT reusable. Only return reusable=true when ALL of the
following are true:
  1. Multiple distinct tools were used in coordination (not a single tool loop).
  2. The workflow has a clear, recognisable start and end.
  3. The steps are generalisable — not tied to a specific entity (e.g. a single
     person's name, one specific URL, or a one-off event).
  4. A different user or the same user on a different day would follow essentially
     the same sequence of steps.

When analysing for reusability, identify and eliminate:
  - Dead ends: tool calls that failed or pivoted to a different approach.
  - Redundant calls: repeated lookups that produced no new information.
  - Suboptimal ordering: steps that would work better in a different sequence.

Produce an optimised workflow that removes waste from the actual execution trace.

Output format — JSON only, no prose:

If NOT reusable:
{"reusable": false}

If reusable:
{
  "reusable": true,
  "title": "<short imperative skill name, e.g. 'Research topic and summarise'>",
  "use_for": "<one sentence: when to invoke this skill>",
  "tags": "<comma-separated keywords>",
  "content": "<numbered steps, each on its own line, optimised — no dead ends>"
}

Be strict. Most workflows should produce {"reusable": false}.\
"""


def maybe_suggest_skill(act_trail: list[str], raw_input: str) -> None:
    """Fire background analysis for a completed ACT loop. Non-blocking. Never raises.

    Skips immediately when cooldown is active or skills.sqlite is absent.
    Logs the threshold-met event before spawning the daemon thread.
    """
    if not act_trail:
        return

    if not _SKILLS_DB_PATH.exists():
        return

    global _last_suggested_at
    now = time.monotonic()
    if now - _last_suggested_at < _COOLDOWN_SECONDS:
        logger.info("%s cooldown active — skipping", _LOG_PREFIX)
        return

    iteration_count = len(act_trail)
    logger.info(
        "%s threshold met — analysing (%d iterations)",
        _LOG_PREFIX,
        iteration_count,
    )

    t = threading.Thread(
        target=_analyse_and_suggest,
        args=(act_trail, raw_input, iteration_count),
        daemon=True,
        name="skill-suggest",
    )
    t.start()


def _analyse_and_suggest(
    act_trail: list[str],
    raw_input: str,
    iteration_count: int,
) -> None:
    """Run in the daemon thread: summarise trail, call LLM, parse, dispatch."""
    summary = _summarise_trail(act_trail)
    prompt = _build_analysis_prompt(raw_input, summary)

    from services.providers import Providers
    try:
        response = Providers.instance().send(
            user_prompt=prompt,
            system_prompt=_ANALYSIS_SYSTEM_PROMPT,
            job='subconscious',
            tools=[],
        )
    except Exception as exc:
        logger.warning("%s LLM call failed: %s", _LOG_PREFIX, exc)
        return

    parsed = _parse_analysis(response.text)
    if parsed is None:
        logger.info("%s verdict: SKIP", _LOG_PREFIX)
        return

    suggestion = _build_suggestion_message(
        parsed['title'],
        parsed['use_for'],
        parsed['content'],
        iteration_count,
    )

    try:
        from api.chat import dispatch_message
        dispatch_message(
            suggestion,
            source='skill_suggestion',
            hidden_input=True,
            intercept=False,
        )
    except Exception as exc:
        logger.warning("%s dispatch_message failed: %s", _LOG_PREFIX, exc)
        return

    global _last_suggested_at
    _last_suggested_at = time.monotonic()
    logger.info("%s dispatched: %s", _LOG_PREFIX, parsed['title'])


def _summarise_trail(act_trail: list[str]) -> list[dict]:
    """Compress each act_trail entry to (tool_name, short_snippet).

    Each entry has the form ``[tool_name(params)] result_text``. Extracts the
    tool name from the bracket prefix and truncates the result to keep the
    total prompt payload under ~2 000 tokens.
    """
    summary = []
    for entry in act_trail:
        tool_name = _extract_tool_name(entry)
        snippet = _extract_result_snippet(entry)
        summary.append({"tool": tool_name, "result": snippet})
    return summary


def _extract_tool_name(entry: str) -> str:
    """Parse the tool name from a ``[tool_name(…)] …`` act_trail entry."""
    if not entry.startswith("["):
        return "unknown"
    end = entry.find("(")
    if end == -1:
        end = entry.find("]")
    return entry[1:end].strip() if end > 1 else "unknown"


def _extract_result_snippet(entry: str) -> str:
    """Extract the result portion of an act_trail entry, capped at snippet limit."""
    bracket_end = entry.find("]")
    result = entry[bracket_end + 1:].strip() if bracket_end != -1 else entry
    return result[:_SNIPPET_MAX_CHARS]


def _build_analysis_prompt(raw_input: str, summary: list[dict]) -> str:
    """Build the LLM analysis prompt from the user query and summarised trail."""
    trail_text = json.dumps(summary, indent=2)
    return (
        f"## User Request\n{raw_input}\n\n"
        f"## Tool Execution Trail\n{trail_text}\n\n"
        "Analyse the trail above and determine whether this workflow is "
        "reusable. Output JSON only."
    )


def _parse_analysis(response_text: str) -> dict | None:
    """Parse LLM JSON response. Returns a dict (reusable workflow) or None (SKIP).

    Strips markdown code fences before parsing. Returns None when the verdict
    is not reusable, when parsing fails, or when required fields are absent.
    """
    if not response_text:
        return None

    payload = response_text.strip()
    if "```" in payload:
        parts = payload.split("```")
        payload = parts[1] if len(parts) >= 2 else payload
        if payload.startswith("json"):
            payload = payload[4:]

    try:
        data = json.loads(payload.strip())
    except json.JSONDecodeError as exc:
        logger.warning("%s failed to parse LLM response: %s", _LOG_PREFIX, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("%s unexpected response type: %s", _LOG_PREFIX, type(data))
        return None

    if not data.get("reusable"):
        return None

    required = ("title", "use_for", "content")
    if not all(data.get(k) for k in required):
        logger.warning("%s reusable=true but missing required fields", _LOG_PREFIX)
        return None

    return data


def _build_suggestion_message(
    title: str,
    use_for: str,
    content: str,
    iteration_count: int,
) -> str:
    """Build the natural-language suggestion delivered to the user via UMP."""
    return (
        f"I noticed that task involved {iteration_count} steps across multiple "
        f"tools. It looks like a reusable workflow that could be saved as a skill."
        f"\n\nHere are the details:\n"
        f"- **Name:** {title}\n"
        f"- **When to use:** {use_for}\n"
        f"- **Steps:**\n{content}\n\n"
        "Would you like me to save this as a skill? Just say \"yes\" and I'll "
        "create it for you."
    )
