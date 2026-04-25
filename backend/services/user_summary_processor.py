"""
UserSummaryProcessor — one-shot internal processor for user-profile synthesis.

Reads up to 200 active ``kind='user_specific'`` rows from data_graph (ordered
by retrieval_weight DESC) and asks the LLM to return a JSON object with a
``short`` and a ``long`` synopsis.  ``postTurn()`` parses the response and
writes both synopses back to data_graph as ``kind='system'`` rows.

``send()`` self-skips via ``_should_synthesise()`` when no new traits have
arrived since the last synthesis.  Triggered once per :class:`SubconsciousWorker`
tick (v0.5.0 §5 step 4); the sentinel keeps repeated ticks idempotent.

North star: /Volumes/llm/chalie-plans/message-processing.md
"""

import json
import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import UserSummarySystemPrompt

logger = logging.getLogger(__name__)

_MAX_TRAIT_ROWS = 200
_MAX_PATTERN_ROWS = 25


def _format_pattern_line(content: dict) -> str:
    """Render a single behavioral_pattern content dict as a compact one-liner.

    Format varies by pattern class so the synthesiser sees the most relevant
    slots without having to parse raw JSON.

    Examples::

        meal_times / time_routine: dinner fri 18:30-20:00 (recurrence=7, last 2026-04-22)
        cuisine_pref / preference: italian (weight=0.85, last 2026-04-20)
        restaurant_pref / recurring_entity: Pizza Hut [restaurant] (recurrence=4, last 2026-04-18)
        meetings / event_cadence: meeting every 7d next 2026-04-29 (recurrence=3, last 2026-04-18)
    """
    vertical = content.get('vertical', 'unknown')
    cls = content.get('class', 'unknown')
    slots = content.get('slots') or {}
    recurrence = content.get('recurrence', 0)
    last_seen = content.get('last_seen', '')
    # Trim to date-only portion for brevity.
    last_seen_short = last_seen[:10] if last_seen else '?'

    if cls == 'time_routine':
        label = slots.get('event_label', '')
        day = slots.get('day_bucket', '')
        window = slots.get('hour_window', '')
        slot_str = f"{label} {day} {window}".strip()
        return (
            f"{vertical} / {cls}: {slot_str} "
            f"(recurrence={recurrence}, last {last_seen_short})"
        )

    if cls == 'preference':
        category = slots.get('category', '')
        choice = slots.get('choice', '')
        weight = slots.get('weight', '')
        slot_str = f"{category} {choice}".strip() if category else choice
        weight_part = f", weight={weight}" if weight != '' else ''
        return (
            f"{vertical} / {cls}: {slot_str} "
            f"(recurrence={recurrence}{weight_part}, last {last_seen_short})"
        )

    if cls == 'recurring_entity':
        kind = slots.get('kind', '')
        name = slots.get('name', '')
        bracket = f" [{kind}]" if kind else ''
        return (
            f"{vertical} / {cls}: {name}{bracket} "
            f"(recurrence={recurrence}, last {last_seen_short})"
        )

    if cls == 'event_cadence':
        event_type = slots.get('event_type', vertical)
        cadence = slots.get('cadence_days', '')
        next_exp = slots.get('next_expected', '')
        next_exp_short = next_exp[:10] if next_exp else ''
        cadence_part = f" every {cadence}d" if cadence != '' else ''
        next_part = f" next {next_exp_short}" if next_exp_short else ''
        return (
            f"{vertical} / {cls}: {event_type}{cadence_part}{next_part} "
            f"(recurrence={recurrence}, last {last_seen_short})"
        )

    # Fallback for unknown classes: dump vertical/class + recurrence.
    return f"{vertical} / {cls}: (recurrence={recurrence}, last {last_seen_short})"


class UserSummaryProcessor(MessageProcessor):
    """Internal processor that synthesises user traits into a short + long profile.

    One-shot: MAX_ITERATIONS=1, no tools, no transcript writes.
    The caller does not need to do any downstream work — postTurn() handles
    all storage.

    Usage::

        UserSummaryProcessor().send()
    """

    CHANNEL = 'user_summary'
    ROLE = 'user_summary'
    JOB = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = UserSummarySystemPrompt
    NATIVE_TOOLS: list[str] = []
    MAX_ITERATIONS = 1
    MAX_TIMEOUT = 120  # seconds
    SKIP_TRANSCRIPT_WRITE = True

    def __init__(self, metadata: dict | None = None):
        super().__init__(raw_input='', metadata=metadata)
        # Capture the LLM response text so postTurn() can parse it.
        self._last_response: str = ''

    def _should_synthesise(self) -> bool:
        """Return True when re-synthesis is warranted.

        Logic:
        - ``latest_trait_ts`` = MAX(last_confirmed_at) WHERE kind IN
          ('user_specific', 'behavioral_pattern') AND active=1
        - ``current_summary`` = row WHERE kind='system' AND key='user_summary' AND active=1

        Cases:
        - No traits or patterns at all → False (nothing to synthesise).
        - Traits/patterns exist, no summary → True (first synthesis needed).
        - Both exist → True only when latest trait/pattern is newer than the summary row.

        The ``kind IN ('user_specific', 'behavioral_pattern')`` filter on the MAX
        query is CRITICAL: querying all kinds would include the ``user_summary`` row
        itself, which is always confirmed more recently after synthesis, creating a
        permanent re-synthesis loop.  Both user-stated traits AND derived behavioural
        patterns are valid synthesis triggers.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.time_utils import parse_utc

            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(last_confirmed_at)
                    FROM data_graph
                    WHERE kind IN ('user_specific', 'behavioral_pattern')
                      AND active = 1
                      AND deleted_at IS NULL
                    """
                ).fetchone()
                latest_trait_ts = row[0] if row else None

                if latest_trait_ts is None:
                    return False

                summary_row = conn.execute(
                    """
                    SELECT last_confirmed_at
                    FROM data_graph
                    WHERE kind = 'system'
                      AND key = 'user_summary'
                      AND active = 1
                      AND deleted_at IS NULL
                    LIMIT 1
                    """
                ).fetchone()

            if summary_row is None:
                return True

            try:
                trait_dt = parse_utc(latest_trait_ts)
                summary_dt = parse_utc(summary_row[0])
            except Exception as exc:
                logger.error(
                    "[USER SUMMARY] _should_synthesise: parse_utc failed on "
                    "trait_ts=%r summary_ts=%r: %s",
                    latest_trait_ts,
                    summary_row[0],
                    exc,
                )
                return False

            return trait_dt > summary_dt

        except Exception as exc:
            logger.warning("[USER SUMMARY] _should_synthesise failed: %s", exc)
            return False

    def send(self, request_id: str | None = None) -> str:
        """Run synthesis only when new traits exist since the last synopsis.

        Delegates to the base ACT loop if synthesis is warranted.  Silently
        returns the empty string (as if the turn produced no text) when the
        synopsis is already current — matching the base-class cap-exit
        convention for internal processors.
        """
        if not self._should_synthesise():
            logger.info("[USER SUMMARY] No new traits since last synthesis; skipping")
            return ''
        return super().send(request_id=request_id)

    def getUserDefinition(self) -> str:
        return "You are a synthesiser. The user is a real human whose traits you are distilling."

    def getUserPrompt(self) -> str:
        """Assemble the synthesis prompt from user traits and active behavioural patterns.

        Section 1 — Facts (key: value pairs, up to _MAX_TRAIT_ROWS).
        Section 2 — Behavioural patterns (compact digest, up to _MAX_PATTERN_ROWS).
        Section 2 is omitted entirely when no active patterns exist.

        Renders ONLY key and value for traits — no weights, no source, no timestamps.
        """
        # ── Section 1: user_specific traits ──────────────────────────────────────
        try:
            from services.data_graph_service import get_data_graph_service

            rows = get_data_graph_service().fetch(
                kinds=['user_specific'],
                limit=_MAX_TRAIT_ROWS,
                order_by='retrieval_weight DESC',
            )
        except Exception as exc:
            logger.warning("[USER SUMMARY] getUserPrompt: trait fetch failed: %s", exc)
            rows = []

        if not rows:
            facts_section = "Facts:\n(no facts available)"
        else:
            lines = [
                f"{r['key']}: {r['value']}"
                for r in rows
                if r.get('key') and r.get('value')
            ]
            facts_section = "Facts:\n" + "\n".join(lines) if lines else "Facts:\n(no facts available)"

        # ── Section 2: active behavioural patterns ───────────────────────────────
        active_patterns = []
        try:
            import json as _json

            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            with db.connection() as conn:
                pattern_rows = conn.execute(
                    """
                    SELECT value
                    FROM data_graph
                    WHERE kind = 'behavioral_pattern'
                      AND active = 1
                      AND deleted_at IS NULL
                    ORDER BY last_confirmed_at DESC
                    LIMIT ?
                    """,
                    (_MAX_PATTERN_ROWS,),
                ).fetchall()
            for (value_json,) in pattern_rows:
                try:
                    content = _json.loads(value_json)
                    if content.get('status') == 'active':
                        active_patterns.append(content)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("[USER SUMMARY] active pattern fetch failed: %s", exc)

        if not active_patterns:
            return facts_section

        pattern_lines = [_format_pattern_line(p) for p in active_patterns]
        patterns_section = (
            "## Behavioural patterns (recurrence, last seen)\n"
            + "\n".join(f"- {line}" for line in pattern_lines)
        )

        return facts_section + "\n\n" + patterns_section

    def store(self, llm_response: str) -> None:
        """Capture the LLM response for postTurn(), then call base (no-op for SKIP_TRANSCRIPT_WRITE)."""
        self._last_response = llm_response
        super().store(llm_response)

    def postTurn(self) -> None:
        """Parse the LLM JSON response and write both synopsis rows to data_graph.

        Expected response shape::

            {"short": "...", "long": "..."}

        On any parse failure or missing/empty keys: logs a warning and returns
        without writing anything.  Pre-existing ``user_summary`` rows are never
        corrupted by a failed parse.
        """
        text = (self._last_response or '').strip()
        if not text:
            logger.warning("[USER SUMMARY] postTurn: empty LLM response — skipping write")
            return

        # Strip markdown code fences if the model wrapped the JSON.
        # Pure stdlib string ops — no regex, no backtracking risk (Sonar S5852).
        stripped = text.removeprefix('```json').removeprefix('```').lstrip()
        stripped = stripped.removesuffix('```').rstrip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[USER SUMMARY] postTurn: JSON parse failed (%s) — skipping write. raw=%r",
                exc,
                text[:200],
            )
            return

        if not isinstance(parsed, dict):
            logger.warning(
                "[USER SUMMARY] postTurn: parsed value is not a dict — skipping write"
            )
            return

        short = (parsed.get('short') or '').strip()
        long_ = (parsed.get('long') or '').strip()

        if not short or not long_:
            logger.warning(
                "[USER SUMMARY] postTurn: 'short' or 'long' missing/empty — skipping write"
            )
            return

        try:
            from services.data_graph_service import get_data_graph_service

            dgs = get_data_graph_service()
            # Write order matters for crash recovery. `_should_synthesise()` compares
            # latest user_specific trait timestamp against the `user_summary` (short)
            # row only. Writing `user_summary_long` FIRST means: if the process
            # crashes between these two calls the short row's last_confirmed_at stays
            # stale, _should_synthesise() returns True, and the next caller
            # re-synthesises both. Reverse order would mark the short row fresh while
            # user_summary_long was never written — a permanent split-brain with no
            # retry trigger.
            dgs.store(
                kind='system',
                key='user_summary_long',
                value=long_,
                source='user_summary_processor',
            )
            dgs.store(
                kind='system',
                key='user_summary',
                value=short,
                source='user_summary_processor',
            )
            logger.info(
                "[USER SUMMARY] postTurn: wrote user_summary (%d chars) and "
                "user_summary_long (%d chars)",
                len(short),
                len(long_),
            )
        except Exception as exc:
            logger.warning("[USER SUMMARY] postTurn: data_graph write failed: %s", exc)
