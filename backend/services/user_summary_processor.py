"""
UserSummaryProcessor — one-shot internal processor for user-profile synthesis.

Reads up to 200 active ``kind='user_specific'`` rows from data_graph (ordered
by retrieval_weight DESC) and asks the LLM to return a JSON object with a
``short`` and a ``long`` synopsis.  ``postTurn()`` parses the response and
writes both synopses back to data_graph as ``kind='system'`` rows.

North star: /Volumes/llm/chalie-plans/message-processing.md
"""

import json
import logging
import re

from services.message_processor import MessageProcessor
from services.system_message_prompt import UserSummarySystemPrompt

logger = logging.getLogger(__name__)

_MAX_TRAIT_ROWS = 200


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

    def getUserDefinition(self) -> str:
        return "You are a synthesiser. The user is a real human whose traits you are distilling."

    def getUserPrompt(self) -> str:
        """Fetch up to 200 user_specific rows and render as ``key: value`` pairs.

        Renders ONLY key and value — no weights, no source, no timestamps.
        """
        try:
            from services.data_graph_service import get_data_graph_service

            rows = get_data_graph_service().fetch(
                kinds=['user_specific'],
                limit=_MAX_TRAIT_ROWS,
                order_by='retrieval_weight DESC',
            )
        except Exception as exc:
            logger.warning("[USER SUMMARY] getUserPrompt: fetch failed: %s", exc)
            rows = []

        if not rows:
            return "Facts:\n(no facts available)"

        lines = [
            f"{r['key']}: {r['value']}"
            for r in rows
            if r.get('key') and r.get('value')
        ]
        if not lines:
            return "Facts:\n(no facts available)"

        return "Facts:\n" + "\n".join(lines)

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
        stripped = re.sub(r'^```(?:json)?\s*', '', text)
        stripped = re.sub(r'\s*```$', '', stripped).strip()

        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as exc:
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
            # Write order matters for crash recovery. `_should_synthesise()` in
            # user_summary_worker compares latest user_specific trait timestamp
            # against the `user_summary` (short) row only. Writing `user_summary_long`
            # FIRST means: if the process crashes between these two calls the short
            # row's last_confirmed_at stays stale, _should_synthesise() returns True,
            # and the next tick re-synthesises both. Reverse order would mark the
            # short row fresh while user_summary_long was never written — a permanent
            # split-brain with no retry trigger.
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
