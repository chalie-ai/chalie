"""UserSynthesis — the single owner of the user-synthesis fact on the graph.

The user synthesis is one ``kind='machine_state'`` ``data_graph`` row, keyed
``user_summary``: a short prose portrait of the user. The user-summary channel
writes it once per synthesis turn; the prompt spine reads it back at assembly
time. Read only by exact key, never recalled — hence the operational
``machine_state`` lane, not ``system``.

This is thin CRUD over :class:`models.machine_state.MachineStateRow` — no ``mp``, no sibling
service, no prompt assembly. It exists so there is exactly ONE home for the
``user_summary`` read/write (Law 9): the raw reads in the cron cognition jobs and
the ``user_summary`` accessors that used to sit on ``DataGraphService`` all
collapse onto :meth:`get` / :meth:`upsert` here. The "should we re-synthesise?"
freshness gate — the trait/pattern-vs-summary timestamp check the synthesis cron
job runs before firing the channel — lives here too as :meth:`needs_refresh`,
since it reads the same ``user_summary`` row.

Provenance: ``persist_user_summary`` — the only writer — runs solely on the
``user_summary`` channel turn, so the ``source`` tag is a fixed constant rather
than a runtime channel lookup (no ``mp`` needed).
"""

from __future__ import annotations

import json
import logging

from models.behavioral_pattern import BehavioralPattern
from models.machine_state import MachineStateRow
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# The machine-state-lane key and its one write source.
_KEY = "user_summary"
_SOURCE = "user_summary"


class UserSynthesis:
    """Read and write the user-synthesis fact on the graph."""

    @classmethod
    def get(cls) -> str:
        """The user synthesis prose, ``""`` when no row is present. Live-row
        scope mirrors the spine's exact-key read
        (``MachineStateRow.active_by_key`` — ``active = 1``)."""
        return cls._value(_KEY)

    @classmethod
    def upsert(cls, content: str) -> None:
        """Persist the user synthesis to its key. A write failure is logged,
        never raised, so one bad synthesis can never crash the post-turn pipeline."""
        try:
            MachineStateRow.store(_KEY, content, source=_SOURCE)
        except Exception as exc:  # noqa: BLE001 — a bad synthesis write must not crash the turn
            logger.exception("user_synthesis.upsert failed key=%r: %s", _KEY, exc)

    @classmethod
    def needs_refresh(cls) -> bool:
        """True only when re-synthesis is needed — the gate the subconscious
        step checks BEFORE running the user-summary channel: no traits/patterns
        → False; traits/patterns exist but no summary → True; otherwise → True
        iff the newest trait/pattern is more recent than the summary row. Any
        read/parse failure is logged and treated as False (skip the turn)."""
        try:
            newest = cls._newest_trait_ts()
            if newest is None:
                return False
            summary_row = MachineStateRow.live().filter("key", _KEY).first()
            if summary_row is None:
                return True
            try:
                return parse_utc(newest) > parse_utc(summary_row.last_confirmed_at)
            except Exception as exc:  # noqa: BLE001 — a bad timestamp must not crash the tick
                logger.exception(
                    "user_synthesis.needs_refresh: parse_utc failed trait_ts=%r summary_ts=%r: %s",
                    newest, summary_row.last_confirmed_at, exc,
                )
                return False
        except Exception as exc:  # noqa: BLE001 — the gate must never crash the subconscious tick
            logger.warning("user_synthesis.needs_refresh failed: %s", exc)
            return False

    @staticmethod
    def _newest_trait_ts() -> str | None:
        """The most recent ``last_confirmed_at`` across live patterns, or
        ``None`` when the lane has no row."""
        row = BehavioralPattern.live().order_by("last_confirmed_at DESC").first()
        return row.last_confirmed_at if row is not None else None

    @classmethod
    def persist_user_summary(cls, text: str) -> None:
        """Parse the user-summary channel's ``{"summary"}`` JSON response and
        upsert the row. A malformed, non-object, or empty-``summary`` response is
        logged and dropped without writing — this prose feeds every system prompt
        via ``PromptService.user_definition``, so a bad turn must leave the
        standing synthesis intact rather than overwrite it. Every reject is
        logged, never raised; the write rides :meth:`upsert`'s own guard."""
        stripped = (text or "").strip()
        if not stripped:
            logger.warning("persist_user_summary: empty response — skipping")
            return
        body = stripped.removeprefix("```json").removeprefix("```").lstrip()
        body = body.removesuffix("```").rstrip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("persist_user_summary: JSON parse failed (%s) — skipping", exc)
            return
        if not isinstance(parsed, dict):
            logger.warning("persist_user_summary: parsed value is not a dict — skipping")
            return
        summary = (parsed.get("summary") or "").strip()
        if not summary:
            logger.warning("persist_user_summary: 'summary' missing/empty — skipping")
            return
        cls.upsert(summary)

    @staticmethod
    def _value(key: str) -> str:
        """The live value at ``('machine_state', key)`` — ``""`` when no active row."""
        row = MachineStateRow.active_by_key(key)
        return (row.value if row else "") or ""
