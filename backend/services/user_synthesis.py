"""UserSynthesis — the single owner of the user-synthesis facts on the graph.

The user synthesis is two ``kind='machine_state'`` ``data_graph`` rows: a short
(``user_summary``) and a long (``user_summary_long``) prose portrait of the user.
The user-summary channel writes both once per synthesis turn; the prompt spine and
the DMN reflection loop read them back at assembly time. Read only by exact key,
never recalled — hence the operational ``machine_state`` lane, not ``system``.

This is thin CRUD over :class:`models.machine_state.MachineStateRow` — no ``mp``, no sibling
service, no prompt assembly. It exists so there is exactly ONE home for the
``user_summary`` / ``user_summary_long`` read/write pair (Law 9): the DB-reaching
statics that used to live in ``DmnConfig`` (§2.5 config strip), the raw reads in
the cron cognition jobs and the ``user_summary*`` accessors that used to sit on
``DataGraphService`` all collapse onto :meth:`get` / :meth:`upsert` here. The
"should we re-synthesise?" freshness gate — the trait/pattern-vs-summary timestamp
check the synthesis cron job runs before firing the channel — lives here too as
:meth:`needs_refresh`, since it reads the same ``user_summary`` row.

Provenance: ``persist_user_summary`` — the only writer — runs solely on the
``user_summary`` channel turn, so the ``source`` tag is a fixed constant rather
than a runtime channel lookup (no ``mp`` needed).
"""

from __future__ import annotations

import json
import logging

from models.behavioral_pattern import BehavioralPattern
from models.fact import FactRow
from models.machine_state import MachineStateRow
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# The two machine-state-lane keys and their one write source.
_KEY_SHORT = "user_summary"
_KEY_LONG = "user_summary_long"
_SOURCE = "user_summary"


class UserSynthesis:
    """Read and write the user-synthesis facts (short + long) on the graph."""

    @classmethod
    def get(cls, shorthand: bool = False) -> str:
        """The user synthesis prose — the short (``shorthand=True``) portrait, or
        the long one falling back to the short when no long row exists. ``""`` when
        neither row is present. Live-row scope mirrors the spine's exact-key read
        (``MachineStateRow.active_by_key`` — ``active = 1``)."""
        if shorthand:
            return cls._value(_KEY_SHORT)
        return cls._value(_KEY_LONG) or cls._value(_KEY_SHORT)

    @classmethod
    def upsert(cls, content: str, shorthand: bool = True) -> None:
        """Persist the user synthesis to its key — the short row when
        ``shorthand`` (default), the long row otherwise. A write failure is logged,
        never raised, so one bad synthesis can never crash the post-turn pipeline."""
        key = _KEY_SHORT if shorthand else _KEY_LONG
        try:
            MachineStateRow.store(key, content, source=_SOURCE)
        except Exception as exc:  # noqa: BLE001 — a bad synthesis write must not crash the turn
            logger.exception("user_synthesis.upsert failed key=%r: %s", key, exc)

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
            summary_row = MachineStateRow.live().filter("key", _KEY_SHORT).first()
            if summary_row is None:
                return True
            try:
                return parse_utc(newest) > parse_utc(summary_row.last_confirmed_at)
            except Exception as exc:  # noqa: BLE001 — a bad timestamp must not crash the tick
                logger.error(
                    "user_synthesis.needs_refresh: parse_utc failed trait_ts=%r summary_ts=%r: %s",
                    newest, summary_row.last_confirmed_at, exc,
                )
                return False
        except Exception as exc:  # noqa: BLE001 — the gate must never crash the subconscious tick
            logger.warning("user_synthesis.needs_refresh failed: %s", exc)
            return False

    @staticmethod
    def _newest_trait_ts() -> str | None:
        """The most recent ``last_confirmed_at`` across live traits and patterns,
        or ``None`` when neither lane has a row."""
        rows = [
            FactRow.live().order_by("last_confirmed_at DESC").first(),
            BehavioralPattern.live().order_by("last_confirmed_at DESC").first(),
        ]
        stamps = [row.last_confirmed_at for row in rows if row is not None]
        return max(stamps) if stamps else None

    @classmethod
    def persist_user_summary(cls, text: str) -> None:
        """Parse the user-summary channel's ``{"short", "long"}`` JSON response and
        upsert the long then the short row — long FIRST so a crash mid-write leaves
        the richer fact in place (legacy ``PersistUserSummaryHook``). Every reject
        is logged, never raised; the two writes ride :meth:`upsert`'s own guard."""
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
        short = (parsed.get("short") or "").strip()
        long_ = (parsed.get("long") or "").strip()
        if not short or not long_:
            logger.warning("persist_user_summary: 'short'/'long' missing/empty — skipping")
            return
        cls.upsert(long_, shorthand=False)
        cls.upsert(short, shorthand=True)

    @staticmethod
    def _value(key: str) -> str:
        """The live value at ``('machine_state', key)`` — ``""`` when no active row."""
        row = MachineStateRow.active_by_key(key)
        return (row.value if row else "") or ""
