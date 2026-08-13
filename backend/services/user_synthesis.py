"""UserSynthesis — the single owner of the user-synthesis fact on the graph.

The user synthesis is one ``kind='machine_state'`` ``data_graph`` row, keyed
``user_summary``: a short prose portrait of the user. The prompt spine reads
it back at assembly time. Read only by exact key, never recalled — hence the
operational ``machine_state`` lane, not ``system``.

This is thin CRUD over :class:`models.machine_state.MachineStateRow` — no ``mp``, no sibling
service, no prompt assembly. It exists so there is exactly ONE home for the
``user_summary`` read/write (Law 9): the raw reads in the cron cognition jobs and
the ``user_summary`` accessors that used to sit on ``DataGraphService`` all
collapse onto :meth:`get` / :meth:`upsert` here.
"""

from __future__ import annotations

import logging

from models.machine_state import MachineStateRow

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

    @staticmethod
    def _value(key: str) -> str:
        """The live value at ``('machine_state', key)`` — ``""`` when no active row."""
        row = MachineStateRow.active_by_key(key)
        return (row.value if row else "") or ""
