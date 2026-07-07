# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Active-record model for one ``turn_executions`` row — the DB-backed lifecycle
of a single MessageProcessor turn (§4.1; §6.3/§6.5/§6.6/§6.7).

Pure CRUD (Rule-3 depth): no mp, no service, no WS. The state vocabulary and the
two open-row query scopes the ``TurnExecutionService`` reads through live here;
the emit/envelope and cross-instance cancel logic are the service's job.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class TurnExecution(Model):
    """One ``turn_executions`` row: is this turn still running, and did someone
    ask it to stop. Projection is the inherited full-field ``to_json`` — every
    field is a real column and nothing is hidden, so the JSON is BOTH the WS
    lifecycle frame AND the REST ``TurnExecutionDTO`` payload (dual shape, §6.3);
    ``type`` rides along because it is a column, not a transient envelope."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "channel", "type", "turn_id", "started_at", "ended_at",
        "cancel_requested", "state", "stop_reason",
    )

    @classmethod
    def get_table(cls) -> str:
        return "turn_executions"

    #: The ``state`` vocabulary — one non-terminal, three terminal (DB CHECK, §6.5).
    WORKING: ClassVar[str] = "working"
    COMPLETED: ClassVar[str] = "completed"
    CANCELLED: ClassVar[str] = "cancelled"
    CRASHED: ClassVar[str] = "crashed"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access from services).
    channel: str
    type: str | None
    turn_id: int
    started_at: str
    ended_at: str | None
    cancel_requested: bool
    state: str
    stop_reason: str | None

    @classmethod
    def open_turn(cls, channel: str, turn_id: int) -> "TurnExecution | None":
        """The single live row for (channel, turn_id) — the cancel target (§6.7).
        ``turn_id`` is a per-channel monotonic counter, not globally unique, so
        the match is channel-scoped; ``ended_at IS NULL`` restricts it to the one
        still-running row (mirrors today's ``request_cancel_by_turn`` targeting)."""
        return (
            cls.filter("channel", channel)
            .filter("turn_id", turn_id)
            .filter("ended_at", None, "IS")
            .first()
        )

    @classmethod
    def open_rows(cls) -> list["TurnExecution"]:
        """Every row a killed process left open (``ended_at IS NULL``) — the boot
        orphan sweep reads these and stamps each ``crashed`` (§6.6: a real DB read,
        never a fabricated in-memory row)."""
        return cls.filter("ended_at", None, "IS").get()
