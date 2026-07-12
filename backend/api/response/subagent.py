"""Outbound DTOs for the subagents endpoint group.

An active async delegate (backgrounded tool call) is a flat row — one per
in-flight delegate, keyed by its own ``sub_id`` rather than a turn_id: a
backgrounded call is fire-and-notify and outlives the ACT iteration that
spawned it (see services/async_delegate_runner.py). :class:`SubagentStopResult`
is the cancel ack: ``cancelled`` is True when a live delegate was stopped,
else ``reason`` is ``"not_found"`` — an unknown/finished sub_id is a harmless
ack, never a 404, since delegates finish on their own and a 404 would error
the panel on a benign race.
"""

from __future__ import annotations

from datetime import datetime

from .response import Response


class Subagent(Response):
    """One active async delegate row — hydrates the Processes panel on
    load/reconnect, since WS push events (``subagent_start``/``subagent_end``)
    are missed while the client is disconnected."""

    sub_id: str
    tool_name: str
    summary: str | None
    started_at: datetime


class SubagentStopResult(Response):
    """200 ack for ``DELETE /api/subagents/<sub_id>`` (subagent cancel).
    ``cancelled`` is True when a live delegate was stopped, else ``reason``
    is ``"not_found"`` — never a 404 (see module docstring)."""

    cancelled: bool | None = None
    reason: str | None = None
