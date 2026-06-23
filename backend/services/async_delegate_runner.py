# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AsyncDelegateRunner — daemon-thread lifecycle for backgrounded tool calls.

When the model opts a tool call into the background (per-call ``async: true``,
exposed only on ``SUPPORTS_ASYNC`` channels), the ACT iteration must not block.
``spawn`` registers the delegate, runs the tool on a daemon thread through the
shared sync-run primitive, and returns the placeholder immediately.

Each delegate is tracked with the metadata the Processes panel renders — the
tool name, the model's act-summary of what it's doing, and when it started —
plus the cooperative ``cancel_event`` the panel's stop button flips. Lifecycle
is pushed to every open client as ``subagent_start`` / ``subagent_end``; a
client that connects mid-flight rehydrates the same snapshot via ``active()``.

The thread captures the originating ``mp`` object so it can deliver the result
as a fresh assistant turn through that ``mp``'s channel, threading the
delegate's ``cancel_event`` into that synthesis turn — so stopping a delegate
also aborts a spiralling synthesis chain at its next boundary, reusing the
existing cancel plumbing rather than a second kill path. A single module-level
singleton lets ``cancel`` from ``api/chat`` reach delegates spawned by any
processor.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from services.time_utils import utc_now
from services.websocket_broker import WebSocketBroker

if TYPE_CHECKING:
    from datetime import datetime

    from abilities._ability import Ability

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Delegate:
    """One in-flight backgrounded tool call: the cooperative cancel signal plus
    the metadata a Processes-panel row renders."""

    tool_name: str
    summary: str | None
    started_at: datetime
    cancel_event: threading.Event

    def snapshot(self, sub_id: str) -> dict[str, object]:
        """JSON-ready panel row for ``active()`` and the WS lifecycle events.
        The cancel event is internal and never exposed."""
        return {
            "sub_id": sub_id,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "started_at": self.started_at.isoformat(),
        }


class AsyncDelegateRunner:
    """Owns the in-flight backgrounded tool calls, their panel metadata, and the
    cancel events the Processes panel flips to stop them."""

    _EVENT_START = "subagent_start"
    _EVENT_END = "subagent_end"

    def __init__(self) -> None:
        self._active: dict[str, _Delegate] = {}

    def spawn(self, ability: Ability, params: dict[str, object], mp: object, summary: str | None) -> str:
        """Register the delegate, push ``subagent_start`` to open clients, and run
        the tool on a daemon thread — the ACT iteration is never blocked.
        ``summary`` is the model's act-summary (what the delegate is doing).
        Copies the calling thread's contextvars so locale/timezone propagate
        exactly as on the synchronous path."""
        name = ability.get_name()
        delegate_id = f"{name}_{uuid4().hex[:8]}"
        delegate = _Delegate(name, summary, utc_now(), threading.Event())
        self._active[delegate_id] = delegate
        self._emit(self._EVENT_START, delegate.snapshot(delegate_id))
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._run, ability, params, mp, delegate_id, delegate.cancel_event),
            daemon=True,
        ).start()
        return (
            f"{name} dispatched (id: {delegate_id}). "
            "You will be notified when it completes."
        )

    def cancel(self, delegate_id: str) -> bool:
        """Flip a running delegate's cancel event — the panel's stop control.
        Returns True if the delegate was active."""
        delegate = self._active.get(delegate_id)
        if delegate:
            delegate.cancel_event.set()
            return True
        return False

    def active(self) -> list[dict[str, object]]:
        """Snapshots of the in-flight delegates — hydrates the Processes panel."""
        return [delegate.snapshot(sub_id) for sub_id, delegate in self._active.items()]

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        """Push one delegate lifecycle event to every open client. No-op when no
        client is connected; a fresh client rehydrates via ``active()``."""
        WebSocketBroker().broadcast({"type": event_type, **payload})

    def _run(
        self,
        ability: Ability,
        params: dict[str, object],
        mp: object,
        delegate_id: str,
        cancel_event: threading.Event,
    ) -> None:
        """Run the tool and deliver its outcome as a later turn through the
        captured ``mp``, then always deregister and emit ``subagent_end`` on exit.

        A normal result is delivered with the delegate's ``cancel_event`` threaded
        into the synthesis turn, so stopping the delegate also aborts a spiralling
        synthesis at its next boundary. If the delegate was cancelled while the
        tool was still running, the now-unwanted result is replaced by a short
        "cancelled by the user" notice — delivered under a FRESH event so the
        notice itself always lands — telling the model the work stopped so it can
        ask the user what to do next (never a silent drop).

        Lazy imports break the mutual deferral: the dispatcher imports this
        runner for its async branch, and api.chat imports this module for the
        cancel endpoints — neither can be top-level."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        try:
            try:
                body = ToolDispatcher._render(ability.get_name(), ToolDispatcher._run(ability, params))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[AsyncDelegateRunner] execution failed for %s: %s",
                    delegate_id, exc,
                )
                body = None

            if cancel_event.is_set():
                body = (
                    f"`{ability.get_name()}` was cancelled by the user. "
                    "Ask the user for follow-up actions."
                )
                logger.info(
                    "[AsyncDelegateRunner] %s cancelled by user — delivering notice: %s",
                    delegate_id, body,
                )
                cancel_event = threading.Event()

            if body is not None:
                try:
                    from api.chat import deliver_async_result  # noqa: PLC0415
                    deliver_async_result(mp, body, cancel_event)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[AsyncDelegateRunner] delivery failed for %s: %s",
                        delegate_id, exc,
                    )
        finally:
            self._active.pop(delegate_id, None)
            self._emit(self._EVENT_END, {"sub_id": delegate_id})


# The shared singleton — one registry across every processor.
async_delegate_runner = AsyncDelegateRunner()
