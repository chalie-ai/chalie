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
new spine's single tool-call chokepoint, and returns the placeholder immediately.

Each delegate is tracked with the metadata the Processes panel renders — the
tool name, the model's act-summary of what it's doing, and when it started —
plus the cooperative ``cancel_event`` the panel's stop button flips. Lifecycle
is pushed to every open client as ``subagent_start`` / ``subagent_end``; a
client that connects mid-flight rehydrates the same snapshot via ``active()``.

**MP lifetime:** this runner outlives the turn that spawned the
delegate — the turn is very likely already finished by the time the tool
actually completes. So each delegate gets its OWN dedicated **inert**
``MessageProcessor`` for its background span (live ``ws``/``dispatch_service``,
built from the originating turn's config/turn_id but never begun), instead of
holding a reference to the finished turn's mp. The originating ``mp`` is still
captured and threaded through to ``_deliver`` unchanged — that delivery step
needs the ORIGINATING turn's config/turn_id/metadata to land the synthesis
reply on the right channel/thread; only the in-flight execution (dispatch +
lifecycle broadcasts) runs off the dedicated one.

A single owned registry (the module-level ``async_delegate_runner``) lets
``cancel`` from ``api/endpoints/subagents`` reach delegates spawned by any processor.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from controllers.message_processor import MessageProcessor
from models.ws_message import WsMessage
from services.time_utils import utc_now
from services.websocket import Websocket

if TYPE_CHECKING:
    from datetime import datetime

    from abilities._ability import Ability
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Delegate:
    """One in-flight backgrounded tool call: the cooperative cancel signal, the
    metadata a Processes-panel row renders, and this delegate's own dedicated
    inert MP for its background span (``None`` only when the captured mp
    carried no config to map — never in production, see ``_dedicated_mp``)."""

    tool_name: str
    summary: str | None
    started_at: datetime
    cancel_event: threading.Event
    mp: "MessageProcessor | None"

    def snapshot(self, sub_id: str) -> dict[str, object]:
        """JSON-ready panel row for ``active()`` and the WS lifecycle events.
        The cancel event (and the dedicated mp) are internal and never exposed."""
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
        ``mp`` is the ORIGINATING (already-authorized) turn's mp: captured only to
        thread through to ``_deliver`` later, since the actual execution runs off
        this delegate's own dedicated inert MP.
        Copies the calling thread's contextvars so locale/timezone propagate
        exactly as on the synchronous path."""
        name = ability.get_name()
        delegate_id = f"{name}_{uuid4().hex[:8]}"
        dedicated_mp = self._dedicated_mp(mp)
        delegate = _Delegate(name, summary, utc_now(), threading.Event(), dedicated_mp)
        self._active[delegate_id] = delegate
        self._emit(dedicated_mp, self._EVENT_START, delegate.snapshot(delegate_id))
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._run, ability, params, mp, dedicated_mp, delegate_id, delegate.cancel_event),
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

    def _dedicated_mp(self, origin_mp: object) -> "MessageProcessor | None":
        """Build this delegate's OWN inert MP: live ``ws``/``dispatch_service``
        for the background span, mapped from the originating mp's config/turn_id —
        never begun (no DB/WS side-effect at construction, I2), and never the
        finished originating turn's own mp instance. ``None`` only when the
        captured mp carries no ``config`` to map from (a delivery-less test
        fixture — never a real processor, which always has one); the caller
        no-ops the dedicated-mp-dependent steps in that case rather than
        inventing a config."""
        config = getattr(origin_mp, "config", None)
        if config is None:
            logger.warning(
                "[AsyncDelegateRunner] captured mp has no config — running without "
                "a dedicated inert MP (no live ws/dispatch for this delegate)",
            )
            return None
        turn_id = cast("int", getattr(origin_mp, "turn_id", -1))
        return MessageProcessor(cast("ProcessorConfig", config), turn_id)

    def _emit(self, mp: "MessageProcessor | None", event_type: str, payload: dict[str, object]) -> None:
        """Push one delegate lifecycle event to every open client. No-op when
        this delegate has no dedicated mp (see ``_dedicated_mp``) — a delegate
        that cannot run signals nothing. The broadcast itself is global
        (``Websocket.broadcast``), so the mp here is only the run/no-run gate."""
        if mp is None:
            return
        Websocket.broadcast(WsMessage(type=event_type, **payload))

    def _deliver(self, origin_mp: object, result_text: str) -> None:
        """Append the finished delegate's outcome as a fresh assistant turn on the
        ORIGINATING thread — a plain ``MessageProcessor.process()`` call,
        independent of any foreground turn (each runs on its own thread, keyed by
        its own turn_id). The synthesis turn opens its OWN turn_executions row (see
        ``MessageProcessor.__init__``), so the Processes-panel stop control cancels
        it the same way as any other turn — no cancel handle needs threading in
        from the delegate.

        Reads the ORIGINATING mp's config/turn_id/metadata (never this delegate's
        dedicated inert mp): the reply must land on the channel/thread the delegate
        was spawned from. Inheriting the originating ``turn_id`` makes the
        synthesised reply a FORK into that same thread — the MessageProcessor
        switches itself to FORK view internally (no external flag). The input row
        is suppressed (``with_hidden_input``) and attachments dropped — both were
        already ingested on the originating turn and must not repeat here.
        """
        config = getattr(origin_mp, "config", None)
        if config is None:
            logger.warning("[AsyncDelegateRunner] async delivery skipped: captured mp has no config")
            return

        synth_config = config.with_hidden_input()
        metadata = dict(getattr(origin_mp, "metadata", None) or {})
        metadata["hidden_input"] = True
        metadata["attachments"] = []
        turn_id = getattr(origin_mp, "turn_id", None)
        # Full UserConfig turn: its lifecycle signals come from MessageProcessor
        # itself. Fire-and-forget — nothing consumes the final text, so the drive
        # thread is never joined.
        MessageProcessor.process(synth_config, result_text, metadata, turn_id if turn_id is not None else -1)

    def _run(
        self,
        ability: Ability,
        params: dict[str, object],
        mp: object,
        dedicated_mp: "MessageProcessor | None",
        delegate_id: str,
        cancel_event: threading.Event,
    ) -> None:
        """Run the tool through this delegate's dedicated mp and deliver its
        outcome as a later turn through the captured (originating) ``mp`` via
        :meth:`_deliver`, then always deregister and emit ``subagent_end`` on exit.

        If the delegate was cancelled while the tool was still running, the
        now-unwanted result is replaced by a short "cancelled by the user"
        notice — telling the model the work stopped so it can ask the user
        what to do next (never a silent drop). Delivery always proceeds: the
        notice is itself a normal synthesis turn, not a special case."""
        try:
            try:
                if dedicated_mp is None:
                    body = None
                else:
                    body = dedicated_mp.dispatch_service.dispatch(ability.get_name(), params)
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

            if body is not None:
                try:
                    self._deliver(mp, body)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[AsyncDelegateRunner] delivery failed for %s: %s",
                        delegate_id, exc,
                    )
        finally:
            self._active.pop(delegate_id, None)
            self._emit(dedicated_mp, self._EVENT_END, {"sub_id": delegate_id})


# The shared registry — one across every processor; each delegate holds its own
# dedicated inert MP for its background span, never a module-level MP.
async_delegate_runner = AsyncDelegateRunner()
