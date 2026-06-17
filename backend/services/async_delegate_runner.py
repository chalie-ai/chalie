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
``spawn`` registers a cancel event, runs the tool on a daemon thread through
the shared sync-run primitive, and returns the placeholder immediately
(spec §4.0 lever 1).

The thread captures the originating ``mp`` object — never a channel — so on
completion it delivers by synthesising a fresh assistant turn through that
``mp``'s own channel (spec §4.0 lever 2 / §4.4). Because the thread holds
the live ``mp``, delivery works even after the originating ACT loop has
closed. A single module-level singleton preserves the old module-dict
semantics: ``cancel`` from ``api/chat`` sees delegates spawned by any
processor (spec §4.4).

Consumers: ``abilities._dispatcher.ToolDispatcher._execute`` (spawn) and
``api/chat`` (the task-drawer ``active_ids`` / ``cancel`` endpoints).
"""

from __future__ import annotations

import contextvars
import logging
import threading
from uuid import uuid4

logger = logging.getLogger(__name__)


class AsyncDelegateRunner:
    """Owns the active backgrounded tool calls and their cancel events."""

    def __init__(self) -> None:
        self._active: dict[str, threading.Event] = {}

    def spawn(self, ability: object, params: dict, mp: object) -> str:
        """Copies the calling thread's contextvars so locale/timezone
        propagate exactly as on the synchronous path. Returns immediately
        — the ACT iteration is never blocked (spec §4.0)."""
        name = ability.get_name()
        delegate_id = f"{name}_{uuid4().hex[:8]}"
        cancel_event = threading.Event()
        self._active[delegate_id] = cancel_event
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._run, ability, params, mp, delegate_id, cancel_event),
            daemon=True,
        ).start()
        return (
            f"{name} dispatched (id: {delegate_id}). "
            "You will be notified when it completes."
        )

    def cancel(self, delegate_id: str) -> bool:
        """Set a running delegate's cancel event. Returns True if found."""
        ev = self._active.get(delegate_id)
        if ev:
            ev.set()
            return True
        return False

    def active_ids(self) -> list[str]:
        """IDs of currently running async delegates."""
        return list(self._active.keys())

    def _run(
        self,
        ability: object,
        params: dict,
        mp: object,
        delegate_id: str,
        cancel_event: threading.Event,
    ) -> None:
        """Lazy imports break the mutual deferral: the dispatcher imports
        this runner for its async branch, and api.chat imports this module
        for the cancel endpoints — neither can be top-level."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        try:
            try:
                result = ToolDispatcher._run(ability, params)
                result_text = ToolDispatcher._render(ability.get_name(), result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[AsyncDelegateRunner] execution failed for %s: %s",
                    delegate_id, exc,
                )
                result_text = None

            if result_text is not None and not cancel_event.is_set():
                try:
                    from api.chat import deliver_async_result  # noqa: PLC0415
                    deliver_async_result(mp, result_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[AsyncDelegateRunner] delivery failed for %s: %s",
                        delegate_id, exc,
                    )
        finally:
            self._active.pop(delegate_id, None)


# The shared singleton — one registry across every processor (spec §4.4).
async_delegate_runner = AsyncDelegateRunner()
