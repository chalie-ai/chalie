"""ActEventEmitter — the single gate for ACT-loop WebSocket events.

A MessageProcessor whose config carries a ``broadcast_to`` target streams live
ACT tool start/end events to the UI; a background loop (``broadcast_to=None``)
stays silent. This object encapsulates that one gate so every emit site shares
the same rule and a dead socket never breaks the loop.

Constructed per-config by the tool dispatcher, this emitter carries only the
tool ``act_tool_start`` / ``act_tool_end`` events. Mid-turn assistant prose is
a separate concern: under the recursive turn chain (TKT-1070) each step that
makes tool calls broadcasts its prose live as an interim ``message`` event
(``MessageProcessor._emit_interim``) and persists it as its own transcript row
(``_store_row``), so a single turn produces multiple assistant rows — one per
step — not one end message at the end.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ActEventEmitter:
    """Broadcasts ACT events for one config, or no-ops when it has no
    ``broadcast_to`` target."""

    def __init__(self, config: object) -> None:
        self._config = config

    def emit(self, event: dict) -> None:
        """Broadcast *event* iff the bound config has a ``broadcast_to`` target;
        swallow broker errors so a dead socket never breaks the ACT loop."""
        if self._config is None or getattr(self._config, "broadcast_to", None) is None:
            return
        from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
        try:
            WebSocketBroker().broadcast(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ActEventEmitter] broadcast failed: %s", exc)
