"""
WebSocketBroker — singleton connection holder + fire-and-forget broadcast.

Chalie has one user, one WebSocket connection, one chat UI.
No sequences, no buffers, no replay — if the WS drops the frontend
does location.reload() and gets full state from the DB.

Usage:
    On connect:    WebSocketBroker().connect(ws)
    On disconnect: WebSocketBroker().disconnect()
    Anywhere:      WebSocketBroker().broadcast({"type": "...", ...})
"""

import json
import logging
import threading

logger = logging.getLogger(__name__)


class WebSocketBroker:
    """Single-connection WebSocket broker. Fire-and-forget broadcast."""

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._ws = None
                    instance._lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def connect(self, ws) -> None:
        """Register the active WebSocket connection."""
        with self._lock:
            self._ws = ws

    def disconnect(self) -> None:
        """Clear the WebSocket reference when the connection closes."""
        with self._lock:
            self._ws = None

    def broadcast(self, data: dict) -> None:
        """Fire-and-forget push to the UI. No-op if nobody is listening."""
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        try:
            ws.send(json.dumps(data))
        except Exception as exc:
            logger.debug("[WS BROKER] broadcast failed (connection likely closed): %s", exc)
