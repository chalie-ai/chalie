"""
WebSocketBroker — singleton connection registry + fire-and-forget fan-out.

Chalie is single-user, but that one user may have the chat UI open on
several tabs/devices at once. The broker holds every live server→client
WebSocket and fans each broadcast out to all of them, so every open client
stays in sync. No sequences, no buffers, no replay: a dropped socket
reconnects on its own (the client retries with backoff and runs a liveness
watchdog), and the durable conversation stays aligned because every user and
assistant message is broadcast on this channel as it happens — each surface
renders what it receives.

Usage:
    On connect:    WebSocketBroker().connect(ws)
    On disconnect: WebSocketBroker().disconnect(ws)
    Anywhere:      WebSocketBroker().broadcast({"type": "...", ...})
"""

import json
import logging
import threading

logger = logging.getLogger(__name__)


class WebSocketBroker:
    """Multi-connection WebSocket broker. Fire-and-forget fan-out broadcast."""

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._connections = set()
                    instance._lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def connect(self, ws) -> None:
        """Register a live WebSocket connection."""
        with self._lock:
            self._connections.add(ws)

    def disconnect(self, ws) -> None:
        """Remove a WebSocket connection when it closes.

        Removing a specific socket (rather than clearing the whole registry)
        keeps every other open tab/device receiving broadcasts.
        """
        with self._lock:
            self._connections.discard(ws)

    def broadcast(self, data: dict) -> None:
        """Fire-and-forget push to every live connection. No-op if nobody is listening.

        Sockets that fail to send (closed/half-open) are pruned so a stale
        connection cannot accumulate or block delivery to healthy clients.
        """
        with self._lock:
            targets = list(self._connections)
        if not targets:
            return
        payload = json.dumps(data)
        dead = []
        for ws in targets:
            try:
                ws.send(payload)
            except Exception as exc:
                logger.debug("[WS BROKER] send failed (connection likely closed): %s", exc)
                dead.append(ws)
        if dead:
            with self._lock:
                self._connections.difference_update(dead)
