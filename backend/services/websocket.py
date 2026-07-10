"""Websocket — the single fire-and-forget fan-out facade for the Chalie
frontend channel.

Chalie is single-user, but that one user may have the chat UI open on several
tabs/devices at once. This class encapsulates the ENTIRE websocket channel: the
live-connection registry, the connect/disconnect lifecycle, and the one public
broadcast operation. Everything connection-related lives here, in one class, as
private members — business code never touches connection state; it simply does
``Websocket.broadcast(instance)`` and is done.

The API handshake handler (:mod:`api.websocket`) is the sole caller of the
private ``_connect`` / ``_disconnect`` lifecycle hooks (it owns the socket
handshake and ping loop — application concerns that stay out of this class).
Everywhere else reaches only for :meth:`broadcast`.

There are no reply channels, no feedback loops, no raw dicts on the wire —
every frame is a :class:`~contracts.json_serializable.JsonSerializable`
instance serialized once via ``.to_json()``. Sockets that fail to send
(closed/half-open) are pruned so a stale connection cannot accumulate or block
delivery to healthy clients.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Protocol

from contracts.json_serializable import JsonSerializable

logger = logging.getLogger(__name__)


class _WebSocket(Protocol):
    """Structural type for a WebSocket connection that can send a string."""

    def send(self, data: str) -> None: ...


class Websocket:
    """The Chalie frontend websocket channel — registry, lifecycle, broadcast.

    All connection state and lifecycle is private to this class; ``broadcast``
    is the one public operation. Process-global by construction (the registry
    is a class attribute), so a broadcast from any turn thread reaches every
    open tab/device.
    """

    _connections: set[_WebSocket] = set()
    _lock = threading.Lock()

    @classmethod
    def _connect(cls, ws: _WebSocket) -> None:
        """Register a live socket (called only by the API handshake handler)."""
        with cls._lock:
            cls._connections.add(ws)

    @classmethod
    def _disconnect(cls, ws: _WebSocket) -> None:
        """Drop a socket when it closes, leaving every other open tab/device
        receiving broadcasts (called only by the API handshake handler)."""
        with cls._lock:
            cls._connections.discard(ws)

    @staticmethod
    def broadcast(instance: JsonSerializable | None) -> None:
        """Serialize ``instance`` once via ``.to_json()`` and push the string to
        every live socket. ``None`` is a no-op — a caller may hand a nullable
        model (e.g. one forwarded through ``mp.push_websocket``) without an extra
        guard at the call site. No-op too if nobody is listening. Sockets that fail to send
        (closed/half-open) are pruned so a stale connection cannot accumulate
        or block delivery to healthy clients.

        Turn-scoped frames are enriched with the shared DOM-contract keys
        ``turn_id`` / ``type`` / ``transcript_row_id`` here and only here —
        models never hand-write these keys into their own WS serialization."""
        if instance is None:
            return
        with Websocket._lock:
            targets = list(Websocket._connections)
        if not targets:
            return
        payload_dict = json.loads(instance.to_json())
        if getattr(instance, "turn_id", None) is not None:
            payload_dict.setdefault("turn_id", instance.turn_id)
            type_val = getattr(instance, "type", None)
            if type_val is not None:
                payload_dict.setdefault("type", type_val)
            row_ref = getattr(instance, "transcript_id", None)
            if row_ref is not None:
                payload_dict.setdefault("transcript_row_id", row_ref)
        payload = json.dumps(payload_dict)
        dead: list[_WebSocket] = []
        for ws in targets:
            try:
                ws.send(payload)
            except Exception as exc:  # noqa: BLE001 — a dead surface is pruned, not raised
                logger.debug("[WS] send failed (connection likely closed): %s", exc)
                dead.append(ws)
        if dead:
            with Websocket._lock:
                Websocket._connections.difference_update(dead)
