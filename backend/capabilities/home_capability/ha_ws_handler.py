"""Persistent WebSocket client for Home Assistant.

A self-service, Home-Assistant-specific handler. It owns the persistent WS
connection to one HA server: authenticates, subscribes to ``state_changed``
events, keeps the link alive with pings, reconnects with backoff, and parses
incoming HA frames. It has NO relation to the Chalie frontend
:class:`~services.websocket.Websocket` facade — it does not touch the frontend
and is not usable by any other part of the Chalie codebase. The home capability
that owns it registers an ``on_event`` callback and decides what to do with
each parsed state change (including bridging it to the frontend, from the
capability, not from here).
"""

import json
import logging
import threading
from collections.abc import Callable
from typing import Protocol, cast

logger = logging.getLogger(__name__)

_RECONNECT_BASE = 5
_RECONNECT_MAX = 60

#: Signature of the callback the owning capability registers: handed the
#: entity id and the parsed ``new_state`` dict for every subscribed state change.
OnHaEvent = Callable[[str, "dict[str, object]"], None]


class _WsConn(Protocol):
    """Minimal live-socket surface :meth:`HaWebSocketHandler._send` writes to."""

    def send(self, payload: str) -> object: ...


class HaWebSocketHandler:
    """Manages a persistent WebSocket connection to one Home Assistant server."""

    def __init__(self, on_event: OnHaEvent | None = None) -> None:
        self._subscriptions: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._url: str = ""
        self._token: str = ""
        self._msg_id = 0
        self._on_event = on_event

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, ws_url: str, token: str) -> None:
        if self.is_alive:
            return
        self._url = ws_url
        self._token = token
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ha-ws-listener",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._connected = False

    def subscribe(self, entity_id: str) -> None:
        with self._lock:
            self._subscriptions.add(entity_id)

    def unsubscribe(self, entity_id: str) -> None:
        with self._lock:
            self._subscriptions.discard(entity_id)

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, sock: _WsConn, payload: dict[str, object]) -> None:
        """Write one HA command frame to the live socket."""
        sock.send(json.dumps(payload))

    def _run_loop(self) -> None:
        """Reconnect loop -- runs until stop is signalled."""
        backoff = _RECONNECT_BASE
        while not self._stop.is_set():
            try:
                self._connect_and_listen()
                backoff = _RECONNECT_BASE
            except Exception as exc:
                logger.warning("[ha-ws] connection error: %s -- retrying in %ds", exc, backoff)
                self._connected = False
                if self._stop.wait(timeout=backoff):
                    break
                backoff = min(backoff * 2, _RECONNECT_MAX)

    def _connect_and_listen(self) -> None:
        import websocket as ws_lib  # websocket-client; imported lazily to avoid hard dep at boot

        sock = ws_lib.create_connection(self._url, timeout=10)
        try:
            auth_req = json.loads(sock.recv())
            if auth_req.get("type") != "auth_required":
                raise ConnectionError(f"Unexpected HA message: {auth_req.get('type')}")

            self._send(sock, {"type": "auth", "access_token": self._token})
            auth_resp = json.loads(sock.recv())
            if auth_resp.get("type") != "auth_ok":
                raise ConnectionError(f"HA auth failed: {auth_resp.get('message', 'unknown')}")

            self._send(sock, {
                "id": self._next_id(),
                "type": "subscribe_events",
                "event_type": "state_changed",
            })
            sub_resp = json.loads(sock.recv())
            if not sub_resp.get("success"):
                raise ConnectionError("Failed to subscribe to state_changed events")

            self._connected = True
            logger.info("[ha-ws] connected and subscribed to state_changed")

            sock.settimeout(30)
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except ws_lib.WebSocketTimeoutException:
                    self._send(sock, {"id": self._next_id(), "type": "ping"})
                    continue

                msg = json.loads(raw)
                if msg.get("type") == "event":
                    self._handle_event(msg)

        finally:
            self._connected = False
            try:
                sock.close()
            except Exception:
                pass

    def _handle_event(self, msg: dict[str, object]) -> None:
        """Parse one HA ``state_changed`` frame and hand it to the registered
        callback. Does nothing when no callback is set or the entity is not
        subscribed."""
        if self._on_event is None:
            return
        event = cast("dict[str, object]", msg.get("event", {}))
        data = cast("dict[str, object]", event.get("data", {}))
        entity_id = cast(str, data.get("entity_id", ""))

        with self._lock:
            if entity_id not in self._subscriptions:
                return

        new_state = cast("dict[str, object]", data.get("new_state", {}))
        try:
            self._on_event(entity_id, new_state)
        except Exception as exc:
            logger.debug("[ha-ws] on_event callback raised: %s", exc)
