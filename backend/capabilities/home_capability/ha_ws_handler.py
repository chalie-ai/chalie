"""Persistent WebSocket client for Home Assistant event subscriptions.

Runs in a dedicated daemon thread. Subscribes to state_changed events and
forwards matching entity changes to Redis output:events for the Chalie
WebSocket to pick up.
"""

import json
import logging
import threading

logger = logging.getLogger(__name__)

_RECONNECT_BASE = 5
_RECONNECT_MAX = 60


class HaWebSocketHandler:
    """Manages a persistent WebSocket connection to Home Assistant."""

    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._url: str = ""
        self._token: str = ""
        self._msg_id = 0

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, ws_url: str, token: str) -> None:
        """Start the WebSocket listener thread (idempotent)."""
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
        """Signal the thread to stop and wait for it to exit."""
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

            sock.send(json.dumps({"type": "auth", "access_token": self._token}))
            auth_resp = json.loads(sock.recv())
            if auth_resp.get("type") != "auth_ok":
                raise ConnectionError(f"HA auth failed: {auth_resp.get('message', 'unknown')}")

            sock.send(json.dumps({
                "id": self._next_id(),
                "type": "subscribe_events",
                "event_type": "state_changed",
            }))
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
                    sock.send(json.dumps({"id": self._next_id(), "type": "ping"}))
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

    def _handle_event(self, msg: dict) -> None:
        event = msg.get("event", {})
        data = event.get("data", {})
        entity_id = data.get("entity_id", "")

        with self._lock:
            if entity_id not in self._subscriptions:
                return

        new_state = data.get("new_state", {})
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            payload = {
                "type": "home_state_changed",
                "entity_id": entity_id,
                "state": new_state.get("state"),
                "friendly_name": new_state.get("attributes", {}).get("friendly_name", entity_id),
                "attributes": new_state.get("attributes", {}),
            }
            store.publish("output:events", json.dumps(payload))
        except Exception as exc:
            logger.debug("[ha-ws] failed to publish state change: %s", exc)
