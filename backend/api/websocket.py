import json
import logging
import uuid
from typing import TYPE_CHECKING

from utils.logger import set_correlation_id
from services.memory_store import MemoryStore
from services.websocket_broker import WebSocketBroker

if TYPE_CHECKING:
    from typing import Callable, Protocol, TypeVar

    _F = TypeVar("_F")

    class _WebSocket(Protocol):
        def send(self, data: str) -> None: ...
        def close(self) -> None: ...
        def receive(self, timeout: float = ...) -> "str | None": ...

    class Sock(Protocol):
        def route(self, rule: str) -> "Callable[[_F], _F]": ...

logger = logging.getLogger(__name__)


def _drain_capability_alerts(store: MemoryStore) -> None:
    try:
        broker = WebSocketBroker()
        alert_keys = store.keys('capability:alert:*')
        for key in alert_keys:
            raw = store.get(key)
            if raw:
                try:
                    data = json.loads(raw)
                    broker.broadcast(data)
                    store.delete(key)
                except Exception as exc:
                    logger.debug("[WS] Failed to send persisted capability alert: %s", exc)
    except Exception as exc:
        logger.debug("[WS] Failed to scan capability alerts: %s", exc)


def _ws_handler(ws: "_WebSocket") -> None:
    from flask import request as flask_request
    from services.auth_session_service import validate_session

    broker = WebSocketBroker()

    if not validate_session(flask_request):
        try:
            ws.send(json.dumps({"type": "error", "message": "Unauthorized"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception as exc:
            logger.debug("[WS] Close after auth failure failed: %s", exc)
        return

    connection_id = str(uuid.uuid4())
    set_correlation_id(connection_id)
    logger.debug("[WS] Connection established", extra={"connection_id": connection_id})

    broker.connect(ws)

    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    _drain_capability_alerts(store)

    try:
        while True:
            raw = ws.receive(timeout=60)
            if raw is None:
                broker.broadcast({"type": "ping"})
                continue
            # Only expected client message is pong — everything else is ignored.
    except Exception as exc:
        logger.debug("[WS] Connection closed: %s", exc)
    finally:
        broker.disconnect()


def register_websocket(sock: "Sock") -> None:
    @sock.route('/ws')
    def ws_handler(ws: "_WebSocket") -> None:
        _ws_handler(ws)
