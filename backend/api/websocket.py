import json
import logging
import uuid
from typing import TYPE_CHECKING, cast

from simple_websocket import Server as _WS
from utils.logger import set_correlation_id
from services.memory_store import MemoryStore
from services.websocket_broker import WebSocketBroker, _WebSocket

if TYPE_CHECKING:
    from flask_sock import Sock

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


def _authenticate_ws(flask_request: object) -> bool:
    """Authenticate a ``/ws`` handshake.

    Cookie session first (the browser path, unchanged). If that fails, fall back
    to the bearer token supplied as the ``?token=`` query arg — the mobile app
    cannot send a cookie over a WebSocket, so it passes the raw wrapper token in
    the URL. ``validate_bearer`` expects an ``Authorization: Bearer`` header, so
    the query token is wrapped into that header form before reuse. A missing or
    invalid token yields no wrapper id => caller closes the handshake unchanged.
    """
    from flask import Request
    from flask.wrappers import Request as FlaskRequest
    from services.auth_session_service import validate_session

    if validate_session(cast("Request", flask_request)):
        return True

    token = cast("Request", flask_request).args.get("token", "")
    if not token:
        return False

    from services.wrapper_auth_service import WrapperAuthService
    from services.database_service import get_shared_db_service

    bearer_request = cast(FlaskRequest, FlaskRequest.from_values(
        headers={"Authorization": "Bearer " + token}
    ))
    try:
        return bool(WrapperAuthService(get_shared_db_service()).validate_bearer(bearer_request))
    except Exception as exc:
        logger.debug("[WS] Bearer token validation failed: %s", exc)
        return False


def _ws_handler(ws: object) -> None:
    from flask import request as flask_request

    broker = WebSocketBroker()

    if not _authenticate_ws(flask_request):
        try:
            cast(_WS, ws).send(json.dumps({"type": "error", "message": "Unauthorized"}))
        except Exception:
            pass
        try:
            cast(_WS, ws).close()
        except Exception as exc:
            logger.debug("[WS] Close after auth failure failed: %s", exc)
        return

    connection_id = str(uuid.uuid4())
    set_correlation_id(connection_id)
    logger.debug("[WS] Connection established", extra={"connection_id": connection_id})

    broker.connect(cast(_WebSocket, ws))

    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    _drain_capability_alerts(store)

    ws_typed = cast(_WS, ws)
    try:
        while True:
            raw = ws_typed.receive(timeout=60)
            if raw is None:
                broker.broadcast({"type": "ping"})
                continue
            # Only expected client message is pong — everything else is ignored.
    except Exception as exc:
        logger.debug("[WS] Connection closed: %s", exc)
    finally:
        broker.disconnect(cast(_WebSocket, ws))


def register_websocket(sock: "Sock") -> None:
    def ws_handler(ws: object) -> None:
        _ws_handler(ws)

    sock.route('/ws')(ws_handler)
