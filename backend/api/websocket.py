import json
import logging
import uuid
from typing import TYPE_CHECKING, cast

from simple_websocket import Server as _WS

from services.memory_store import MemoryStore
from services.websocket_broker import WebSocketBroker, _WebSocket
from utils.logger import set_correlation_id

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


def _validate_cookie_session(flask_request: object) -> bool:
    """The browser path: a signed session cookie rides the HTTP upgrade."""
    from flask import Request
    from services.auth_session_service import validate_session

    return validate_session(cast(Request, flask_request))


def _validate_bearer_frame(token: str) -> bool:
    """The native-client path: a raw wrapper token arrives as the first WS
    frame (``{"type":"auth","token":...}``) instead of in the URL, so it never
    leaks into reverse-proxy/access logs or ``Referer``. ``validate_bearer``
    expects an ``Authorization: Bearer`` header, so the frame token is wrapped
    into that header form before reuse."""
    from flask import Request
    from services.wrapper_auth_service import WrapperAuthService
    from services.database_service import get_shared_db_service

    if not token:
        return False

    bearer_request = cast(Request, Request.from_values(
        headers={"Authorization": "Bearer " + token}
    ))
    try:
        return bool(WrapperAuthService(get_shared_db_service()).validate_bearer(bearer_request))
    except Exception as exc:
        logger.debug("[WS] Bearer token validation failed: %s", exc)
        return False


def _ws_handler(ws: object) -> None:
    from flask import request as flask_request
    from services.feature_flags import internal_dev_enabled

    broker = WebSocketBroker()
    ws_typed = cast(_WS, ws)

    # Cookie session first (the browser path) — checked at handshake time, as
    # before. A native client has no cookie, so it authenticates with the first
    # WS frame instead of a query-string token (see ``_handshake_bearer``).
    if not _validate_cookie_session(flask_request):
        if not internal_dev_enabled() or not _handshake_bearer(ws_typed):
            try:
                ws_typed.send(json.dumps({"type": "error", "message": "Unauthorized"}))
            except Exception:
                pass
            try:
                ws_typed.close()
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


def _handshake_bearer(ws: _WS) -> bool:
    """Native-client auth: wait for the first WS frame, expect
    ``{"type":"auth","token":"<raw>"}``, validate the bearer, and close the
    handshake on any other outcome. The token travels in a frame, never in the
    URL, so it cannot leak into reverse-proxy/access logs or ``Referer``."""
    try:
        raw = ws.receive(timeout=10)
    except Exception as exc:
        logger.debug("[WS] Auth frame receive failed: %s", exc)
        return False
    if raw is None:
        return False
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict) or payload.get("type") != "auth":
        return False
    token = payload.get("token")
    if not isinstance(token, str):
        return False
    return _validate_bearer_frame(token)


def register_websocket(sock: "Sock") -> None:
    def ws_handler(ws: object) -> None:
        _ws_handler(ws)

    sock.route('/ws')(ws_handler)
