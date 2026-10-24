import json
import logging
import uuid
from typing import TYPE_CHECKING, cast

from simple_websocket import Server as _WS

from models.ws_message import WsMessage
from services.websocket import Websocket, _WebSocket
from utils.logger import set_correlation_id

if TYPE_CHECKING:
    from flask_sock import Sock

logger = logging.getLogger(__name__)


def _validate_cookie_session(flask_request: object) -> bool:
    """The browser path: a signed session cookie rides the HTTP upgrade."""
    from flask import Request
    from services.auth_session_service import validate_session

    return validate_session(cast(Request, flask_request))


def _ws_handler(ws: object) -> None:
    from flask import request as flask_request

    ws_typed = cast(_WS, ws)

    # The signed session cookie rides the HTTP upgrade — checked at handshake
    # time, before the socket is registered with the broker.
    if not _validate_cookie_session(flask_request):
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

    socket = cast(_WebSocket, ws)
    Websocket._connect(socket)

    try:
        while True:
            raw = ws_typed.receive(timeout=60)
            if raw is None:
                Websocket.broadcast(WsMessage(type="ping"))
                continue
            # Only expected client message is pong — everything else is ignored.
    except Exception as exc:
        logger.debug("[WS] Connection closed: %s", exc)
    finally:
        Websocket._disconnect(socket)


def register_websocket(sock: "Sock") -> None:
    def ws_handler(ws: object) -> None:
        _ws_handler(ws)

    sock.route('/ws')(ws_handler)
