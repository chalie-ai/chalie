import json
import logging
import threading
import uuid
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from pydantic import ValidationError
from simple_websocket import Server as _WS

from api import _deployment_origins
from api.request.browser import (
    BrowserKeyInput,
    BrowserMouseInput,
    BrowserSizeInput,
    BrowserTextInput,
    BrowserWheelInput,
    browser_input,
)
from models.ws_message import WsMessage
from services.websocket import Websocket, _WebSocket
from tools.browser import tabs
from tools.browser.screencast import Viewer, stream
from utils.logger import set_correlation_id

if TYPE_CHECKING:
    from flask import Request
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


# ── Live browser tab ─────────────────────────────────────────────────────────

_CLOSE_FORBIDDEN = 4403    # final: the client stops reconnecting
_CLOSE_UNKNOWN_TAB = 4404
_RECEIVE_POLL = 1.0        # s — how often the input loop checks its sender is still up

_CTRL, _META, _SHIFT = 2, 4, 8
_BUTTONS = {"left": 1, "right": 2, "middle": 4, "none": 0}
# Chrome runs no native editing shortcuts for synthetic keys, so these go as
# editor commands: (key, shift held) → command.
_EDIT_COMMANDS = {("a", False): "selectAll", ("z", False): "undo", ("z", True): "redo", ("y", False): "redo"}


def _allowed_origin(flask_request: "Request") -> bool:
    """Whether the upgrade's ``Origin`` may drive a tab: absent, this server's own host
    (either scheme, as a TLS proxy changes it) or the configured deployment domain.

    The session cookie rides an upgrade from any site's page, so this is what keeps
    another site from watching and steering the user's browser."""
    origin = flask_request.headers.get("Origin")
    if origin is None:
        return True
    origin = origin.lower()
    return (urlsplit(origin).netloc == flask_request.host.lower()
            or origin in (allowed.lower() for allowed in _deployment_origins()))


def _ws_browser_handler(ws: object, tab_id: str) -> None:
    """Stream the tab *tab_id* and forward the viewer's input to it.

    Authenticates like ``/ws``, then refuses an upgrade from another site's
    page. The stream thread is the socket's only writer once it starts; this
    thread only receives, posting each valid input message straight to the
    page without waiting on it.
    """
    from flask import request as flask_request

    ws_typed = cast(_WS, ws)
    if not _validate_cookie_session(flask_request):
        ws_typed.send(json.dumps({"type": "error", "message": "Unauthorized"}))
        ws_typed.close(reason=_CLOSE_FORBIDDEN, message="Unauthorized")
        return
    if not _allowed_origin(flask_request):
        ws_typed.send(json.dumps({"type": "error", "message": "Forbidden origin"}))
        ws_typed.close(reason=_CLOSE_FORBIDDEN, message="Forbidden origin")
        return
    tab = tabs.get(tab_id)
    if tab is None:
        ws_typed.send(json.dumps({"type": "closed"}))
        ws_typed.close(reason=_CLOSE_UNKNOWN_TAB, message="No such tab")
        return

    viewer = Viewer(tab)
    viewer.open()
    sender = threading.Thread(target=stream, args=(ws_typed, viewer), name=f"browser-stream-{tab_id[:8]}",
                              daemon=True)
    sender.start()
    try:
        while sender.is_alive():
            raw = ws_typed.receive(timeout=_RECEIVE_POLL)
            if raw is not None:
                _forward(viewer, raw)
    finally:
        viewer.close()
        sender.join()


def _forward(viewer: Viewer, raw: str | bytes) -> None:
    """Post one input message to the viewer's page; a malformed one is logged and dropped."""
    try:
        message = browser_input.validate_json(raw)
    except ValidationError as e:
        # Never the raw input: a dropped key or text message may carry what the user typed.
        details = "; ".join(f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                            for err in e.errors(include_url=False, include_input=False))
        logger.warning("[BROWSER] dropped a malformed input message for tab %s: %s", viewer.tab.id, details)
        return
    if isinstance(message, BrowserSizeInput):
        viewer.resize(int(message.width), int(message.height))
        return
    viewer.input(*_cdp_input(message))


def _cdp_input(message: BrowserMouseInput | BrowserWheelInput | BrowserKeyInput | BrowserTextInput,
               ) -> tuple[str, dict[str, object]]:
    """The ``Input`` command for one viewer input message."""
    if isinstance(message, BrowserMouseInput):
        return "Input.dispatchMouseEvent", {
            "type": message.event, "x": message.x, "y": message.y, "button": message.button,
            "buttons": 0 if message.event == "mouseReleased" else _BUTTONS[message.button],
            "clickCount": message.click_count, "modifiers": message.modifiers,
        }
    if isinstance(message, BrowserKeyInput):
        return "Input.dispatchKeyEvent", _key_event(message)
    if isinstance(message, BrowserTextInput):
        return "Input.insertText", {"text": message.text}
    return "Input.dispatchMouseEvent", {
        "type": "mouseWheel", "x": message.x, "y": message.y, "deltaX": message.delta_x,
        "deltaY": message.delta_y, "modifiers": message.modifiers,
    }


def _key_event(message: BrowserKeyInput) -> dict[str, object]:
    params: dict[str, object] = {"key": message.key, "code": message.code,
                                 "windowsVirtualKeyCode": message.key_code, "modifiers": message.modifiers}
    if message.event == "keyUp":
        return {"type": "keyUp", **params}
    command = None
    if message.modifiers & (_CTRL | _META):
        command = _EDIT_COMMANDS.get((message.key.lower(), bool(message.modifiers & _SHIFT)))
    if command is not None:
        return {"type": "rawKeyDown", **params, "commands": [command]}
    if message.text:
        return {"type": "keyDown", **params, "text": message.text}
    return {"type": "rawKeyDown", **params}


def register_websocket(sock: "Sock") -> None:
    def ws_handler(ws: object) -> None:
        _ws_handler(ws)

    def ws_browser_handler(ws: object, tab_id: str) -> None:
        _ws_browser_handler(ws, tab_id)

    sock.route('/ws')(ws_handler)
    sock.route('/ws/browser/<tab_id>')(ws_browser_handler)
