"""
WebSocket endpoint — server→client push channel.

All client→server requests use HTTP (POST /chat, POST /action, POST /upload).
The WebSocket is receive-only from the client's perspective. The server sends:
  ← {"type": "status", "stage": "..."}
  ← {"type": "message", "content": "...", ...}
  ← {"type": "act_narration", "text": "...", "step": N}
  ← {"type": "done", "duration_ms": N}
  ← {"type": "drift|task|reminder|escalation|notification", ...}
  ← {"type": "permission_request", ...}
  ← {"type": "ping"}

The only client→server message the server expects is {"type": "pong"}.
Any other client message is silently ignored.
"""

import json
import logging
import uuid

from utils.logger import set_correlation_id
from services.websocket_broker import WebSocketBroker

logger = logging.getLogger(__name__)


def _drain_capability_alerts(store) -> None:
    """Replay persisted capability alert keys via broker and delete them after delivery."""
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


def _ws_handler(ws) -> None:
    """WebSocket connection lifecycle: auth → register → keep-alive loop.

    The receive loop exists solely for keep-alive: on receive timeout (60s)
    the server sends a ping; the client responds with pong. All other
    client→server traffic uses HTTP endpoints.
    """
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
        broker.disconnect(ws)


def register_websocket(sock):
    """Register the /ws endpoint on a flask-sock instance."""

    @sock.route('/ws')
    def ws_handler(ws):
        _ws_handler(ws)
