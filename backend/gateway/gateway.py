"""
Gateway proxy routes — the tunnel between interface daemons and Chalie.

Daemons call these endpoints (no auth headers). The interface_id in the URL
path identifies the caller. The dashboard validates scopes and forwards
allowed requests to Chalie's backend API using its own wrapper token.

Routes:
    POST /register                   — daemon self-registers on startup
    POST /gw/<interface_id>/signals  — push signal to world state
    POST /gw/<interface_id>/messages — push message to reasoning loop
    GET  /gw/<interface_id>/context  — get filtered user context
    GET/PUT  /gw/<interface_id>/meta — daemon metadata (configs, API keys, etc.)
    CRUD     /gw/<interface_id>/data — daemon file storage
"""

import logging
import os
import threading
import time
import uuid

import requests

from flask import Blueprint, request, jsonify, send_file

logger = logging.getLogger(__name__)

_CONTENT_TYPE_JSON = "application/json"
_ERR_INVALID_FILENAME = "Invalid filename"

gateway_bp = Blueprint("gateway", __name__)

# Set by app.py at startup
_chalie_client = None
_dashboard_port = 3000
_data_dir = ""
_DISCOVER_TIMEOUT = 5


def init_gateway(chalie_client, dashboard_port: int, data_dir: str = "") -> None:
    """Inject the ChalieClient instance, dashboard port, and data directory."""
    global _chalie_client, _dashboard_port, _data_dir
    _chalie_client = chalie_client
    _dashboard_port = dashboard_port
    _data_dir = data_dir


def _deferred_chalie_register(dashboard_iface_id: str, name: str, host: str, port: int,
                               chalie_iface_id: str | None = None, delay: float = 3.0):
    """Register or refresh an interface with Chalie after a short delay.

    Runs in a background daemon thread. The delay gives the daemon time to
    start listening on its port before Chalie tries to health-check it.
    """
    from . import db

    def _run():
        time.sleep(delay)
        try:
            if chalie_iface_id:
                _chalie_client.refresh_capabilities(chalie_iface_id)
                logger.info("[Gateway] Refreshed capabilities for '%s'", name)
            else:
                result = _chalie_client.register_interface(name, host, port)
                new_id = result.get("interface_id")
                if new_id:
                    db.update_interface(dashboard_iface_id, chalie_interface_id=new_id)
                    logger.info("[Gateway] Chalie registration succeeded for '%s' → %s", name, new_id)
                else:
                    logger.warning("[Gateway] Chalie registration failed for '%s': %s",
                                   name, result.get("error"))
        except Exception as e:
            logger.warning("[Gateway] Chalie registration failed for '%s': %s", name, e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _daemon_data_dir(interface_id: str) -> str:
    """Return the data directory for a daemon, creating it if needed."""
    path = os.path.join(_data_dir, "interfaces", interface_id)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(name: str) -> str | None:
    """Sanitise a filename — reject path traversal attempts."""
    name = name.strip().strip("/\\")
    if not name or "/" in name or "\\" in name or name.startswith(".."):
        return None
    if name.startswith("."):
        return None
    return name


def _safe_data_path(interface_id: str, filename: str) -> str | None:
    """Build a resolved path and verify it stays inside the daemon data dir."""
    base = os.path.realpath(_daemon_data_dir(interface_id))
    candidate = os.path.realpath(os.path.join(base, filename))
    if not candidate.startswith(base + os.sep) and candidate != base:
        return None
    return candidate


def _get_interface(interface_id: str):
    """Look up an installed interface. Returns (dict, None) or (None, response)."""
    from . import db
    iface = db.get_interface(interface_id)
    if not iface:
        return None, (jsonify({"error": "Unknown interface"}), 404)
    return iface, None


def _check_scope(approved_scopes: dict, category: str, key: str) -> bool:
    """Check if a scope key is approved for the given category."""
    allowed = approved_scopes.get(category, {})
    if isinstance(allowed, dict):
        return key in allowed
    if isinstance(allowed, list):
        return key in allowed
    return False


def _is_scope_approved(iface: dict) -> bool:
    """Check if the interface has any approved scopes (i.e. user accepted it)."""
    scopes = iface.get("approved_scopes", {})
    if not scopes:
        return False
    # At least one category must have entries
    for cat in ("context", "signals", "messages"):
        if scopes.get(cat):
            return True
    return False


# ── Registration ───────────────────────────────────────────────────────────

@gateway_bp.route("/register", methods=["POST"])
def register_daemon():
    """Auto-registration endpoint called by daemon SDK on startup.

    The daemon sends its meta (name, version, scopes, etc.) and its port.
    The dashboard creates an interface record and returns the gateway URL
    the daemon should use for all subsequent calls.

    Body: {name, version, description, author, scopes, host, port}
    Returns: {gateway_url, interface_id, status}
    """
    from . import db

    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    port = body.get("port")
    # Daemons always run locally (same machine / same container).
    # request.remote_addr can resolve to ::1, ::ffff:127.0.0.1, or
    # 172.x.x.x inside Docker — normalize to 127.0.0.1.
    host = "127.0.0.1"

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not port:
        return jsonify({"error": "port is required"}), 400

    try:
        port = int(port)
    except (TypeError, ValueError):
        return jsonify({"error": "port must be a number"}), 400

    # Check if this daemon is already registered (by host:port)
    existing = db.list_interfaces()
    for iface in existing:
        if iface["host"] == str(host) and iface["port"] == port:
            # Already registered — return existing gateway URL
            gateway_url = f"http://localhost:{_dashboard_port}/gw/{iface['id']}"
            logger.info(
                "[Gateway] Daemon '%s' at %s:%d re-registered (already known as %s)",
                name, host, port, iface["id"],
            )
            # Update meta in case it changed
            db.update_interface(iface["id"],
                name=name,
                version=body.get("version"),
                description=body.get("description"),
                author=body.get("author"),
                requested_scopes=body.get("scopes", {}),
                status="online",
            )

            # Ensure Chalie knows about this interface's tools.
            # Deferred: the daemon may not be listening on its port yet.
            chalie_iface_id = iface.get("chalie_interface_id")
            if _chalie_client:
                _deferred_chalie_register(
                    iface["id"], name, str(host), port, chalie_iface_id,
                )

            return jsonify({
                "gateway_url": gateway_url,
                "interface_id": iface["id"],
                "status": iface.get("status", "pending"),
            }), 200

    # New daemon — create interface record
    interface_id = str(uuid.uuid4())
    from services.time_utils import utc_now
    now = utc_now().isoformat()

    scopes = body.get("scopes", {})

    # Store in dashboard DB — scopes NOT approved yet (pending user approval)
    db.insert_interface({
        "id": interface_id,
        "name": name,
        "host": str(host),
        "port": port,
        "status": "pending",
        "version": body.get("version"),
        "description": body.get("description"),
        "author": body.get("author"),
        "requested_scopes": scopes,
        "approved_scopes": {},  # Empty until user approves
        "chalie_interface_id": None,
        "paired_at": now,
        "last_seen_at": now,
    })

    # Register with Chalie so tools appear in ACT loop.
    # Deferred: the daemon hasn't started listening on its port yet.
    if _chalie_client:
        _deferred_chalie_register(interface_id, name, str(host), port)

    # Create daemon's data folder
    _daemon_data_dir(interface_id)

    gateway_url = f"http://localhost:{_dashboard_port}/gw/{interface_id}"

    logger.info(
        "[Gateway] New daemon registered: '%s' at %s:%d → %s (pending scope approval)",
        name, host, port, gateway_url,
    )

    return jsonify({
        "gateway_url": gateway_url,
        "interface_id": interface_id,
        "status": "pending",
    }), 201


# ── Signals ────────────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/signals", methods=["POST"])
def proxy_signal(interface_id: str):
    """Forward a signal to Chalie's world state after scope validation."""
    iface, err = _get_interface(interface_id)
    if err:
        return err

    if not _is_scope_approved(iface):
        return jsonify({"error": "Scopes not yet approved by user"}), 403

    body = request.get_json(silent=True) or {}
    signal_type = (body.get("signal_type") or "").strip()
    if not signal_type:
        return jsonify({"error": "signal_type is required"}), 400

    content = body.get("content")
    if not content:
        return jsonify({"error": "content is required"}), 400

    # Scope check
    if not _check_scope(iface["approved_scopes"], "signals", signal_type):
        logger.info(
            "[Gateway] Signal '%s' denied for '%s' — not in approved scopes",
            signal_type, iface["name"],
        )
        return jsonify({"error": f"Signal type '{signal_type}' not approved"}), 403

    # Forward to Chalie
    result = _chalie_client.push_signal(
        signal_type=signal_type,
        content=str(content),
        source=f"interface:{iface['name']}",
        activation_energy=body.get("activation_energy", 0.5),
        metadata=body.get("metadata"),
    )

    if "error" in result and result["error"]:
        return jsonify(result), 502
    return jsonify({"ok": True}), 202


# ── Messages ───────────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/messages", methods=["POST"])
def proxy_message(interface_id: str):
    """Forward a message to Chalie's reasoning loop after scope validation."""
    iface, err = _get_interface(interface_id)
    if err:
        return err

    if not _is_scope_approved(iface):
        return jsonify({"error": "Scopes not yet approved by user"}), 403

    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    topic = body.get("topic")

    # Scope check
    approved_messages = iface["approved_scopes"].get("messages", {})
    if not approved_messages:
        return jsonify({"error": "No message scopes approved"}), 403

    if topic and not _check_scope(iface["approved_scopes"], "messages", topic):
        return jsonify({"error": f"Message topic '{topic}' not approved"}), 403

    # Inject into Chalie's reasoning loop — Chalie will process and respond in chat
    try:
        from services.memory_client import MemoryClientService
        from services.reasoning_loop_service import ReasoningSignal

        signal = ReasoningSignal(
            signal_type='user_message',
            source=f"interface:{iface['name']}",
            content=f"[{iface['name']}] {text}",
            activation_energy=0.9,
            metadata={
                "interface_id": interface_id,
                "topic": topic,
                "source": "interface",
                **(body.get("metadata") or {}),
            },
            wrapper_id='__chat_ui__',
        )
        store = MemoryClientService.create_connection()
        store.rpush('reasoning:priority', signal.to_json())
    except Exception as e:
        logger.error("[Gateway] Failed to inject message into reasoning loop: %s", e)
        return jsonify({"error": str(e)}), 502

    return jsonify({"ok": True}), 202


# ── Context ────────────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/context", methods=["GET"])
def proxy_context(interface_id: str):
    """Return user context filtered by approved scopes."""
    iface, err = _get_interface(interface_id)
    if err:
        return err

    if not _is_scope_approved(iface):
        return jsonify({}), 200

    # Fetch full context from Chalie
    full_ctx = _chalie_client.get_context()
    if not full_ctx:
        return jsonify({}), 200

    # Filter by approved context scopes
    approved_context = iface["approved_scopes"].get("context", {})
    if isinstance(approved_context, dict):
        approved_keys = set(approved_context.keys())
    elif isinstance(approved_context, list):
        approved_keys = set(approved_context)
    else:
        approved_keys = set()

    filtered = {k: v for k, v in full_ctx.items() if k in approved_keys}
    return jsonify(filtered), 200


# ── Metadata ──────────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/meta", methods=["GET"])
def get_meta(interface_id: str):
    """Return the daemon's metadata object."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    from . import db
    return jsonify(db.get_meta(interface_id)), 200


@gateway_bp.route("/gw/<interface_id>/meta", methods=["PUT"])
def put_meta(interface_id: str):
    """Replace the daemon's entire metadata object."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400
    from . import db
    db.set_meta(interface_id, body)
    return jsonify({"ok": True}), 200


@gateway_bp.route("/gw/<interface_id>/meta", methods=["PATCH"])
def patch_meta(interface_id: str):
    """Merge updates into the daemon's metadata object."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400
    from . import db
    result = db.patch_meta(interface_id, body)
    return jsonify(result), 200


# ── Render proxy ──────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/render", methods=["GET"])
def proxy_render(interface_id: str):
    """Proxy the daemon's renderInterface() output.

    SDK v2 daemons return JSON blocks (Content-Type: application/json).
    Legacy SDK v1 daemons return HTML (Content-Type: text/html).
    The gateway detects the response type and handles each accordingly.
    """
    iface, err = _get_interface(interface_id)
    if err:
        return err
    try:
        resp = requests.get(
            f"http://{iface['host']}:{iface['port']}/",
            timeout=_DISCOVER_TIMEOUT,
        )
        if not resp.ok:
            return jsonify({"error": f"Daemon returned {resp.status_code}"}), 502

        content_type = resp.headers.get("Content-Type", "")
        gw_base = f"/gw/{interface_id}"

        if _CONTENT_TYPE_JSON in content_type:
            # SDK v2: blocks response — inject gateway path for action wiring
            data = resp.json()
            data["gateway"] = gw_base
            return jsonify(data), 200
        else:
            # SDK v1 legacy: raw HTML — inject CHALIE_GW_BASE global
            script = f'<script>window.CHALIE_GW_BASE="{gw_base}";</script>'
            html = script + resp.text
            return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        return jsonify({"error": f"Daemon unreachable: {e}"}), 502


# ── Execute proxy ─────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/execute", methods=["POST"])
def proxy_execute(interface_id: str):
    """Proxy a capability execution request to the daemon."""
    iface, err = _get_interface(interface_id)
    if err:
        return err
    body = request.get_data()
    try:
        resp = requests.post(
            f"http://{iface['host']}:{iface['port']}/execute",
            data=body,
            headers={"Content-Type": _CONTENT_TYPE_JSON},
            timeout=_DISCOVER_TIMEOUT,
        )
        return resp.text, resp.status_code, {"Content-Type": _CONTENT_TYPE_JSON}
    except Exception as e:
        return jsonify({"error": f"Daemon unreachable: {e}"}), 502


# ── Data files ────────────────────────────────────────────────────────────

@gateway_bp.route("/gw/<interface_id>/data", methods=["GET"])
def list_data_files(interface_id: str):
    """List all files in the daemon's data folder."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    data_path = _daemon_data_dir(interface_id)
    files = []
    for name in os.listdir(data_path):
        fpath = os.path.join(data_path, name)
        if os.path.isfile(fpath):
            files.append({
                "name": name,
                "size": os.path.getsize(fpath),
            })
    return jsonify(files), 200


@gateway_bp.route("/gw/<interface_id>/data/<filename>", methods=["GET"])
def get_data_file(interface_id: str, filename: str):
    """Read a file from the daemon's data folder."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    safe = _safe_filename(filename)
    if not safe:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    fpath = _safe_data_path(interface_id, safe)
    if not fpath:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    if not os.path.isfile(fpath):
        return jsonify({"error": "File not found"}), 404
    return send_file(fpath, as_attachment=False)


@gateway_bp.route("/gw/<interface_id>/data/<filename>", methods=["PUT"])
def put_data_file(interface_id: str, filename: str):
    """Create or overwrite a file in the daemon's data folder."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    safe = _safe_filename(filename)
    if not safe:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    fpath = _safe_data_path(interface_id, safe)
    if not fpath:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    data = request.get_data()
    with open(fpath, "wb") as f:
        f.write(data)
    return jsonify({"ok": True, "name": safe, "size": len(data)}), 200


@gateway_bp.route("/gw/<interface_id>/data/<filename>", methods=["DELETE"])
def delete_data_file(interface_id: str, filename: str):
    """Delete a file from the daemon's data folder."""
    _, err = _get_interface(interface_id)
    if err:
        return err
    safe = _safe_filename(filename)
    if not safe:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    fpath = _safe_data_path(interface_id, safe)
    if not fpath:
        return jsonify({"error": _ERR_INVALID_FILENAME}), 400
    if not os.path.isfile(fpath):
        return jsonify({"error": "File not found"}), 404
    os.remove(fpath)
    return jsonify({"ok": True}), 200
