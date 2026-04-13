"""
REST API package — Flask app factory with Blueprint registration, WebSocket,
and static file serving (replaces nginx).
"""

import os
import logging
from pathlib import Path
from flask import Flask, redirect, send_from_directory
from flask_cors import CORS

from .auth import require_session as require_session


logger = logging.getLogger(__name__)

# Resolve frontend directories relative to backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _BACKEND_DIR.parent / 'frontend'
_INTERFACE_DIR = _FRONTEND_DIR / 'interface'
_BRAIN_DIR = _FRONTEND_DIR / 'brain'
_ONBOARDING_DIR = _FRONTEND_DIR / 'on-boarding'
_LOGIN_DIR = _FRONTEND_DIR / 'login'
_SHARED_DIR = _FRONTEND_DIR / 'shared'


def _get_or_generate_session_secret() -> str:
    """Return the Flask session signing secret.

    Priority:
    1. SESSION_SECRET_KEY environment variable (for multi-instance or reverse-proxy setups)
    2. Persisted value in data/.session_secret (auto-generated on first run, mode 0600)
    """
    import secrets

    env_key = os.environ.get('SESSION_SECRET_KEY', '').strip()
    if env_key:
        return env_key

    secret_file = _BACKEND_DIR / 'data' / '.session_secret'
    if secret_file.exists():
        try:
            value = secret_file.read_text().strip()
            if value:
                return value
        except Exception as e:
            logger.warning(f"[Flask] Could not read {secret_file}: {e}")

    # Generate a new secret and persist it
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    secret_file.write_text(value)
    secret_file.chmod(0o600)
    logger.info(f"[Flask] Generated new session secret → {secret_file}")
    return value


def _init_dashboard_gateway(app):
    """Initialise the dashboard gateway for interface daemons.

    Registers the gateway and app-management blueprints, inits the dashboard
    DB, and creates a ChalieClient for proxying signals/context to the backend.
    All runs inside the same Flask process — no separate server or proxy needed.
    """
    import sys

    # Ensure the repo root is on sys.path so `frontend.server` is importable.
    # In Docker: /app is the repo root, /app/backend is the working dir.
    repo_root = str(_BACKEND_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from frontend.server import db as dashboard_db
    from frontend.server.gateway import gateway_bp, init_gateway
    from frontend.server.interfaces import interfaces_bp as apps_bp, init_interfaces
    from frontend.server.chalie_client import ChalieClient

    from runtime_config import get as rc_get

    port = int(rc_get('port', 8081))

    # Dashboard DB lives alongside Chalie's data
    data_dir = str(_BACKEND_DIR / 'data')
    db_path = os.path.join(data_dir, 'dashboard.db')
    dashboard_db.init_db(db_path)

    # ChalieClient talks to ourselves (localhost) — no network hop.
    # Auto-generate a wrapper bearer token so the gateway can authenticate
    # with Chalie's own API endpoints (signals, context, interfaces).
    client = ChalieClient(f'http://127.0.0.1:{port}')
    boot_token = dashboard_db.get_config('boot_token')
    if not boot_token:
        try:
            from services.wrapper_auth_service import WrapperAuthService
            from services.database_service import get_shared_db_service
            wrapper_svc = WrapperAuthService(get_shared_db_service())
            boot_token, _wrapper_id = wrapper_svc.create_token(
                name="Dashboard Gateway",
                capabilities={"signals": ["*"]},
                permissions={"query": ["memory", "context"], "update": ["context"]},
                metadata={"type": "dashboard", "version": "1.0.0"},
                wrapper_id_override="__dashboard_gateway__",
            )
            dashboard_db.set_config('boot_token', boot_token)
            logger.info("[Dashboard] Auto-generated boot token for gateway auth")
        except Exception as e:
            logger.warning("[Dashboard] Could not generate boot token: %s", e)
    if boot_token:
        client.token = boot_token

    init_gateway(client, port, data_dir)
    init_interfaces(client, port, data_dir)

    app.register_blueprint(gateway_bp)
    app.register_blueprint(apps_bp)

    logger.info("[Dashboard] Gateway + app management blueprints registered")


def create_app():
    """Create and configure Flask application with all blueprints."""
    app = Flask(__name__)

    # Set secret key for cookie signing.
    # Auto-generated on first run and persisted to data/.session_secret (mode 0600).
    # Override with SESSION_SECRET_KEY env var only if you need cross-instance session sharing.
    app.secret_key = _get_or_generate_session_secret()

    # Upload limit (50MB for document uploads)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    # Reverse proxy support: trust X-Forwarded-For, X-Forwarded-Proto, etc.
    # This ensures request.remote_addr, request.host, and request.scheme
    # reflect the client's values when behind nginx/caddy/cloudflare.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # CORS — allow all origins (single-user personal assistant)
    CORS(app)

    # Register blueprints
    from .user_auth import user_auth_bp
    from .system import system_bp
    from .conversation import conversation_bp
    from .memory import memory_bp
    from .proactive import proactive_bp
    from .privacy import privacy_bp
    from .stubs import stubs_bp
    from .push import push_bp
    from .tools import tools_bp
    from .providers import providers_bp
    from .scheduler import scheduler_bp
    from .lists import lists_bp
    from .moments import moments_bp
    from .documents import documents_bp
    from .voice import voice_bp
    from .chat_image import chat_image_bp
    from .interfaces import interfaces_bp
    from .signals import signals_bp
    from .context import context_bp
    from .wrappers import wrappers_bp
    from .updates import updates_bp
    from .query import query_bp
    from .intents import intents_bp
    from .browser import browser_bp
    from .capabilities import capabilities_bp
    from .probe import probe_bp

    app.register_blueprint(user_auth_bp)
    app.register_blueprint(system_bp)
    app.register_blueprint(conversation_bp)
    app.register_blueprint(memory_bp)
    app.register_blueprint(proactive_bp)
    app.register_blueprint(privacy_bp)
    app.register_blueprint(stubs_bp)
    app.register_blueprint(push_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(providers_bp)
    app.register_blueprint(scheduler_bp)
    app.register_blueprint(lists_bp)
    app.register_blueprint(moments_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(voice_bp)
    app.register_blueprint(chat_image_bp)
    app.register_blueprint(interfaces_bp)
    app.register_blueprint(signals_bp)
    app.register_blueprint(context_bp)
    app.register_blueprint(wrappers_bp)
    app.register_blueprint(updates_bp)
    app.register_blueprint(query_bp)
    app.register_blueprint(intents_bp)
    app.register_blueprint(browser_bp)
    app.register_blueprint(capabilities_bp)
    app.register_blueprint(probe_bp)

    # ── Dashboard gateway (interface daemons) ─────────────────────
    _init_dashboard_gateway(app)

    # WebSocket endpoint (replaces SSE for chat + drift)
    from flask_sock import Sock
    sock = Sock(app)
    from .websocket import register_websocket
    register_websocket(sock)

    # ── Static file serving (replaces nginx) ─────────────────────────

    @app.route('/shared/<path:filename>')
    def shared_static(filename):
        """Serve shared frontend assets (theme.css, etc.)."""
        return send_from_directory(str(_SHARED_DIR), filename)

    @app.route('/brain/<path:filename>')
    def brain_static(filename):
        """Serve brain dashboard SPA."""
        filepath = _BRAIN_DIR / filename
        if filepath.is_file():
            return send_from_directory(str(_BRAIN_DIR), filename)
        return send_from_directory(str(_BRAIN_DIR), 'index.html')

    @app.route('/brain/')
    @app.route('/brain')
    def brain_index():
        """Serve brain dashboard index. Redirects to login if unauthenticated."""
        from services.auth_session_service import validate_session
        from flask import request
        if not validate_session(request):
            return redirect('/login/?next=/brain/')
        return send_from_directory(str(_BRAIN_DIR), 'index.html')

    @app.route('/on-boarding/<path:filename>')
    def onboarding_static(filename):
        """Serve onboarding SPA."""
        filepath = _ONBOARDING_DIR / filename
        if filepath.is_file():
            return send_from_directory(str(_ONBOARDING_DIR), filename)
        return send_from_directory(str(_ONBOARDING_DIR), 'index.html')

    @app.route('/on-boarding/')
    @app.route('/on-boarding')
    def onboarding_index():
        """Serve onboarding index."""
        return send_from_directory(str(_ONBOARDING_DIR), 'index.html')

    @app.route('/login/<path:filename>')
    def login_static(filename):
        """Serve login page assets."""
        filepath = _LOGIN_DIR / filename
        if filepath.is_file():
            return send_from_directory(str(_LOGIN_DIR), filename)
        return send_from_directory(str(_LOGIN_DIR), 'index.html')

    @app.route('/login/')
    @app.route('/login')
    def login_index():
        """Serve login page."""
        return send_from_directory(str(_LOGIN_DIR), 'index.html')

    # Main interface SPA — catch-all (must be last)
    @app.route('/<path:filename>')
    def interface_static(filename):
        """Serve main interface SPA files."""
        # Skip API routes (they're handled by blueprints with url_prefix or route names)
        filepath = _INTERFACE_DIR / filename
        if filepath.is_file():
            return send_from_directory(str(_INTERFACE_DIR), filename)
        # SPA fallback: serve index.html for client-side routing
        return send_from_directory(str(_INTERFACE_DIR), 'index.html')

    @app.route('/')
    def interface_index():
        """Serve main interface index."""
        return send_from_directory(str(_INTERFACE_DIR), 'index.html')

    logger.info("[REST API] All blueprints + WebSocket + static serving registered")
    return app
