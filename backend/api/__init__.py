"""
REST API package — Flask app factory with Blueprint registration, WebSocket,
and static file serving (replaces nginx).
"""

import os
import logging
from pathlib import Path
from flask import Flask, send_from_directory, send_file
from flask_cors import CORS

from .auth import require_session


logger = logging.getLogger(__name__)

# Resolve frontend directories relative to backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _BACKEND_DIR.parent / 'frontend'
_INTERFACE_DIR = _FRONTEND_DIR / 'interface'
_BRAIN_DIR = _FRONTEND_DIR / 'brain'
_ONBOARDING_DIR = _FRONTEND_DIR / 'on-boarding'


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
    from .app_update_api import app_update_bp

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
    app.register_blueprint(app_update_bp)

    # WebSocket endpoint (replaces SSE for chat + drift)
    from flask_sock import Sock
    sock = Sock(app)
    from .websocket import register_websocket
    register_websocket(sock)

    # ── Static file serving (replaces nginx) ─────────────────────────

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
        """Serve brain dashboard index."""
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
