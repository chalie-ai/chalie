"""
REST API package — Flask app factory with Blueprint registration, WebSocket,
and static file serving (replaces nginx).
"""

import importlib
import mimetypes
import pkgutil
import logging
from pathlib import Path
from flask import Flask, Blueprint, Response, redirect, send_from_directory
from flask.typing import ResponseReturnValue
from flask_cors import CORS

from services.file_mapper_service import FileMapperService
from .auth import require_session as require_session


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIME type registration (cross-platform safety net)
#
# Python's `mimetypes` module reads the Windows registry on Windows, which is
# frequently stale or missing the standard JS/CSS/JSON entries. When that
# happens, Flask's `send_from_directory` ships `.js` files as `text/plain`,
# and strict browsers (Chrome, Edge) refuse to execute them — breaking the
# entire frontend.
#
# Registering these at import time, before any Flask app is constructed,
# guarantees that `mimetypes.guess_type()` returns the correct value on every
# OS. `mimetypes.add_type` overrides any pre-existing mapping for the given
# extension, so it neutralises the bad registry entries on Windows without
# affecting macOS/Linux (which already return the correct types).
# ---------------------------------------------------------------------------

mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('application/json', '.json')
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('text/html', '.html')


def _register_blueprints(app: Flask) -> None:
    """Auto-discover and register every Blueprint defined in this package.

    Walks `backend/api/*.py`, imports each module, and registers any top-level
    `Blueprint` instance it exposes. Modules without a Blueprint (e.g. `auth`,
    `websocket`) are skipped silently. Drop a new `foo.py` exposing `foo_bp`
    in this folder and it lights up on next boot — no edits here required.
    """
    package = importlib.import_module(__name__)
    seen: set[int] = set()
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith('_'):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for attr_name, attr in vars(module).items():
            if not isinstance(attr, Blueprint) or id(attr) in seen:
                continue
            app.register_blueprint(attr)
            seen.add(id(attr))
            logger.info("[REST API] Registered %s.%s", module_info.name, attr_name)


def _configure_app(app: Flask) -> None:
    """Apply Flask config, proxy middleware, and CORS to a new app instance."""
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    from werkzeug.middleware.proxy_fix import ProxyFix
    setattr(app, 'wsgi_app', ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1))

    CORS(app)


def _register_static_routes(app: Flask) -> None:
    """Register static-file and SPA routes serving the Vue 3 builds.

    Two Vite builds are served verbatim (Vite content-hashes its own assets):
      • interface  →  apps/interface/dist  — chat SPA at '/', plus the login,
        on-boarding, and pairing multi-page entries (dist/login/index.html,
        dist/on-boarding/index.html, dist/pairing/index.html).
      • brain      →  apps/brain/dist      — admin SPA at '/brain/', auth-gated.

    index.html documents are served no-cache (they point at hashed asset URLs);
    SEND_FILE_MAX_AGE_DEFAULT=0 keeps hashed assets revalidating, which is correct
    because a new build produces new filenames.
    """
    interface_dir = FileMapperService.get_frontend_path("apps", "interface", "dist")
    brain_dir = FileMapperService.get_frontend_path("apps", "brain", "dist")

    def _send_index(directory: Path, filename: str = 'index.html') -> Response:
        resp = send_from_directory(str(directory), filename)
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp

    # ── Brain admin SPA (auth-gated, matches the legacy /brain/ gate) ─────
    @app.route('/brain', methods=["GET"])
    def brain_index_no_slash() -> ResponseReturnValue:
        return redirect('/brain/', code=301)

    @app.route('/brain/', methods=["GET"])
    def brain_index() -> ResponseReturnValue:
        from services.auth_session_service import validate_session
        from flask import request
        if not validate_session(request):
            return redirect('/login/?next=/brain/')
        return _send_index(brain_dir)

    @app.route('/brain/<path:filename>', methods=["GET"])
    def brain_static(filename: str) -> ResponseReturnValue:
        from services.auth_session_service import validate_session
        from flask import request
        if not validate_session(request):
            return redirect(f'/login/?next=/brain/{filename}')
        candidate = brain_dir / filename
        if candidate.is_file():
            return send_from_directory(str(brain_dir), filename)
        return _send_index(brain_dir)

    # ── Login + on-boarding (interface multi-page entries, pre-auth) ──────
    @app.route('/login', methods=["GET"])
    def login_index_no_slash() -> ResponseReturnValue:
        return redirect('/login/', code=301)

    @app.route('/login/', methods=["GET"])
    def login_index() -> ResponseReturnValue:
        return _send_index(interface_dir, 'login/index.html')

    @app.route('/on-boarding', methods=["GET"])
    def onboarding_index_no_slash() -> ResponseReturnValue:
        return redirect('/on-boarding/', code=301)

    @app.route('/on-boarding/', methods=["GET"])
    def onboarding_index() -> ResponseReturnValue:
        return _send_index(interface_dir, 'on-boarding/index.html')

    # The native (Tauri) client redirects here before any login to scan its
    # pairing QR; web never reaches it (the gate diverts only on the Tauri runtime).
    @app.route('/pairing', methods=["GET"])
    def pairing_index_no_slash() -> ResponseReturnValue:
        return redirect('/pairing/', code=301)

    @app.route('/pairing/', methods=["GET"])
    def pairing_index() -> ResponseReturnValue:
        return _send_index(interface_dir, 'pairing/index.html')

    # ── Interface chat SPA — catch-all (MUST be registered last) ──────────
    @app.route('/<path:filename>', methods=["GET"])
    def interface_static(filename: str) -> ResponseReturnValue:
        candidate = interface_dir / filename
        if candidate.is_file():
            return send_from_directory(str(interface_dir), filename)
        return _send_index(interface_dir)

    @app.route('/', methods=["GET"])
    def interface_index() -> ResponseReturnValue:
        return _send_index(interface_dir)


def create_app() -> Flask:
    """Create and configure Flask application with all blueprints."""
    app = Flask(__name__)

    _configure_app(app)
    _register_blueprints(app)

    # WebSocket endpoint (replaces SSE for chat + drift)
    from flask_sock import Sock
    sock = Sock(app)
    from .websocket import register_websocket
    register_websocket(sock)

    # ── Static file serving (replaces nginx) ─────────────────────────
    _register_static_routes(app)

    logger.info("[REST API] All blueprints + WebSocket + static serving registered")
    return app
