"""
REST API package — Flask-RESTx app factory with Namespace auto-discovery,
WebSocket, and static file serving (replaces nginx).
"""

import importlib
import mimetypes
import pkgutil
import logging
from pathlib import Path
from flask import Flask, Blueprint, Response, redirect, send_from_directory
from flask.typing import ResponseReturnValue
from flask_cors import CORS
from flask_restx import Api, Namespace

from services.file_mapper_service import FileMapperService
from .auth import require_session as require_session
from .auth import internal_only


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


def _register_namespaces(app: Flask, api: Api) -> None:
    """Auto-discover and register every Namespace defined in this package.

    Walks ``backend/api/*.py``, imports each module, and registers any top-level
    ``Namespace`` instance on the given ``Api`` object. Modules without a
    Namespace (e.g. ``auth``, ``websocket``) are skipped silently.
    """
    package = importlib.import_module(__name__)
    seen: set[int] = set()
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith('_'):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for attr_name, attr in vars(module).items():
            if isinstance(attr, Namespace) and id(attr) not in seen:
                api.add_namespace(attr)
                seen.add(id(attr))
                logger.info("[REST API] Registered %s.%s", module_info.name, attr_name)
        # Also register plain Blueprints (gateway etc.) that are not Namespaces
        for attr_name, attr in vars(module).items():
            if isinstance(attr, Blueprint) and not isinstance(attr, Namespace) and id(attr) not in seen:
                app.register_blueprint(attr)
                seen.add(id(attr))
                logger.info("[REST API] Registered blueprint %s.%s", module_info.name, attr_name)

    # api.init_app() is deferred to create_app(): RESTx registers its own root
    # '/' route during init, which would shadow the SPA's '/' handler. Init must
    # run AFTER the static routes so the SPA wins the '/' match.


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}
"""Baseline hardening stamped on every response — blocks MIME-sniffing, cross-origin framing, and referrer leakage."""


def _deployment_origins() -> list[str]:
    """Cross-origin allowlist = the configured deployment domain (both schemes).

    Blank/unset → empty list, i.e. same-origin only (browsers do not enforce CORS
    on same-origin requests). Resolved once at app construction; the System page
    persists a domain change and restarts, so the new policy is read on next boot.
    """
    try:
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        domain = (SettingsService(get_shared_db_service()).get(SettingsService.DEPLOYMENT_DOMAIN) or "").strip()
    except Exception as exc:
        logger.warning("[REST API] deployment_domain unreadable; CORS limited to same-origin: %s", exc)
        return []
    return [f"https://{domain}", f"http://{domain}"] if domain else []


def _configure_app(app: Flask) -> None:
    """Apply Flask config, proxy middleware, CORS, and baseline security headers to a new app instance."""
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    # Emit every registered DTO in the OpenAPI definitions; nested-envelope
    # shapes (a list of associations inside a result DTO) otherwise dangle.
    app.config['RESTX_INCLUDE_ALL_MODELS'] = True

    from werkzeug.middleware.proxy_fix import ProxyFix
    setattr(app, 'wsgi_app', ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1))

    CORS(app, origins=_deployment_origins(), supports_credentials=True)

    @app.after_request
    def _apply_security_headers(response: Response) -> Response:
        """Stamp the baseline security headers on every response; JSON payloads
        also get ``no-store`` so sensitive API data is never cached downstream.
        ``setdefault`` lets a route's own explicit header (e.g. the SPA index's
        ``no-cache``) win."""
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if response.mimetype == "application/json":
            response.headers.setdefault("Cache-Control", "no-store")
        return response


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
    @internal_only
    def pairing_index_no_slash() -> ResponseReturnValue:
        return redirect('/pairing/', code=301)

    @app.route('/pairing/', methods=["GET"])
    @internal_only
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
    """Create and configure Flask application with all namespaces and routes."""
    app = Flask(__name__)

    # A fresh Api per app keeps create_app() a true factory: the test suite
    # builds many apps, and a module-level Api would re-register routes onto an
    # already-served app (Flask forbids add_url_rule after the first request).
    api = Api(
        title="Chalie API",
        version="1.0",
        description="REST API for the Chalie personal intelligence layer",
        doc="/swagger/",
    )

    _configure_app(app)
    _register_namespaces(app, api)

    # WebSocket endpoint (replaces SSE for chat + drift)
    from flask_sock import Sock
    sock = Sock(app)
    from .websocket import register_websocket
    register_websocket(sock)

    # ── Static file serving (replaces nginx) ─────────────────────────
    _register_static_routes(app)

    # Flush RESTx namespaces onto the app LAST so its root '/' route is added
    # after the SPA '/' handler — the SPA must win the '/' match (see above).
    api.init_app(app)

    logger.info("[REST API] All namespaces + WebSocket + static serving registered")
    return app
