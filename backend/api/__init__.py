"""
REST API package — Flask app factory with Blueprint registration, WebSocket,
and static file serving (replaces nginx).
"""

import os
import re
import importlib
import pkgutil
import logging
from pathlib import Path
from flask import Flask, Blueprint, Response, redirect, send_from_directory
from flask_cors import CORS

import paths
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


# ---------------------------------------------------------------------------
# Asset version injection
#
# Every <script src="…">, <link href="…">, <img src="…">, <source src="…"> in a
# served HTML page is rewritten so the filename itself carries the current app
# version — e.g. `app.js` → `app-0.3.3.js`, `style.css` → `style-0.3.3.css`.
#
# Static file routes transparently strip the `-{VERSION}` suffix before looking
# the file up on disk, so the on-disk filenames stay clean.
#
# Versioned *paths* (not query strings) are chosen because some intermediate
# caches and Service Workers treat `foo.js?v=1` and `foo.js?v=2` as the same
# entry; a distinct filename is universally treated as a new resource.
# ---------------------------------------------------------------------------

_VERSION_FILE = _BACKEND_DIR.parent / 'VERSION'


def _read_asset_version() -> str:
    """Return the version string used in asset filenames. Falls back to 'dev'."""
    try:
        value = _VERSION_FILE.read_text(encoding='utf-8').strip()
        return value or 'dev'
    except OSError:
        return 'dev'


_ASSET_VERSION = _read_asset_version()

_ASSET_REF_RE = re.compile(
    r'''(<(?:script|link|img|source)\b[^>]*?\s(?:src|href)\s*=\s*)(["'])([^"']+?)\2''',
    re.IGNORECASE,
)

# Extension that receives `-{version}` injection. Bare paths (no extension) and
# HTML itself are left alone.
_VERSIONABLE_EXT_RE = re.compile(r'^(.*?)(\.[^./]+)$')

_VERSION_SUFFIX_RE = re.compile(
    rf'(.+?)-{re.escape(_ASSET_VERSION)}(\.[^./]+)$'
)


def _resolve_same_origin_asset(url: str, base_dir: Path) -> Path | None:
    """Map a URL found in HTML back to a file on disk — or None if external."""
    if not url or url.startswith(('http://', 'https://', '//', 'data:', 'mailto:', 'tel:', '#')):
        return None
    if url.startswith('/'):
        candidate = _FRONTEND_DIR / url.lstrip('/')
    else:
        candidate = base_dir / url
    return candidate if candidate.is_file() else None


def _inject_version_into_url(url: str) -> str:
    """Insert `-{VERSION}` before the final extension. Returns url unchanged if it has none."""
    if not _ASSET_VERSION or _ASSET_VERSION == 'dev':
        return url
    match = _VERSIONABLE_EXT_RE.match(url)
    if not match:
        return url
    return f"{match.group(1)}-{_ASSET_VERSION}{match.group(2)}"


def _strip_version_from_path(path: str) -> str:
    """Inverse of _inject_version_into_url — remove `-{VERSION}` before the extension."""
    match = _VERSION_SUFFIX_RE.match(path)
    return f"{match.group(1)}{match.group(2)}" if match else path


def _version_html(html: str, base_dir: Path) -> str:
    """Rewrite every same-origin asset reference to carry the version in its filename."""
    def repl(match: re.Match) -> str:
        prefix, quote, url = match.group(1), match.group(2), match.group(3)
        if '?' in url:
            return match.group(0)
        asset = _resolve_same_origin_asset(url, base_dir)
        if asset is None:
            return match.group(0)
        # Service-worker script must stay at its registration URL — browsers
        # identify SW registrations by scriptURL. Renaming would orphan the
        # existing registration and install a second, conflicting SW.
        if asset.name == 'sw.js':
            return match.group(0)
        versioned = _inject_version_into_url(url)
        if versioned == url:
            return match.group(0)
        return f"{prefix}{quote}{versioned}{quote}"

    return _ASSET_REF_RE.sub(repl, html)


def _serve_versioned_html(directory: Path, filename: str = 'index.html') -> Response:
    """Read an HTML file, inject asset versions, return a no-cache response."""
    path = directory / filename
    html = path.read_text(encoding='utf-8')
    versioned = _version_html(html, directory)
    resp = Response(versioned, mimetype='text/html; charset=utf-8')
    # HTML itself must never be cached — otherwise the browser would keep
    # serving an old doc that points at old versioned URLs forever.
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


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

    secret_file = paths.SESSION_SECRET_PATH
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
    data_dir = str(paths.DATA_DIR)
    dashboard_db.init_db(str(paths.DASHBOARD_DB_PATH))

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


def create_app():
    """Create and configure Flask application with all blueprints."""
    app = Flask(__name__)

    # Set secret key for cookie signing.
    # Auto-generated on first run and persisted to data/.session_secret (mode 0600).
    # Override with SESSION_SECRET_KEY env var only if you need cross-instance session sharing.
    app.secret_key = _get_or_generate_session_secret()

    # Upload limit (50MB for document uploads)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    # Static assets: force browser to revalidate on every request via ETag/Last-Modified.
    # Flask returns 304 when unchanged, so bandwidth cost is minimal, but a simple
    # refresh always picks up redeployed assets. Paired with a no-cache Service Worker.
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    # Reverse proxy support: trust X-Forwarded-For, X-Forwarded-Proto, etc.
    # This ensures request.remote_addr, request.host, and request.scheme
    # reflect the client's values when behind nginx/caddy/cloudflare.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # CORS — allow all origins (single-user personal assistant)
    CORS(app)

    _register_blueprints(app)

    # ── Dashboard gateway (interface daemons) ─────────────────────
    _init_dashboard_gateway(app)

    # WebSocket endpoint (replaces SSE for chat + drift)
    from flask_sock import Sock
    sock = Sock(app)
    from .websocket import register_websocket
    register_websocket(sock)

    # ── Static file serving (replaces nginx) ─────────────────────────

    @app.route('/shared/<path:filename>', methods=["GET"])
    def shared_static(filename):
        """Serve shared frontend assets (theme.css, etc.)."""
        real = _strip_version_from_path(filename)
        return send_from_directory(str(_SHARED_DIR), real)

    def _serve_spa(directory: Path, filename: str):
        """Serve a static file, or fall back to a versioned index.html.

        Incoming paths may carry the `-{VERSION}` suffix injected by the HTML
        rewriter; strip it so the request resolves to the real on-disk file.
        """
        real = _strip_version_from_path(filename)
        filepath = directory / real
        if filepath.is_file():
            if filepath.suffix.lower() in ('.html', '.htm'):
                return _serve_versioned_html(directory, real)
            return send_from_directory(str(directory), real)
        return _serve_versioned_html(directory)

    @app.route('/brain/<path:filename>', methods=["GET"])
    def brain_static(filename):
        """Serve brain dashboard SPA."""
        return _serve_spa(_BRAIN_DIR, filename)

    @app.route('/brain', methods=["GET"])
    def brain_index_no_slash():
        """Canonicalize /brain → /brain/ so relative asset paths resolve correctly."""
        return redirect('/brain/', code=301)

    @app.route('/brain/', methods=["GET"])
    def brain_index():
        """Serve brain dashboard index. Redirects to login if unauthenticated."""
        from services.auth_session_service import validate_session
        from flask import request
        if not validate_session(request):
            return redirect('/login/?next=/brain/')
        return _serve_versioned_html(_BRAIN_DIR)

    @app.route('/on-boarding/<path:filename>', methods=["GET"])
    def onboarding_static(filename):
        """Serve onboarding SPA."""
        return _serve_spa(_ONBOARDING_DIR, filename)

    @app.route('/on-boarding', methods=["GET"])
    def onboarding_index_no_slash():
        """Canonicalize /on-boarding → /on-boarding/ so relative asset paths resolve correctly."""
        return redirect('/on-boarding/', code=301)

    @app.route('/on-boarding/', methods=["GET"])
    def onboarding_index():
        """Serve onboarding index."""
        return _serve_versioned_html(_ONBOARDING_DIR)

    @app.route('/login/<path:filename>', methods=["GET"])
    def login_static(filename):
        """Serve login page assets."""
        return _serve_spa(_LOGIN_DIR, filename)

    @app.route('/login', methods=["GET"])
    def login_index_no_slash():
        """Canonicalize /login → /login/ so relative asset paths resolve correctly."""
        return redirect('/login/', code=301)

    @app.route('/login/', methods=["GET"])
    def login_index():
        """Serve login page."""
        return _serve_versioned_html(_LOGIN_DIR)

    # Main interface SPA — catch-all (must be last)
    @app.route('/<path:filename>', methods=["GET"])
    def interface_static(filename):
        """Serve main interface SPA files."""
        return _serve_spa(_INTERFACE_DIR, filename)

    @app.route('/', methods=["GET"])
    def interface_index():
        """Serve main interface index."""
        return _serve_versioned_html(_INTERFACE_DIR)

    logger.info("[REST API] All blueprints + WebSocket + static serving registered")
    return app
