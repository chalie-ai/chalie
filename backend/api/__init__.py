"""
REST API package — Flask app factory with Blueprint registration, WebSocket,
and static file serving (replaces nginx).
"""

import re
import importlib
import mimetypes
import pkgutil
import logging
from pathlib import Path
from flask import Flask, Blueprint, Response, redirect, send_from_directory
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

_FRONTEND_DIR = FileMapperService.get_frontend_path()
_INTERFACE_DIR = FileMapperService.get_frontend_path("interface")
_BRAIN_DIR = FileMapperService.get_frontend_path("brain")
_ONBOARDING_DIR = FileMapperService.get_frontend_path("on-boarding")
_LOGIN_DIR = FileMapperService.get_frontend_path("login")
_SHARED_DIR = FileMapperService.get_frontend_path("shared")


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

_VERSION_FILE = FileMapperService.get_version_path()


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


def _serve_spa(directory: Path, filename: str) -> Response:
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


def _configure_app(app: Flask) -> None:
    """Apply Flask config, proxy middleware, and CORS to a new app instance."""
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    CORS(app)


def _register_static_routes(app: Flask) -> None:
    """Register all static-file and SPA routes on the Flask app."""

    @app.route('/shared/<path:filename>', methods=["GET"])
    def shared_static(filename):
        """Serve shared frontend assets (theme.css, etc.)."""
        real = _strip_version_from_path(filename)
        return send_from_directory(str(_SHARED_DIR), real)

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


def create_app():
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
