"""Feature test: the Vue interface build is served at canonical URLs (/, /login/, /on-boarding/).

Real Flask app via create_app(), real on-disk dist files. No mocks.
"""
import pathlib
from collections.abc import Generator

import pytest
from services.file_mapper_service import FileMapperService


@pytest.fixture
def built_dist() -> Generator[pathlib.Path, None, None]:
    """Provide a minimal Vite-style dist for the routes to serve, restoring any
    real on-disk build afterward so the test never clobbers build output.

    The routes hardcode the FileMapperService dist path, so the fixture writes
    into the real dir (no monkeypatching / no mocks) and reverts on teardown.
    Writes index.html (base '/'), assets/app.js, login/index.html, and
    on-boarding/index.html — covering all three interface entry points.
    """
    dist = FileMapperService.get_frontend_path("apps", "interface", "dist")
    assets = dist / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    index = dist / "index.html"
    app_js = assets / "app.js"
    login_dir = dist / "login"
    login_dir.mkdir(parents=True, exist_ok=True)
    login_index = login_dir / "index.html"
    onboarding_dir = dist / "on-boarding"
    onboarding_dir.mkdir(parents=True, exist_ok=True)
    onboarding_index = onboarding_dir / "index.html"

    original_index = index.read_bytes() if index.exists() else None
    app_js_preexisting = app_js.exists()
    original_login = login_index.read_bytes() if login_index.exists() else None
    original_onboarding = onboarding_index.read_bytes() if onboarding_index.exists() else None

    index.write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<script type="module" src="/assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    app_js.write_text("export const ok = true;\n", encoding="utf-8")
    login_index.write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<span class="login-entry-marker">login</span></body></html>',
        encoding="utf-8",
    )
    onboarding_index.write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<span class="onboarding-entry-marker">on-boarding</span></body></html>',
        encoding="utf-8",
    )

    try:
        yield dist
    finally:
        if original_index is not None:
            index.write_bytes(original_index)
        else:
            index.unlink(missing_ok=True)
        if not app_js_preexisting:
            app_js.unlink(missing_ok=True)
        if original_login is not None:
            login_index.write_bytes(original_login)
        else:
            login_index.unlink(missing_ok=True)
        if original_onboarding is not None:
            onboarding_index.write_bytes(original_onboarding)
        else:
            onboarding_index.unlink(missing_ok=True)


def test_root_serves_index_and_hashed_asset(built_dist: pathlib.Path) -> None:
    """/ serves the Vue interface SPA; /assets/* served verbatim; unknown deep
    paths fall back to index.html (SPA history mode)."""
    from api import create_app

    app = create_app()
    client = app.test_client()

    # index.html at the mount root
    r = client.get("/")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data
    # Asset ref uses canonical base '/' (not '/next/')
    assert b'/assets/app.js' in r.data

    # hashed asset served verbatim with a JS mimetype
    r = client.get("/assets/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["Content-Type"]

    # SPA fallback: unknown deep link returns index.html, not 404
    r = client.get("/some/deep/link")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data


def test_login_and_onboarding_entries_served(built_dist: pathlib.Path) -> None:
    """Login and on-boarding multi-page entries are served at their canonical paths."""
    from api import create_app

    app = create_app()
    client = app.test_client()

    # Login entry
    r = client.get("/login/")
    assert r.status_code == 200
    assert b'login-entry-marker' in r.data

    # On-boarding entry
    r = client.get("/on-boarding/")
    assert r.status_code == 200
    assert b'onboarding-entry-marker' in r.data

    # Bare /login → 301 to /login/
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["Location"].endswith("/login/")

    # Bare /on-boarding → 301 to /on-boarding/
    r = client.get("/on-boarding", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["Location"].endswith("/on-boarding/")
