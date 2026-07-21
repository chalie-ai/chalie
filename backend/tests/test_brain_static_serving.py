"""Feature test: the Vue Brain build is served at the canonical /brain/ URL.

Real Flask app via create_app(), real on-disk dist, real auth sessions — no mocks.
Unauthenticated redirect is asserted by the natural absence of a session cookie
(validate_session returns False when no cookie is present).
"""
import pathlib
import sqlite3
from collections.abc import Generator

import pytest

from services.file_mapper_service import FileMapperService


@pytest.fixture
def brain_dist() -> Generator[pathlib.Path, None, None]:
    """Provide a minimal Vite-style dist for /brain/, restoring any
    real on-disk build afterward so the test never clobbers build output.

    Mirrors the built_dist fixture in test_next_static_serving.py.
    Asset ref uses canonical /brain/assets/ prefix (base '/brain/').
    """
    dist = FileMapperService.get_frontend_path("apps", "brain", "dist")
    assets = dist / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    index = dist / "index.html"
    app_js = assets / "brain.abc.js"
    original_index = index.read_bytes() if index.exists() else None
    app_js_preexisting = app_js.exists()
    index.write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<script type="module" src="/brain/assets/brain.abc.js"></script></body></html>',
        encoding="utf-8",
    )
    app_js.write_text("export const brain = true;\n", encoding="utf-8")
    try:
        yield dist
    finally:
        if original_index is not None:
            index.write_bytes(original_index)
        else:
            index.unlink(missing_ok=True)
        if not app_js_preexisting:
            app_js.unlink(missing_ok=True)


@pytest.mark.unit
def test_brain_bare_path_redirects_301(brain_dist: pathlib.Path) -> None:
    """/brain → /brain/ with 301 (no auth needed for redirect)."""
    from api import create_app

    app = create_app()
    r = app.test_client().get("/brain", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["Location"].endswith("/brain/")


@pytest.mark.unit
def test_brain_unauth_redirects_to_login(brain_dist: pathlib.Path) -> None:
    """/brain/ without a session cookie → 302 to /login/?next=/brain/.

    No mock needed: validate_session returns False naturally when the request
    carries no chalie_session cookie.
    """
    from api import create_app

    app = create_app()
    r = app.test_client().get("/brain/", follow_redirects=False)
    assert r.status_code == 302
    location = r.headers["Location"]
    assert "/login/" in location
    assert "next=/brain/" in location


def test_brain_authed_serves_index(db: sqlite3.Connection, brain_dist: pathlib.Path) -> None:
    """/brain/ with a valid real session → 200 + SPA index.html."""
    from api import create_app
    from contracts.constants.auth import SESSION_COOKIE_NAME
    from services.auth_session_service import create_session
    from flask import Response as FlaskResponse

    app = create_app()
    client = app.test_client()
    # Mint a real session token via the real auth service (writes to the test
    # sqlite auth_sessions table + MemoryStore singleton). No mock.
    token = create_session(FlaskResponse())
    client.set_cookie(SESSION_COOKIE_NAME, token)
    r = client.get("/brain/")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data


def test_brain_authed_deep_path_spa_fallback(db: sqlite3.Connection, brain_dist: pathlib.Path) -> None:
    """/brain/providers/some-deep-link → 200 SPA fallback (history mode reload)."""
    from api import create_app
    from contracts.constants.auth import SESSION_COOKIE_NAME
    from services.auth_session_service import create_session
    from flask import Response as FlaskResponse

    app = create_app()
    client = app.test_client()
    token = create_session(FlaskResponse())
    client.set_cookie(SESSION_COOKIE_NAME, token)
    r = client.get("/brain/providers/some-deep-link")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data
