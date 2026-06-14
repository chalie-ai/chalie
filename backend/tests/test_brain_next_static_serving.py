"""Feature test: the Vue Brain build is served at /brain-next/ alongside legacy /brain/.

Real Flask app via create_app(), real on-disk dist, no mocks for the
filesystem. Auth is bypassed with a patch on validate_session exactly as
every other auth-gated route test in this suite does (see conftest.authed_client,
test_api_system.py, test_policies_api.py — all patch
``services.auth_session_service.validate_session``).
"""

import pytest
from unittest.mock import patch
from services.file_mapper_service import FileMapperService


@pytest.fixture
def brain_next_dist():
    """Provide a minimal Vite-style dist for /brain-next/, restoring any
    real on-disk build afterward so the test never clobbers build output.

    Mirrors the built_dist fixture in test_next_static_serving.py.
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
        '<script type="module" src="/brain-next/assets/brain.abc.js"></script></body></html>',
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
def test_brain_next_bare_path_redirects_301(brain_next_dist):
    """/brain-next → /brain-next/ with 301 (no auth needed for redirect)."""
    from api import create_app

    app = create_app()
    r = app.test_client().get("/brain-next", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["Location"].endswith("/brain-next/")


@pytest.mark.unit
def test_brain_next_unauth_redirects_to_login(brain_next_dist):
    """/brain-next/ without a session → 302 to /login/?next=/brain-next/."""
    from api import create_app

    with patch("services.auth_session_service.validate_session", return_value=False):
        app = create_app()
        r = app.test_client().get("/brain-next/", follow_redirects=False)

    assert r.status_code == 302
    location = r.headers["Location"]
    assert "/login/" in location
    assert "next=/brain-next/" in location


@pytest.mark.unit
def test_brain_next_authed_serves_index(brain_next_dist):
    """/brain-next/ with a valid session → 200 + SPA index.html."""
    from api import create_app

    with patch("services.auth_session_service.validate_session", return_value=True):
        app = create_app()
        r = app.test_client().get("/brain-next/")

    assert r.status_code == 200
    assert b'<div id="app">' in r.data


@pytest.mark.unit
def test_brain_next_authed_unknown_deep_path_spa_fallback(brain_next_dist):
    """/brain-next/providers/some-deep-link → 200 SPA fallback (history mode reload)."""
    from api import create_app

    with patch("services.auth_session_service.validate_session", return_value=True):
        app = create_app()
        r = app.test_client().get("/brain-next/providers/some-deep-link")

    assert r.status_code == 200
    assert b'<div id="app">' in r.data
