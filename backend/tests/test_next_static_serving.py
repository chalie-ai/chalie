"""Feature test: the Vue interface build is served at /next/ alongside legacy static.

Real Flask app, real on-disk dist. No mocks.
"""
import pytest
from services.file_mapper_service import FileMapperService


@pytest.fixture
def built_dist():
    """Provide a minimal Vite-style dist for the route to serve, restoring any
    real on-disk build afterward so the test never clobbers build output.

    The route hardcodes the FileMapperService dist path, so the fixture writes
    into the real dir (no monkeypatching / no mocks) and reverts on teardown.
    """
    dist = FileMapperService.get_frontend_path("apps", "interface", "dist")
    assets = dist / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    index = dist / "index.html"
    app_js = assets / "app.js"
    original_index = index.read_bytes() if index.exists() else None
    app_js_preexisting = app_js.exists()
    index.write_text(
        '<!doctype html><html><body><div id="app"></div>'
        '<script type="module" src="/next/assets/app.js"></script></body></html>',
        encoding="utf-8",
    )
    app_js.write_text("export const ok = true;\n", encoding="utf-8")
    try:
        yield dist
    finally:
        if original_index is not None:
            index.write_bytes(original_index)
        else:
            index.unlink(missing_ok=True)
        if not app_js_preexisting:
            app_js.unlink(missing_ok=True)


def test_next_serves_index_and_hashed_asset(built_dist):
    from api import create_app

    app = create_app()
    client = app.test_client()

    # index.html at the mount root
    r = client.get("/next/")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data
    # NOT version-injected — the asset ref stays exactly as Vite emitted it
    assert b'/next/assets/app.js' in r.data

    # hashed asset served verbatim with a JS mimetype
    r = client.get("/next/assets/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["Content-Type"]

    # SPA fallback: unknown deep link returns index.html, not 404
    r = client.get("/next/some/deep/link")
    assert r.status_code == 200
    assert b'<div id="app">' in r.data


def test_next_bare_path_redirects_to_trailing_slash():
    """Bare /next must 301 to /next/ — not fall through to the legacy catch-all."""
    from api import create_app

    app = create_app()
    r = app.test_client().get("/next")
    assert r.status_code == 301
    assert r.headers["Location"].endswith("/next/")


def test_legacy_interface_root_still_served():
    """Coexistence: the old hand-rolled interface index still serves at '/'."""
    from api import create_app

    app = create_app()
    r = app.test_client().get("/")
    assert r.status_code == 200
