"""Feature test: unknown ``/api/*`` paths fail loud as JSON 404, never SPA HTML.

The interface SPA catch-all serves ``index.html`` for any unmatched GET so
client-side routes deep-link correctly. ``/api/`` is the one namespace where
that fallback is actively harmful: an unregistered API path answered with the
SPA index reads as HTML 200 to every client — a missing or renamed route is
masked as success instead of failing loud (Manifesto P9; this exact masking
hid a stale-backend/new-frontend mismatch in the wild).

Real production path, zero mocks: the real ``create_app()`` app (real static
routes over the committed ``dist`` builds, real registered API namespaces) and
a real test client, with the real ``db`` fixture so app construction (which
reads ``Setting`` for CORS config) succeeds against a real SQLite file.
"""

from __future__ import annotations

import sqlite3

import pytest
from flask.testing import FlaskClient

pytestmark = pytest.mark.unit


@pytest.fixture
def client(db: sqlite3.Connection) -> FlaskClient:
    """A real test client over the real app factory."""
    from api import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


class TestUnknownApiPathsFailLoud:
    def test_unknown_api_path_returns_json_404_envelope(self, client: FlaskClient) -> None:
        resp = client.get("/api/definitely-not-a-route")
        assert resp.status_code == 404
        assert resp.content_type.startswith("application/json")
        assert resp.get_json() == {"success": False, "result": [], "error": "Not found"}

    def test_bare_api_segment_returns_json_404_envelope(self, client: FlaskClient) -> None:
        resp = client.get("/api")
        assert resp.status_code == 404
        assert resp.get_json() == {"success": False, "result": [], "error": "Not found"}

    def test_registered_api_route_is_untouched_by_the_guard(self, client: FlaskClient) -> None:
        # A real registered route must still dispatch normally — here the auth
        # gate answers first with the bare 401 shape, proving the request
        # reached the API layer rather than the static fallback.
        resp = client.get("/api/providers/all")
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}

    def test_unknown_non_api_path_still_serves_the_spa_index(self, client: FlaskClient) -> None:
        # The SPA fallback itself must survive the guard: client-side routes
        # (anything outside /api/) keep deep-linking into index.html.
        resp = client.get("/some-client-side-route")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")
