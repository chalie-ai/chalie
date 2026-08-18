# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the MCP server's store-outage path — the silent-500 incident.

The incident these tests pin down: the MCP server process was alive and
healthy (the thread-liveness supervisor saw a running thread and reported
nothing) while the SQLite file every bearer token is validated against
became unreadable. Every request then died *inside* token validation, and
without the ``try/except`` in ``BearerTokenMiddleware.dispatch`` the
traceback escaped into the ASGI server's own stdout logger — it never
reached the application log — so the outage was completely invisible: the
server was "up" for everyone while serving nothing.

The behavior under test: a validation failure is logged loudly with
``logger.exception`` (full traceback) and answered with 503 "MCP server
temporarily unavailable" — a state an operator can actually see, and a
status that says "restarting the MCP server will not help; the store is
the broken component".

Feature tests against the real production path: the real
``mcp_server.server`` app (``create_mcp_server`` + ``_build_app``) behind
the real ``BearerTokenMiddleware``, driven through Starlette's
``TestClient``, against the real per-test SQLite database bound by the
``db`` fixture. Zero mocks: the store is made genuinely unreadable by
overwriting the database file with non-database bytes — the same physical
failure as the incident, not a simulated exception.
"""

import logging
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mcp_server.server import _build_app, create_mcp_server
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit


def _mcp_client() -> TestClient:
    """The real MCP app driven through the ASGI TestClient.

    Built without the context manager on purpose: all three tests are
    rejected by ``BearerTokenMiddleware`` *before* ``call_next`` reaches
    the router, so the app's lifespan (the session manager's task group)
    is never exercised and never needs to run.
    """
    return TestClient(_build_app(create_mcp_server()))


# ---------------------------------------------------------------------------
# Test 1 — unreadable store: 503 + loud logging, not a silent 500
# ---------------------------------------------------------------------------


def test_unreadable_store_returns_503_and_logs_loudly(
    db: sqlite3.Connection, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A request against a corrupted database answers 503 (not 500) and the
    failure is logged with a full traceback — the outage must be visible."""
    # Genuinely corrupt the bound database file — the real incident was a
    # file SQLite could no longer read, not a mocked exception. The db
    # fixture re-pointed the Database gateway at this test's own copy.
    db_path = FileMapperService.get_db_path()
    # Guard against corrupting the production data/chalie.db by mistake:
    # the gateway must resolve inside this test's temp directory.
    assert str(db_path).startswith(str(tmp_path))
    db_path.write_bytes(b"this is not a database" * 100)
    # Database.close_all() does not exist (only the per-thread
    # Database.close()), so no close is issued: connections are
    # per-(thread, path) and every page read re-hits the file, so the
    # middleware's worker thread reads the corrupted file whether it opens
    # a fresh connection or reuses one.

    with caplog.at_level(logging.ERROR):
        response = _mcp_client().post(
            "/mcp", headers={"Authorization": "Bearer anything"}
        )

    assert response.status_code == 503
    assert "MCP server temporarily unavailable" in response.text

    # The failure must be logged loudly — with exception info attached,
    # i.e. a full traceback — rather than escaping silently into the ASGI
    # server's stdout logger.
    loud = [
        record
        for record in caplog.records
        if record.name == "mcp_server.server"
        and "[MCP] Token validation failed" in record.message
        and record.exc_info is not None
    ]
    assert loud, "expected a logger.exception record with exc_info attached"


# ---------------------------------------------------------------------------
# Test 2 — the 503 branch must not swallow normal auth rejection
# ---------------------------------------------------------------------------


def test_missing_bearer_header_still_returns_401(db: sqlite3.Connection) -> None:
    """A missing Authorization header is a client error (401), not a store
    outage (503) — the 503 branch must not swallow ordinary auth rejection.
    """
    response = _mcp_client().post("/mcp")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.text


# ---------------------------------------------------------------------------
# Test 3 — healthy store: an unknown token is still a 401, not a 503
# ---------------------------------------------------------------------------


def test_unknown_token_on_healthy_store_returns_401(db: sqlite3.Connection) -> None:
    """With the database intact, a bad token is a client error (401), not a
    store outage (503) — a healthy store must still distinguish the two.
    """
    response = _mcp_client().post(
        "/mcp", headers={"Authorization": "Bearer definitely-not-a-real-token"}
    )
    assert response.status_code == 401
    assert "Invalid or revoked token" in response.text
