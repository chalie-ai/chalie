# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — the turn_execution DTO returned by the live send endpoint.

``POST /api/threads/-1`` returns the turn's freshly-opened turn_execution row
inline (not just a bare turn_id) so the FE holds the whole lifecycle handle
with no WS round-trip. This drives the real endpoint against the real test db
— zero mocks — and cross-checks the response body against a fresh DB read of
the same row.
"""

import sqlite3
import threading
from typing import cast

import pytest
from flask.testing import FlaskClient

pytestmark = pytest.mark.unit


def test_post_send_returns_full_turn_execution_dto_with_live_turn_id(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """POST /api/threads/-1 returns the turn's freshly-opened turn_execution row
    inline — not just a bare turn_id — so the FE holds the whole lifecycle
    handle with no WS round-trip. The id it names is real: a fresh read of the
    same row from the db matches the response body field for field."""
    client, db_conn, _store = authed_client

    resp = client.post('/api/threads/-1', data={'text': 'hello there'})

    assert resp.status_code == 200
    body = resp.get_json()['result']
    assert body['state'] == 'working'
    assert body['cancel_requested'] is False
    assert body['channel'] == 'user'
    assert body['ended_at'] is None
    assert isinstance(body['id'], int)
    turn_id = cast(int, body['turn_id'])

    row = db_conn.execute(
        "SELECT state, cancel_requested, ended_at, turn_id FROM turn_executions WHERE id = ?", (body['id'],)
    ).fetchone()
    assert row is not None
    assert row["state"] == "working"
    assert bool(row["cancel_requested"]) is False
    assert row["ended_at"] is None
    assert row["turn_id"] == turn_id

    # Stop the background turn the dispatch spawned (no provider in test env) and
    # WAIT for it to actually exit — MessageProcessor.run() names its daemon
    # thread f"turn-{turn_id}" precisely so a caller can find and join it. A
    # fire-and-forget DELETE only requests cooperative cancellation; the thread's
    # real stop point is asynchronous. Joining it here (a real synchronization
    # primitive, not a sleep) keeps this test's db teardown from racing a write
    # still in flight, which would otherwise leak into whichever test runs next.
    client.delete(f'/api/threads/{turn_id}')
    turn_thread = next((t for t in threading.enumerate() if t.name == f"turn-{turn_id}"), None)
    if turn_thread is not None:
        turn_thread.join(timeout=10)
        assert not turn_thread.is_alive(), "the background turn must exit before the test's db is torn down"
