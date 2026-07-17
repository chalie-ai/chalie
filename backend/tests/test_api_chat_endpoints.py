"""Feature tests for the chat API HTTP contracts.

Drives the real Flask app against a real SQLite database via authed_client.
Pins synchronous endpoint contracts only; the async full-turn path (real UMP + LLM) is out of scope.
"""

import sqlite3
import threading

import pytest
from flask.testing import FlaskClient


@pytest.mark.unit
class TestChatEndpoints:

    def test_thread_empty_message_rejected_and_no_turn_started(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """An empty POST /api/thread/-1 fails loud with 422/message required and must
        NOT register a turn — a follow-up interrupt finds nothing in flight."""
        client, _db_conn, _store = authed_client

        resp = client.post('/api/thread/-1', data={'text': '   '})

        assert resp.status_code == 422
        assert resp.get_json()['error'] == 'message required'

        # Cross-step proof: no turn was registered for the empty message, so
        # interrupting any turn_id finds nothing in flight.
        interrupt = client.delete('/api/thread/1')
        assert interrupt.status_code == 200
        assert interrupt.get_json() == {'ok': True, 'interrupted': None, 'reason': 'no_active_turn'}

    def test_thread_post_allocates_turn_id_synchronously(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /api/thread/-1 returns 200 with the synchronously-allocated
        ``{turn_id, type}`` — not the old empty 201. The id is REAL: its opener
        row is already in the transcript before the response returns (the request
        thread writes it, so the FE holds a live handle with no WS round-trip)."""
        client, db_conn, _store = authed_client

        resp = client.post('/api/thread/-1', data={'text': 'hello there'})

        assert resp.status_code == 200
        body = resp.get_json()
        turn_id = body['turn_id']
        assert isinstance(turn_id, int) and turn_id >= 1
        assert body['type'] == 'user'

        # Cross-step proof the id is real, not fabricated: the opener row was
        # written synchronously (before the response) and is queryable now.
        row = db_conn.execute(
            "SELECT content, role FROM transcript WHERE channel='user' AND turn_id=? ORDER BY id LIMIT 1",
            (turn_id,),
        ).fetchone()
        assert row is not None and row[0] == 'hello there' and row[1] == 'user'

        # Stop the background turn the dispatch spawned (no provider in test env) and
        # WAIT for it to actually exit — MessageProcessor.run() names its daemon
        # thread f"turn-{turn_id}" precisely so a caller can find and join it. A
        # fire-and-forget DELETE only requests cooperative cancellation; joining the
        # real thread (not a sleep) keeps this test's db teardown from racing a
        # write still in flight, which would otherwise leak into the next test.
        client.delete(f'/api/thread/{turn_id}')
        turn_thread = next((t for t in threading.enumerate() if t.name == f"turn-{turn_id}"), None)
        if turn_thread is not None:
            turn_thread.join(timeout=10)
            assert not turn_thread.is_alive(), "the background turn must exit before the test's db is torn down"

    def test_thread_post_nonexistent_turn_id_rejected_and_writes_nothing(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /api/thread/<id> naming a turn that does not exist must bubble the
        constructor's ``Invalid turn_id specified`` — no phantom turn is minted, no
        transcript row is written."""
        client, db_conn, _store = authed_client
        rows_before = db_conn.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]

        with pytest.raises(ValueError, match='Invalid turn_id specified'):
            client.post('/api/thread/999999', data={'text': 'reply into nothing'})

        rows_after = db_conn.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
        assert rows_after == rows_before

    def test_thread_interrupt_idle_returns_no_active_turn(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """DELETE /api/thread/<turn_id> with no turn in flight returns 200 and says so."""
        client, _db_conn, _store = authed_client

        resp = client.delete('/api/thread/1')

        assert resp.status_code == 200
        assert resp.get_json() == {'ok': True, 'interrupted': None, 'reason': 'no_active_turn'}
