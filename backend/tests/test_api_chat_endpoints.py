"""Feature tests for the chat API HTTP contracts.

Drives the real Flask app against a real SQLite database via authed_client.
Pins synchronous endpoint contracts only; the async full-turn path (real UMP + LLM) is out of scope.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient


@pytest.mark.unit
class TestChatEndpoints:

    def test_thread_empty_message_rejected_and_no_turn_started(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """An empty POST /api/thread fails loud with 400/message required and must
        NOT register a turn — a follow-up interrupt finds nothing in flight."""
        client, _db_conn, _store = authed_client

        resp = client.post('/api/thread', data={'text': '   '})

        assert resp.status_code == 400
        data = resp.get_json()
        assert data['status'] == 'error'
        assert data['reason'] == 'message required'

        # Cross-step proof: no UMP turn was registered for the empty message.
        interrupt = client.post('/chat/interrupt')
        assert interrupt.status_code == 200
        assert interrupt.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_chat_interrupt_idle_returns_no_active_turn(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /chat/interrupt with no turn in flight returns 200 and says so."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat/interrupt')

        assert resp.status_code == 200
        assert resp.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_action_missing_skill_rejected(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /action without a skill is rejected synchronously with 400."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={})

        assert resp.status_code == 400
        assert 'skill' in resp.get_json()['error']

    def test_action_unknown_skill_rejected_synchronously(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """An unknown skill resolves synchronously to 400 in the HTTP body — the
        result never crosses the WS bus."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={'skill': 'no-such-tool-xyz'})

        assert resp.status_code == 400
        assert 'Unknown skill' in resp.get_json()['error']

    def test_action_real_skill_runs_synchronously_and_does_not_persist(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """A rich-card action runs the real skill against the real DB and returns
        its result inline: the body carries content/mode/duration, the list row is
        really written, and NO transcript row is created — the action is a silent
        mutation, not a conversation turn, and no data crosses the WS bus."""
        client, db_conn, _store = authed_client
        transcripts_before = db_conn.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]

        resp = client.post('/action', json={'skill': 'list', 'action': 'create', 'name': 'QA Groceries List'})

        assert resp.status_code == 200
        body = resp.get_json()
        assert body['content']
        assert body['mode'] == 'ACT'
        assert isinstance(body['duration_ms'], int)

        # Downstream proof the real skill ran end-to-end against the real DB.
        row = db_conn.execute("SELECT name FROM lists WHERE name = ?", ('QA Groceries List',)).fetchone()
        assert row is not None and row[0] == 'QA Groceries List'

        # Silent mutation: a card action must NOT write a transcript turn.
        transcripts_after = db_conn.execute("SELECT COUNT(*) FROM transcript").fetchone()[0]
        assert transcripts_after == transcripts_before
