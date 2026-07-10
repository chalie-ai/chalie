"""Feature tests for api/chat.py HTTP contracts.

Drives the real Flask app against a real SQLite database via authed_client.
Pins synchronous endpoint contracts only; the async full-turn path (real UMP + LLM) is out of scope.
"""

import sqlite3
from datetime import datetime, timezone

import pytest
from flask.testing import FlaskClient


def _make_unauthed_client() -> FlaskClient:
    """Real, unauthenticated Flask client -- the authed_client fixture patches
    validate_session, which would hide the require_auth guard these tests exist
    to prove (same pattern as test_voice_auth._make_client)."""
    from api import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app.test_client()


@pytest.mark.unit
class TestChatEndpoints:

    def test_chat_empty_message_rejected_and_no_turn_started(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """An empty POST /chat fails loud with 400/message required and must NOT
        register a turn — a follow-up interrupt finds nothing in flight."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat', data={'text': '   '})

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

    def test_chat_stop_is_alias_for_interrupt(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /chat/stop (deprecated alias) exists and handles the idle
        state with the same contract as /chat/interrupt."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat/stop')

        assert resp.status_code == 200
        assert resp.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_action_missing_skill_rejected(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /action without a skill is rejected synchronously with 400."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={})

        assert resp.status_code == 400
        assert 'skill' in resp.get_json()['error']

    def test_action_with_skill_accepted(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /action with a skill returns 202 immediately — the result is
        delivered asynchronously via WS, never in the HTTP response body."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={'skill': 'no-such-tool-xyz'})

        assert resp.status_code == 202
        assert resp.get_json() == {'status': 'accepted'}

    def test_start_turn_records_utc_started_at(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object], monkeypatch: pytest.MonkeyPatch) -> None:
        """_start_turn populates _active_ump with a timezone-aware UTC started_at,
        so /chat/status can report it and the staleness guard can compute age.

        The real background daemon (_run_chat_background -> MessageProcessor +
        LLM) is out of scope here (see module docstring); stubbing it to a no-op
        keeps this test synchronous and prevents it from clearing _active_ump
        before the assertion.
        """
        import api.chat as chat_mod

        fixed = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(chat_mod, 'utc_now', lambda: fixed)
        monkeypatch.setattr(chat_mod, '_run_chat_background', lambda *a, **kw: None)

        client, _db_conn, _store = authed_client
        client.post('/chat', data={'text': 'hello'})

        active = chat_mod._get_active_ump()
        assert active is not None
        assert active.started_at == fixed
        assert active.started_at.tzinfo is not None  # timezone-aware

    def test_chat_status_idle_when_no_turn(self, authed_client) -> None:
        """GET /chat/status with no turn in flight returns in_progress: False."""
        client, _db_conn, _store = authed_client

        resp = client.get('/chat/status')

        assert resp.status_code == 200
        assert resp.get_json() == {'in_progress': False}

    def test_chat_status_in_progress_when_turn_active(self, authed_client, monkeypatch) -> None:
        """GET /chat/status while a turn is in flight returns in_progress: True
        and the timezone-aware ISO started_at."""
        import api.chat as chat_mod

        # Stub the background runner so _active_ump stays populated (module
        # docstring: async full-turn path is out of scope for this file).
        monkeypatch.setattr(chat_mod, '_run_chat_background', lambda *a, **kw: None)

        fixed = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(chat_mod, 'utc_now', lambda: fixed)

        client, _db_conn, _store = authed_client
        client.post('/chat', data={'text': 'hello'})

        resp = client.get('/chat/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['in_progress'] is True
        assert data['started_at'] == '2026-07-10T12:00:00+00:00'

    def test_chat_status_requires_auth(self, db) -> None:
        """GET /chat/status without a session is rejected -- the chat
        blueprint's require_auth gate applies to status too."""
        client = _make_unauthed_client()
        resp = client.get('/chat/status')

        assert resp.status_code == 401
        assert resp.get_json() == {"error": "Authentication required"}

    def test_stale_active_ump_is_cleared_on_read(self, authed_client, monkeypatch) -> None:
        """A turn older than _MAX_TURN_AGE is lazily cleared by _get_active_ump,
        so a wedged registry entry can never block /chat/status forever."""
        from datetime import timedelta
        import api.chat as chat_mod

        # Start a real turn to populate _active_ump, but stub the background
        # runner so it doesn't invoke the UMP/LLM or clear _active_ump.
        monkeypatch.setattr(chat_mod, '_run_chat_background', lambda *a, **kw: None)
        client, _db_conn, _store = authed_client
        client.post('/chat', data={'text': 'hello'})

        # Freeze "now" well past the turn's real started_at, then read.
        started = chat_mod._active_ump.started_at
        future = started + chat_mod._MAX_TURN_AGE + timedelta(minutes=1)
        monkeypatch.setattr(chat_mod, 'utc_now', lambda: future)

        assert chat_mod._get_active_ump() is None  # stale entry cleared
        assert chat_mod._active_ump is None
