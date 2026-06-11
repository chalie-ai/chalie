"""Feature tests for api/chat.py HTTP contracts — POST /chat, /chat/interrupt,
/chat/stop, /action.

Drives the real Flask app (create_app via authed_client) against a real
SQLite database. Pins the synchronous endpoint contracts; the async
full-turn path (real UMP + LLM) is out of scope here.
"""

import pytest


@pytest.mark.unit
class TestChatEndpoints:

    def test_chat_empty_message_ignored_and_no_turn_started(self, authed_client):
        """An empty POST /chat is acknowledged with 202/ignored and must NOT
        register a turn — a follow-up interrupt finds nothing in flight."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat', data={'text': '   '})

        assert resp.status_code == 202
        data = resp.get_json()
        assert data['status'] == 'ignored'
        assert data['reason'] == 'empty message'

        # Cross-step proof: no UMP turn was registered for the empty message.
        interrupt = client.post('/chat/interrupt')
        assert interrupt.status_code == 200
        assert interrupt.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_chat_interrupt_idle_returns_no_active_turn(self, authed_client):
        """POST /chat/interrupt with no turn in flight returns 200 and says so."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat/interrupt')

        assert resp.status_code == 200
        assert resp.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_chat_stop_is_alias_for_interrupt(self, authed_client):
        """POST /chat/stop (deprecated alias) exists and handles the idle
        state with the same contract as /chat/interrupt."""
        client, _db_conn, _store = authed_client

        resp = client.post('/chat/stop')

        assert resp.status_code == 200
        assert resp.get_json() == {'ok': True, 'reason': 'no_active_turn'}

    def test_action_missing_skill_rejected(self, authed_client):
        """POST /action without a skill is rejected synchronously with 400."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={})

        assert resp.status_code == 400
        assert 'skill' in resp.get_json()['error']

    def test_action_with_skill_accepted(self, authed_client):
        """POST /action with a skill returns 202 immediately — the result is
        delivered asynchronously via WS, never in the HTTP response body."""
        client, _db_conn, _store = authed_client

        resp = client.post('/action', json={'skill': 'no-such-tool-xyz'})

        assert resp.status_code == 202
        assert resp.get_json() == {'status': 'accepted'}
