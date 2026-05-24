"""Tests for TKT-598: cooperative ACT loop stop."""

import pytest

pytestmark = pytest.mark.unit


# ── _iteration_cap_reached() ─────────────────────────────────────────────────


class TestIterationCap:

    def test_ump_unbounded(self):
        from services.user_message_processor import UserMessageProcessor
        proc = UserMessageProcessor(raw_input="test")
        proc._current_iteration = 200
        assert proc._iteration_cap_reached() is False

    def test_subagent_processor_unbounded(self):
        from services.subagent_processor import SubagentProcessor
        proc = SubagentProcessor(raw_input="test", agent_type="general_purpose")
        proc._current_iteration = 200
        assert proc._iteration_cap_reached() is False

    def test_background_processor_caps_at_max(self):
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt

        class _BgProc(MessageProcessor):
            CHANNEL = "dmn"
            ROLE = "proactive_thought"
            SKIP_TRANSCRIPT_WRITE = True
            SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt
            MAX_ITERATIONS = 5
            def get_user_prompt(self): return "test"
            def get_user_definition(self): return "test"

        proc = _BgProc(raw_input="test")
        proc._current_iteration = 4
        assert proc._iteration_cap_reached() is False
        proc._current_iteration = 5
        assert proc._iteration_cap_reached() is True


# ── Subagent registry ────────────────────────────────────────────────────────


class TestSubagentRegistry:

    def test_cancel_unknown_returns_false(self):
        from abilities.subagent import cancel_subagent
        assert cancel_subagent("nonexistent-id") is False

    def test_get_active_returns_dict(self):
        from abilities.subagent import get_active_subagents
        result = get_active_subagents()
        assert isinstance(result, dict)


# ── _params_update pipeline ──────────────────────────────────────────────────


class TestParamsUpdate:

    def test_propagated_through_dispatcher(self, db):
        from services.act_dispatcher_service import ActDispatcherService

        dispatcher = ActDispatcherService(timeout=5.0)
        dispatcher.handlers["subagent"] = lambda ch, act: {
            "text": "ack",
            "_params_update": {"sub_id": "abc123"},
        }

        result = dispatcher.dispatch_action("user", {"type": "subagent", "prompt": "test"})
        assert result["status"] == "success"
        assert result["_params_update"]["sub_id"] == "abc123"


# ── HTTP stop endpoints ─────────────────────────────────────────────────────


class TestStopEndpoints:

    def test_stop_no_active_turn(self, authed_client):
        client, _db, _store = authed_client
        resp = client.post("/chat/stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["reason"] == "no_active_turn"

    def test_stop_unknown_subagent(self, authed_client):
        client, _db, _store = authed_client
        resp = client.post("/chat/subagent/nonexistent/stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["reason"] == "not_found"
