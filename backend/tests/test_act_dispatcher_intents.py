"""
Tests for the wrapper-intent routing path in ActDispatcherService.

Verifies that when an action_type has no innate-skill handler, the dispatcher
checks for a capable wrapper and emits a CognitiveIntent via IntentService.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.act_dispatcher_service import ActDispatcherService

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """ActDispatcherService with innate-skill registration suppressed."""
    with patch("services.innate_skills.register_innate_skills"):
        svc = ActDispatcherService(timeout=2.0)
        yield svc


def _make_wrapper(wrapper_id="wrp_ide", intent_types=None):
    """Build a minimal wrapper dict as WrapperAuthService.list_wrappers() returns."""
    return {
        "wrapper_id": wrapper_id,
        "name": "Test Wrapper",
        "capabilities": {
            "intents": intent_types or ["open_pr", "run_command"],
            "signals": [],
        },
        "permissions": {},
        "metadata": {},
        "last_seen_at": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Wrapper intent routing: happy path
# ---------------------------------------------------------------------------

class TestWrapperIntentRouting:

    def test_unknown_action_routed_to_capable_wrapper(self, service, db):
        """When a wrapper declares the action type, an intent is emitted."""
        wrapper = _make_wrapper(intent_types=["open_pr"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        mock_intent_svc = MagicMock()

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc), \
             patch("services.intent_service.IntentService", return_value=mock_intent_svc):

            result = service.dispatch_action("topic", {"type": "open_pr", "params": {"branch": "feature/x"}})

        assert result["status"] == "intent_emitted"
        assert result["action_type"] == "open_pr"
        assert result["target_wrapper"] == "wrp_ide"
        assert "intent_id" in result
        assert result["confidence"] > 0

    def test_intent_emitted_contains_params(self, service, db):
        """The emitted CognitiveIntent payload includes action params."""
        wrapper = _make_wrapper(intent_types=["deploy"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        captured_intents = []

        def capture_emit(intent):
            captured_intents.append(intent)

        mock_intent_svc = MagicMock()
        mock_intent_svc.emit.side_effect = capture_emit

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc), \
             patch("services.intent_service.IntentService", return_value=mock_intent_svc):

            service.dispatch_action("topic", {
                "type": "deploy",
                "params": {"env": "staging"},
                "description": "Deploy to staging",
            })

        assert len(captured_intents) == 1
        intent = captured_intents[0]
        assert intent.intent_type == "execute"
        assert intent.payload["action_type"] == "deploy"
        assert intent.payload["params"]["env"] == "staging"
        assert intent.target_wrapper == "wrp_ide"

    def test_result_notes_contain_intent_id_and_target(self, service, db):
        """Result notes mention the intent_id and target wrapper."""
        wrapper = _make_wrapper(intent_types=["run_command"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        mock_intent_svc = MagicMock()

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc), \
             patch("services.intent_service.IntentService", return_value=mock_intent_svc):

            result = service.dispatch_action("topic", {"type": "run_command"})

        assert "wrp_ide" in result["notes"]
        assert "intent_id" in result["notes"]

    def test_first_capable_wrapper_is_selected(self, service, db):
        """When multiple wrappers declare the capability, the first is used."""
        w1 = _make_wrapper(wrapper_id="wrp_first", intent_types=["git_push"])
        w2 = _make_wrapper(wrapper_id="wrp_second", intent_types=["git_push"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [w1, w2]

        mock_intent_svc = MagicMock()

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc), \
             patch("services.intent_service.IntentService", return_value=mock_intent_svc):

            result = service.dispatch_action("topic", {"type": "git_push"})

        assert result["target_wrapper"] == "wrp_first"


# ---------------------------------------------------------------------------
# Wrapper intent routing: no capable wrapper
# ---------------------------------------------------------------------------

class TestNoCapableWrapper:

    def test_no_wrapper_falls_through_to_error(self, service, db):
        """When no wrapper handles the action, the error path is reached."""
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = []

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc):

            result = service.dispatch_action("topic", {"type": "no_such_action"})

        assert result["status"] == "error"
        assert "Unknown action type" in result["result"]

    def test_wrapper_without_matching_capability_falls_through(self, service, db):
        """A wrapper with different capabilities doesn't intercept unrelated actions."""
        wrapper = _make_wrapper(intent_types=["open_pr"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc):

            result = service.dispatch_action("topic", {"type": "completely_different"})

        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Wrapper intent routing: innate skill takes priority
# ---------------------------------------------------------------------------

class TestInnateSkillPriority:

    def test_innate_skill_handler_takes_priority_over_wrapper(self, service, db):
        """A registered innate-skill handler is always preferred over wrapper intents."""
        # Register a handler directly
        service.handlers["my_action"] = lambda topic, action: "skill result"

        wrapper = _make_wrapper(intent_types=["my_action"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc):

            result = service.dispatch_action("topic", {"type": "my_action"})

        # Should have used the innate skill, not the wrapper intent
        assert result["status"] == "success"
        assert result["result"] == "skill result"


# ---------------------------------------------------------------------------
# Wrapper intent routing: exception resilience
# ---------------------------------------------------------------------------

class TestExceptionResilience:

    def test_wrapper_svc_exception_falls_through_to_error(self, service, db):
        """If WrapperAuthService raises, the wrapper intent path returns None (no crash)."""
        with patch("services.wrapper_auth_service.WrapperAuthService", side_effect=RuntimeError("db down")):

            result = service.dispatch_action("topic", {"type": "exotic_action"})

        # Graceful degradation — falls through to "Unknown action type" error
        assert result["status"] == "error"

    def test_intent_emit_exception_falls_through(self, service, db):
        """If IntentService.emit raises, the wrapper intent path returns None."""
        wrapper = _make_wrapper(intent_types=["risky_action"])
        mock_wrapper_svc = MagicMock()
        mock_wrapper_svc.list_wrappers.return_value = [wrapper]

        mock_intent_svc = MagicMock()
        mock_intent_svc.emit.side_effect = RuntimeError("store unavailable")

        with patch("services.wrapper_auth_service.WrapperAuthService", return_value=mock_wrapper_svc), \
             patch("services.intent_service.IntentService", return_value=mock_intent_svc):

            result = service.dispatch_action("topic", {"type": "risky_action"})

        # Exception in wrapper path → falls through to error
        assert result["status"] == "error"
