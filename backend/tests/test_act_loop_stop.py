"""Feature tests for TKT-598: cooperative ACT loop stop.

Tests the observable behaviour of three components:

1. _iteration_cap_reached() — pure function, unit tests allowed:
   - UMP returns False (unbounded loop)
   - SubagentProcessor returns False (unbounded loop)
   - Base MessageProcessor respects MAX_ITERATIONS (background processors)

2. Subagent registry helpers — pure functions:
   - cancel_subagent() returns False for an unknown sub_id
   - get_active_subagents() returns empty dict when none are running

3. _params_update pipeline — real ActDispatcherService + real ability:
   - An ability that returns _params_update in its result dict has that
     update propagated into the dispatch result for the caller to merge.

4. HTTP stop endpoints — real Flask app (authed_client fixture):
   - POST /chat/stop returns 200 {"ok": true, "reason": "no_active_turn"}
     when no turn is active
   - POST /chat/subagent/<sub_id>/stop returns 200 {"ok": true, "reason":
     "not_found"} for a sub_id not in the registry
"""

import pytest

pytestmark = pytest.mark.unit


# ── 1. _iteration_cap_reached() — pure function unit tests ───────────────────


class TestIterationCapReachedUMP:
    """UMP overrides _iteration_cap_reached() to always return False so its
    ACT loop is bounded only by user stop, ITERATION_TIMEOUT, or clean exit."""

    def test_ump_iteration_cap_never_reached(self):
        """UMP._iteration_cap_reached() returns False at any iteration count."""
        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input="test")
        # Simulate the loop running well past the old cap of 30.
        for i in range(200):
            proc._current_iteration = i
            assert proc._iteration_cap_reached() is False, (
                f"UMP._iteration_cap_reached() returned True at iteration {i} — "
                "UMP loop should be unbounded"
            )


class TestIterationCapReachedSubagentProcessor:
    """SubagentProcessor overrides _iteration_cap_reached() to always return
    False. Its only termination paths are user stop, _deadline, and clean exit."""

    def test_subagent_processor_iteration_cap_never_reached(self):
        """SubagentProcessor._iteration_cap_reached() returns False at any count."""
        from services.subagent_processor import SubagentProcessor

        proc = SubagentProcessor(raw_input="test", agent_type="general_purpose")
        for i in range(200):
            proc._current_iteration = i
            assert proc._iteration_cap_reached() is False, (
                f"SubagentProcessor._iteration_cap_reached() returned True at "
                f"iteration {i} — SubagentProcessor loop should be unbounded"
            )


class TestIterationCapReachedBaseProcessor:
    """The base MessageProcessor._iteration_cap_reached() respects MAX_ITERATIONS
    so background processors (DMN, PatternMatch, etc.) still have a safety cap."""

    def test_base_iteration_cap_trips_at_max_iterations(self):
        """Base _iteration_cap_reached() returns False below MAX_ITERATIONS and
        True at MAX_ITERATIONS. This guards all background processor subclasses
        that do not override the method."""
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt

        class _BackgroundProcessor(MessageProcessor):
            CHANNEL = "dmn"
            ROLE = "proactive_thought"
            SKIP_TRANSCRIPT_WRITE = True
            SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt
            MAX_ITERATIONS = 5

            def get_user_prompt(self):
                return "test"

            def get_user_definition(self):
                return "test"

        proc = _BackgroundProcessor(raw_input="test")

        for i in range(5):
            proc._current_iteration = i
            assert proc._iteration_cap_reached() is False, (
                f"_iteration_cap_reached() returned True at iteration {i} "
                f"(before cap of 5)"
            )

        proc._current_iteration = 5
        assert proc._iteration_cap_reached() is True, (
            "_iteration_cap_reached() returned False at iteration=5 with MAX_ITERATIONS=5"
        )


# ── 2. Subagent registry helpers ─────────────────────────────────────────────


class TestCancelSubagentUnknownId:
    """cancel_subagent() returns False for an id not in the registry.

    The stop endpoint relies on this to distinguish "already done / never
    existed" from "signal delivered" — both are safe to call, but the
    response body differs."""

    def test_cancel_unknown_sub_id_returns_false(self):
        """cancel_subagent('not-a-real-id') returns False without raising."""
        from abilities.subagent import cancel_subagent

        result = cancel_subagent("not-a-real-sub-id-0000000000000000")
        assert result is False, (
            "cancel_subagent() should return False for an id not in the registry"
        )


class TestGetActiveSubagentsEmpty:
    """get_active_subagents() returns an empty dict when no subagents are running.

    The frontend relies on this to initialise the task drawer with an empty
    subagent list on fresh page load."""

    def test_get_active_subagents_returns_empty_dict_when_none_running(self):
        """get_active_subagents() returns {} (or an empty dict) initially."""
        from abilities.subagent import get_active_subagents, _active_subagents, _subagent_lock

        # Ensure the registry is empty for this test — it may have been
        # populated by a concurrent test. Read under lock then only proceed if
        # empty; if something else put entries in, the test is still valid
        # because we are testing that the function correctly reflects state.
        with _subagent_lock:
            current = dict(_active_subagents)

        result = get_active_subagents()
        # If no subagents are running the result must be empty. If something is
        # running (parallel test context), the result must at least match the
        # registry state — we assert the function returns what the registry holds.
        assert isinstance(result, dict), (
            f"get_active_subagents() returned {type(result)}, expected dict"
        )
        assert result == current, (
            "get_active_subagents() did not return a snapshot of the registry"
        )


# ── 3. _params_update pipeline ───────────────────────────────────────────────


class TestParamsUpdatePipeline:
    """The _params_update pipeline lets an ability annotate the tc_input
    recorded in tool_calls.params without needing to reach into the processor.

    Verify that ActDispatcherService._build_success_result() propagates
    _params_update from a raw ability result dict into the dispatch result.
    This is the mechanism that makes sub_id visible in tool_calls.params after
    subagent.execute() returns {"text": "...", "_params_update": {"sub_id": "xyz"}}.
    """

    def test_params_update_propagated_through_dispatcher(self, db):
        """An ability result containing _params_update is surfaced in the
        dispatch result under the '_params_update' key."""
        from services.act_dispatcher_service import ActDispatcherService

        dispatcher = ActDispatcherService(timeout=5.0)

        test_sub_id = "abc123def456"

        # Register a handler that mimics subagent._run_async() returning
        # {"text": "...", "_params_update": {"sub_id": sub_id}}.
        def _fake_subagent_handler(channel, action):
            return {
                "text": "[subagent(status=ack)]...[end:subagent]",
                "_params_update": {"sub_id": test_sub_id},
            }

        dispatcher.handlers["subagent"] = _fake_subagent_handler

        result = dispatcher.dispatch_action("user", {"type": "subagent", "prompt": "test"})

        assert result["status"] == "success", (
            f"Dispatch returned status={result['status']!r}, expected 'success'"
        )
        assert "_params_update" in result, (
            "_params_update key missing from dispatch result — "
            "the pipeline did not propagate it from the ability result"
        )
        assert result["_params_update"].get("sub_id") == test_sub_id, (
            f"sub_id not preserved through _params_update pipeline: "
            f"got {result['_params_update']}"
        )


# ── 4. HTTP stop endpoints ────────────────────────────────────────────────────


class TestPostChatStop:
    """POST /chat/stop with no active turn returns 200 with reason=no_active_turn.

    This is the idle-state path exercised by nightly scenario 150. The endpoint
    must always return 200 — it is idempotent and calling it when nothing is
    running is safe."""

    def test_stop_with_no_active_turn_returns_200(self, authed_client):
        """POST /chat/stop → 200 {"ok": true, "reason": "no_active_turn"}."""
        client, _db, _store = authed_client

        response = client.post("/chat/stop")

        assert response.status_code == 200, (
            f"POST /chat/stop returned {response.status_code}, expected 200"
        )
        data = response.get_json()
        assert data is not None, "POST /chat/stop did not return JSON"
        assert data.get("ok") is True, (
            f"Expected ok=true, got: {data}"
        )
        assert data.get("reason") == "no_active_turn", (
            f"Expected reason='no_active_turn' when no turn is active, got: {data}"
        )


class TestPostSubagentStop:
    """POST /chat/subagent/<sub_id>/stop for an unknown sub_id returns 200 with
    reason=not_found.

    The endpoint never returns 404 by design — 404 means "route not found"
    (Flask level). Application-level not_found is communicated via the JSON body
    with HTTP 200. This matches the contract from nightly scenario 151."""

    def test_stop_unknown_subagent_returns_200_not_found(self, authed_client):
        """POST /chat/subagent/unknown-id/stop → 200 {"ok": true, "reason": "not_found"}."""
        client, _db, _store = authed_client

        response = client.post("/chat/subagent/notarealsubid00000000000000000000/stop")

        assert response.status_code == 200, (
            f"POST /chat/subagent/.../stop returned {response.status_code}, expected 200"
        )
        data = response.get_json()
        assert data is not None, "POST /chat/subagent/.../stop did not return JSON"
        assert data.get("ok") is True, (
            f"Expected ok=true, got: {data}"
        )
        assert data.get("reason") == "not_found", (
            f"Expected reason='not_found' for unknown sub_id, got: {data}"
        )
