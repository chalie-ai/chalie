"""Feature tests for SubagentAbility input validation, dispatch modes, and
envelope format.

Covers: prompt/agent_type/wait validation, envelope format (success + failure),
fire-and-forget ack shape, and the action-envelope reserved-key collision guard.

Per-type wiring (native_tools, deadline) lives in test_subagent_processor.py.
End-to-end nightly scenarios (037, 077, 110) own the full ACT-loop path.
"""

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _execute(params: dict) -> dict:
    from abilities.subagent import SubagentAbility
    return SubagentAbility().execute(channel="user", params=params, telemetry=None)


def _is_error_tag(text: str, error_value: str) -> bool:
    """Return True if text is a [subagent(...)] block containing error=<value>."""
    return (
        text.startswith("[subagent(")
        and f"error={error_value}" in text
        and "[end:subagent]" in text
    )


# ---------------------------------------------------------------------------
# A2–A3. Validation error paths
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_empty_prompt_returns_error_tag(self):
        """Empty prompt and whitespace-only prompt both produce prompt-required error."""
        result = _execute({"prompt": ""})
        text = result["text"]
        assert _is_error_tag(text, "prompt-required"), repr(text)

        result2 = _execute({"prompt": "   "})
        text2 = result2["text"]
        assert _is_error_tag(text2, "prompt-required"), repr(text2)

    def test_invalid_agent_type_returns_error_tag(self):
        """agent_type='nonsense' must return [subagent(error=invalid-type:nonsense)]."""
        result = _execute({"prompt": "do something", "agent_type": "nonsense"})
        text = result["text"]
        assert text.startswith("[subagent("), repr(text)
        assert "error=invalid-type:nonsense" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)


# ---------------------------------------------------------------------------
# A4. Default type is general_purpose / fire-and-forget ack shape
# ---------------------------------------------------------------------------


class TestAsyncDispatch:
    def test_wait_false_returns_ack_with_correct_agent_type(self):
        """Fire-and-forget without agent_type defaults to general_purpose.
        The synchronous ack must not carry an error and must record the
        agent_type in the JSON body."""
        result = _execute({"prompt": "research X", "wait": False})
        text = result["text"]

        assert text.startswith("[subagent("), repr(text)
        assert "error=" not in text, repr(text)
        assert "wait=False" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)

        lines = text.splitlines()
        body = json.loads(lines[1])
        assert body["success"] is True
        assert "sub_id" in body and len(body["sub_id"]) > 0
        assert body["type"] == "general_purpose"
        assert "Working on it" in body["response"]


# ---------------------------------------------------------------------------
# A8. Envelope format — success
# ---------------------------------------------------------------------------


class TestEnvelopeFormatSuccess:
    def test_envelope_format_success(self):
        """_build_envelope with status='success' must match the canonical format:
        starts with [subagent.complete(type=<type>)], contains
        'Subagent's response:', the body text, ends with [end:subagent.complete]."""
        from abilities.subagent import _build_envelope

        body = "Found 3 relevant sources. Summary follows."
        result = _build_envelope(body, "web_surfer", status="success")

        assert result.startswith("[subagent.complete(type=web_surfer)]"), repr(result)
        assert "Subagent's response:" in result, repr(result)
        assert body in result, repr(result)
        assert result.rstrip().endswith("[end:subagent.complete]"), repr(result)
        assert "status=failure" not in result, repr(result)


# ---------------------------------------------------------------------------
# A9. Envelope format — failure
# ---------------------------------------------------------------------------


class TestEnvelopeFormatFailure:
    def test_envelope_format_failure(self):
        """_build_envelope with status='failure' must include status=failure,
        'Reason:', and the suggestion to 'retry, escalate, or fall back'."""
        from abilities.subagent import _build_envelope

        reason = "Connection timed out after 3 attempts"
        result = _build_envelope(reason, "web_surfer", status="failure")

        assert "status=failure" in result, repr(result)
        assert "Reason:" in result, repr(result)
        assert reason in result, repr(result)
        assert (
            "retry" in result.lower()
            or "escalate" in result.lower()
            or "fall back" in result.lower()
        ), repr(result)
        assert result.rstrip().endswith("[end:subagent.complete]"), repr(result)


# ---------------------------------------------------------------------------
# A10. Wait-cap / dispatch-TIMEOUT consistency
#
# wait=true blocks the parent ACT iteration up to SUBAGENT_TYPES[t]['wait_cap']
# seconds. The dispatcher then enforces SubagentAbility.TIMEOUT on top of
# that. If wait_cap > TIMEOUT, a legitimate sync run gets killed by the
# dispatcher with the generic "Action exceeded Ns timeout" string instead of
# returning the canonical [subagent(...)] tag. This is the kind of latent
# desync that bites months after a one-line cap bump.
# ---------------------------------------------------------------------------


class TestWaitCapDispatchTimeoutConsistency:
    def test_dispatch_timeout_covers_largest_wait_cap(self):
        """SubagentAbility.TIMEOUT must be >= max(wait_cap) across all
        SUBAGENT_TYPES (with a small buffer for envelope render). Bumping a
        per-type wait_cap without bumping TIMEOUT in lock-step would let the
        dispatcher kill a legitimate sync run before it returns."""
        from abilities.subagent import SUBAGENT_TYPES, SubagentAbility

        max_wait_cap = max(t["wait_cap"] for t in SUBAGENT_TYPES.values())
        assert SubagentAbility.TIMEOUT >= max_wait_cap, (
            f"SubagentAbility.TIMEOUT={SubagentAbility.TIMEOUT}s is below the "
            f"largest wait_cap={max_wait_cap}s. The dispatcher will kill any "
            "wait=true call that uses the larger cap before the SubagentProcessor "
            "finishes. Bump TIMEOUT in lock-step with wait_cap."
        )

    def test_every_wait_cap_is_at_most_default_timeout(self):
        """wait_cap must never exceed default_timeout — the SubagentProcessor's
        own deadline is set from min(default_timeout, override) and a wait_cap
        higher than default_timeout would be unreachable in practice and
        misleading in the docs."""
        from abilities.subagent import SUBAGENT_TYPES

        for agent_type, entry in SUBAGENT_TYPES.items():
            assert entry["wait_cap"] <= entry["default_timeout"], (
                f"{agent_type}: wait_cap={entry['wait_cap']}s exceeds "
                f"default_timeout={entry['default_timeout']}s — the wait_cap "
                "would be silently ignored."
            )


# ---------------------------------------------------------------------------
# AC1. Action-envelope collision guard
#
# Regression for the incident on 2026-05-02 where INPUT_SCHEMA declared a
# 'type' property that overwrote the dispatcher's action-envelope 'type' key
# via Python dict spread in handleTool(). The model emitting
# {name: "subagent", input: {prompt: ..., type: "web_surfer", wait: true}}
# caused the dispatcher to receive action_type='web_surfer' instead of
# 'subagent', producing "Unknown action type: web_surfer".
#
# Fix: flip spread order in handleTool() so 'type' is set AFTER **tc_input,
# and rename the offending INPUT_SCHEMA property from 'type' to 'agent_type'.
#
# These tests verify both layers:
# 1. Schema layer: no INPUT_SCHEMA for any registered ability has a top-level
#    property named 'type' or 'exchange_id'.
# 2. Dispatch layer: handleTool() correctly routes a subagent call whose
#    'input' contains a colliding 'type' key to the subagent handler.
# ---------------------------------------------------------------------------


DISPATCHER_RESERVED_KEYS = {"type", "exchange_id"}


class TestSchemaCollisionPrevention:
    def test_no_ability_schema_property_collides_with_dispatcher_reserved_keys(self):
        """Every registered ability's INPUT_SCHEMA top-level property names must
        not overlap with the dispatcher's reserved envelope keys ('type',
        'exchange_id'). A collision would allow a model-emitted argument to
        silently overwrite the dispatch routing key.

        This test is the systemic latch: it fails whenever a future schema
        re-introduces the bug class — before any nightly run burns tokens."""
        from abilities._registry import AbilityRegistry

        violations = []
        for ability in AbilityRegistry.all():
            schema = ability.INPUT_SCHEMA
            props = set(schema.get("properties", {}).keys())
            collisions = props & DISPATCHER_RESERVED_KEYS
            if collisions:
                violations.append(
                    f"{ability.NAME}: schema properties {sorted(collisions)} "
                    f"collide with dispatcher reserved keys"
                )

        assert not violations, (
            "Ability INPUT_SCHEMA properties must not use dispatcher reserved keys "
            f"({sorted(DISPATCHER_RESERVED_KEYS)}). Violations:\n"
            + "\n".join(violations)
        )


class TestHandleToolDispatchCollisionGuard:
    def test_handle_tool_routes_subagent_even_when_input_contains_type_key(self):
        """handleTool() must dispatch to the 'subagent' handler even when the
        LLM-emitted tool call input dict contains a key literally named 'type'
        (the pre-fix collision vector).

        Observable contract: the result string must NOT start with
        'Unknown action type' and must start with '[subagent(' (the ack tag).

        wait=False is used so the ack is returned synchronously without
        invoking any LLM. The SubagentProcessor daemon thread is spawned and
        discarded — we assert only on the synchronous return value."""
        from services.act_dispatcher_service import ActDispatcherService
        from services.message_processor import MessageProcessor

        # Minimal concrete processor — skips transcript writes so no DB needed.
        class _BareProcessor(MessageProcessor):
            CHANNEL = "test"
            ROLE = "test"
            SKIP_TRANSCRIPT_WRITE = True

            def getUserPrompt(self):
                return "test"

            def getUserDefinition(self):
                return "test user"

        proc = _BareProcessor(raw_input="test")
        proc._dispatcher = ActDispatcherService()

        # This is the exact shape the model emitted during the 2026-05-02
        # incident: input contains 'type' = 'web_surfer' (old field name).
        # With the fix in place the spread order guarantees the envelope
        # 'type' = 'subagent' wins; without the fix it was overwritten and
        # the dispatcher saw 'web_surfer'.
        tc = {
            "name": "subagent",
            "id": "test_collision_01",
            "input": {
                "prompt": "research something",
                "type": "web_surfer",   # old colliding key — pre-fix field name
                "wait": False,
            },
        }

        result_text = proc.handleTool(tc)

        assert not result_text.startswith("Unknown action type"), (
            f"Dispatcher received wrong action_type — collision guard failed. "
            f"result={result_text!r}"
        )
        assert result_text.startswith("[subagent("), (
            f"Expected ack tag starting with '[subagent(', got: {result_text!r}"
        )


# ---------------------------------------------------------------------------
# AD1. _deliver_envelope discriminator — turn_active Event
#
# Regression for the fire-and-forget delivery bug: _deliver_envelope used to
# check isinstance(parent_ref, UserMessageProcessor) which always returned True
# even after the parent's turn was over (it's just a Python object reference).
# The result was silently appended to a dead processor's _pending_steers and
# never reached the user. Fix: check parent_ref._turn_active.is_set().
# ---------------------------------------------------------------------------


class TestDeliverEnvelopeDiscriminator:
    def test_case_a_delivers_to_pending_steers_when_parent_active(self):
        """When parent _turn_active is set, envelope goes to _pending_steers."""
        import threading
        from abilities.subagent import _deliver_envelope
        from services.user_message_processor import UserMessageProcessor

        envelope = "[subagent.complete(type=web_surfer)]\nTest result\n[end:subagent.complete]"

        # Build a real UMP instance (SKIP_TRANSCRIPT_WRITE avoids DB)
        parent = UserMessageProcessor.__new__(UserMessageProcessor)
        parent._pending_steers = []
        parent._turn_active = threading.Event()
        parent._turn_active.set()
        parent._current_iteration = 3

        _deliver_envelope(envelope, parent)

        assert envelope in parent._pending_steers

    def test_case_b_fires_when_parent_turn_finished(self):
        """When parent _turn_active is cleared, envelope routes to Case B."""
        import threading
        from abilities.subagent import _deliver_envelope
        from services.user_message_processor import UserMessageProcessor
        from unittest.mock import patch

        envelope = "[subagent.complete(type=web_surfer)]\nDone\n[end:subagent.complete]"

        parent = UserMessageProcessor.__new__(UserMessageProcessor)
        parent._pending_steers = []
        parent._turn_active = threading.Event()
        parent._turn_active.clear()  # turn is over
        parent._current_iteration = 5

        with patch('abilities.subagent._spawn_return_processor') as mock_spawn:
            _deliver_envelope(envelope, parent)

        mock_spawn.assert_called_once_with(envelope)
        assert envelope not in parent._pending_steers

    def test_case_b_fires_when_parent_is_none(self):
        """When parent_ref is None (no UMP), envelope routes to Case B."""
        from abilities.subagent import _deliver_envelope
        from unittest.mock import patch

        envelope = "[subagent.complete(type=web_surfer)]\nDone\n[end:subagent.complete]"

        with patch('abilities.subagent._spawn_return_processor') as mock_spawn:
            _deliver_envelope(envelope, None)

        mock_spawn.assert_called_once_with(envelope)


# ---------------------------------------------------------------------------
# AD2. _spawn_return_processor wires OutputService
#
# Regression: _spawn_return_processor used to call
# SubagentReturnProcessor.send() and discard the return value. The response
# never reached the WebSocket. Fix: call OutputService.enqueue_proactive()
# after send() — same pattern as scheduler_service.py.
# ---------------------------------------------------------------------------


class TestSpawnReturnProcessorOutput:
    def test_enqueue_proactive_called_with_response(self):
        """_spawn_return_processor must call OutputService.enqueue_proactive()
        with the response text after SubagentReturnProcessor.send() returns."""
        from unittest.mock import patch, MagicMock
        import threading
        from abilities.subagent import _spawn_return_processor

        envelope = "[subagent.complete(type=web_surfer)]\nResult text\n[end:subagent.complete]"

        mock_output_svc = MagicMock()
        mock_proc_instance = MagicMock()
        mock_proc_instance.send.return_value = "Here is what I found from the subagent."

        with patch(
            'services.user_message_processor.SubagentReturnProcessor',
            return_value=mock_proc_instance,
        ), patch(
            'services.output_service.OutputService',
            return_value=mock_output_svc,
        ):
            _spawn_return_processor(envelope)

            # Join inside the patch context so the thread resolves mocked imports
            for t in threading.enumerate():
                if t.name == "subagent-return":
                    t.join(timeout=5)

            mock_proc_instance.send.assert_called_once()
            mock_output_svc.enqueue_proactive.assert_called_once_with(
                topic='user',
                response="Here is what I found from the subagent.",
                source='subagent_return',
            )

    def test_enqueue_proactive_not_called_on_empty_response(self):
        """If SubagentReturnProcessor.send() returns empty, do not publish."""
        from unittest.mock import patch, MagicMock
        import threading
        from abilities.subagent import _spawn_return_processor

        envelope = "[subagent.complete(type=web_surfer)]\n\n[end:subagent.complete]"

        mock_output_svc = MagicMock()
        mock_proc_instance = MagicMock()
        mock_proc_instance.send.return_value = ""

        with patch(
            'services.user_message_processor.SubagentReturnProcessor',
            return_value=mock_proc_instance,
        ), patch(
            'services.output_service.OutputService',
            return_value=mock_output_svc,
        ):
            _spawn_return_processor(envelope)

            for t in threading.enumerate():
                if t.name == "subagent-return":
                    t.join(timeout=5)

            mock_output_svc.enqueue_proactive.assert_not_called()


# ---------------------------------------------------------------------------
# SubagentReturnProcessor isolation contract
# ---------------------------------------------------------------------------


class TestSubagentReturnProcessorIsolation:
    def test_skip_input_row_is_true(self):
        from services.user_message_processor import SubagentReturnProcessor
        assert SubagentReturnProcessor.SKIP_INPUT_ROW is True

    def test_channel_is_user(self):
        from services.user_message_processor import SubagentReturnProcessor
        assert SubagentReturnProcessor.CHANNEL == 'user'

    def test_skip_transcript_write_is_false(self):
        from services.user_message_processor import SubagentReturnProcessor
        assert SubagentReturnProcessor.SKIP_TRANSCRIPT_WRITE is False
