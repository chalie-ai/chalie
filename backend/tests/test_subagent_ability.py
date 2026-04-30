"""Feature tests for SubagentAbility input validation, dispatch modes, and
envelope format.

Covers: INPUT_SCHEMA shape, prompt/type/wait validation, per-type native tool
surface, per-type deadline, wait-cap semantics, envelope format (success +
failure), and fire-and-forget ack shape. The blocking-wait path is in
test_subagent_wait.py. End-to-end nightly scenarios (037, 077, 110) own the
full ACT-loop path.
"""

import json
import time

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
# A1. INPUT_SCHEMA shape
# ---------------------------------------------------------------------------


class TestInputSchema:
    def test_input_schema_shape_matches_spec(self):
        """INPUT_SCHEMA must have prompt (required), type (default general_purpose),
        wait (default false). No timeout field."""
        from abilities.subagent import SubagentAbility

        schema = SubagentAbility.INPUT_SCHEMA
        assert schema["type"] == "object"
        props = schema["properties"]

        assert "prompt" in props
        assert props["prompt"]["type"] == "string"

        assert "type" in props
        assert props["type"]["type"] == "string"
        assert props["type"]["default"] == "general_purpose"
        assert set(props["type"]["enum"]) == {"web_surfer", "summariser", "general_purpose"}

        assert "wait" in props
        assert props["wait"]["type"] == "boolean"
        assert props["wait"]["default"] is False

        # timeout parameter must be gone
        assert "timeout" not in props

        assert schema["required"] == ["prompt"]


# ---------------------------------------------------------------------------
# A2–A3. Validation error paths
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_empty_prompt_returns_error_tag(self):
        result = _execute({"prompt": ""})
        text = result["text"]
        assert _is_error_tag(text, "prompt-required"), repr(text)

    def test_whitespace_only_prompt_returns_error_tag(self):
        result = _execute({"prompt": "   "})
        text = result["text"]
        assert _is_error_tag(text, "prompt-required"), repr(text)

    def test_invalid_type_returns_error_tag(self):
        """type='nonsense' must return [subagent(error=invalid-type:nonsense)]."""
        result = _execute({"prompt": "do something", "type": "nonsense"})
        text = result["text"]
        assert text.startswith("[subagent("), repr(text)
        assert "error=invalid-type:nonsense" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)


# ---------------------------------------------------------------------------
# A4. Default type is general_purpose
# ---------------------------------------------------------------------------


class TestDefaultType:
    def test_default_type_is_general_purpose(self):
        """Omitting 'type' defaults to general_purpose — ack must not contain error."""
        t0 = time.monotonic()
        result = _execute({"prompt": "do something", "wait": False})
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"execute() blocked for {elapsed:.2f}s"
        text = result["text"]
        assert "error=" not in text, repr(text)
        assert text.startswith("[subagent("), repr(text)
        # The ack body contains the agent type
        assert "general_purpose" in text, repr(text)


# ---------------------------------------------------------------------------
# A5. Per-type native tool surface
# ---------------------------------------------------------------------------


class TestPerTypeNativeTools:
    def test_each_type_has_correct_native_tools(self):
        """Instantiating SubagentProcessor with each agent_type sets
        ALWAYS_AVAILABLE to that type's native_tools list from SUBAGENT_TYPES."""
        from abilities.subagent import SUBAGENT_TYPES
        from services.subagent_processor import SubagentProcessor

        for agent_type, entry in SUBAGENT_TYPES.items():
            proc = SubagentProcessor(raw_input="x", agent_type=agent_type)
            assert proc.ALWAYS_AVAILABLE == entry["native_tools"], (
                f"{agent_type}: expected {entry['native_tools']}, "
                f"got {proc.ALWAYS_AVAILABLE}"
            )


# ---------------------------------------------------------------------------
# A6. Per-type deadline
# ---------------------------------------------------------------------------


class TestPerTypeDeadline:
    def test_each_type_has_correct_default_timeout(self):
        """Instantiating SubagentProcessor with each agent_type sets _deadline
        to approximately time.time() + default_timeout (within 1s tolerance)."""
        from abilities.subagent import SUBAGENT_TYPES
        from services.subagent_processor import SubagentProcessor

        for agent_type, entry in SUBAGENT_TYPES.items():
            t_before = time.time()
            proc = SubagentProcessor(raw_input="x", agent_type=agent_type)

            expected_timeout = entry["default_timeout"]
            delta = proc._deadline - t_before

            assert abs(delta - expected_timeout) <= 1.0, (
                f"{agent_type}: expected deadline ~{expected_timeout}s from now, "
                f"got delta={delta:.3f}s (t_before={t_before:.3f}, "
                f"deadline={proc._deadline:.3f})"
            )
            # Also verify the deadline is in the future at construction time
            assert proc._deadline >= t_before + expected_timeout - 1, (
                f"{agent_type}: deadline {proc._deadline:.3f} < t_before + timeout - 1"
            )


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
        # Must NOT contain status=failure
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
        assert "retry" in result.lower() or "escalate" in result.lower() or "fall back" in result.lower(), \
            repr(result)
        assert result.rstrip().endswith("[end:subagent.complete]"), repr(result)


# ---------------------------------------------------------------------------
# A10. Fire-and-forget (wait=False)
# ---------------------------------------------------------------------------


class TestAsyncDispatch:
    def test_wait_false_returns_ack_immediately_without_blocking(self):
        """Fire-and-forget must return in <1 s and carry the ack JSON payload."""
        t0 = time.monotonic()
        result = _execute({"prompt": "research X", "wait": False})
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"execute() blocked for {elapsed:.2f}s (expected <1s)"

        text = result["text"]
        assert text.startswith("[subagent("), repr(text)
        assert "wait=False" in text, repr(text)
        assert "[end:subagent]" in text, repr(text)

        # Body is the JSON ack
        lines = text.splitlines()
        # opener / body / [end:subagent]
        assert len(lines) >= 3, repr(text)
        body_line = lines[1]
        body = json.loads(body_line)
        assert body["success"] is True
        assert "sub_id" in body
        assert isinstance(body["sub_id"], str) and len(body["sub_id"]) > 0
        assert "Working on it" in body["response"]
        # 'type' must be present in ack
        assert "type" in body

        # Give the daemon thread a moment to start without joining it
        time.sleep(0.2)
