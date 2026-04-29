"""Feature tests for SubagentAbility input validation and dispatch modes.

Covers: INPUT_SCHEMA shape, prompt/timeout/wait validation, clamp to ceiling,
and fire-and-forget ack shape. The blocking-wait path is in test_subagent_wait.py.
End-to-end nightly scenarios (037, 077, 110) own the full ACT-loop path.
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
        from abilities.subagent import SubagentAbility

        schema = SubagentAbility.INPUT_SCHEMA
        assert schema["type"] == "object"
        props = schema["properties"]

        assert "prompt" in props
        assert props["prompt"]["type"] == "string"

        assert "timeout" in props
        assert props["timeout"]["type"] == "integer"
        assert props["timeout"]["default"] == 1800
        assert props["timeout"]["minimum"] == 180

        assert "wait" in props
        assert props["wait"]["type"] == "boolean"
        assert props["wait"]["default"] is False

        assert schema["required"] == ["prompt"]


# ---------------------------------------------------------------------------
# A2–A4. Validation error paths
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

    def test_timeout_below_minimum_returns_error_tag(self):
        result = _execute({"prompt": "x", "timeout": 120})
        text = result["text"]
        assert _is_error_tag(text, "timeout-below-min-180s"), repr(text)

    def test_wait_true_with_timeout_above_300_returns_error_tag(self):
        result = _execute({"prompt": "x", "wait": True, "timeout": 600})
        text = result["text"]
        assert _is_error_tag(text, "wait-true-requires-timeout-le-300s"), repr(text)


# ---------------------------------------------------------------------------
# A5. Timeout clamp
# ---------------------------------------------------------------------------


class TestTimeoutClamp:
    def test_timeout_clamped_to_max_when_above_7200(self):
        """timeout=99999 must not raise an error — clamping is internal.

        Observable behavior: execute() returns an ack tag (not an error tag),
        proving the >7200 value was clamped rather than rejected.
        """
        result = _execute({"prompt": "research X", "timeout": 99999, "wait": False})
        text = result["text"]
        # Must NOT be an error tag
        assert "error=" not in text, f"Expected ack, got error: {repr(text)}"
        # Must be a subagent ack block
        assert text.startswith("[subagent("), repr(text)
        assert "[end:subagent]" in text, repr(text)


# ---------------------------------------------------------------------------
# A6. Fire-and-forget (wait=False)
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

        # Give the daemon thread a moment to start without joining it
        time.sleep(0.2)
