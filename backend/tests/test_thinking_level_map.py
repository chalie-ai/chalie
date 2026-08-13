# Unit tests for ThinkingLevel mapping — local, deterministic logic only.
#
# Scope rule: what lives here builds or classifies a payload IN PROCESS. Nothing
# here contacts a provider, and nothing here asserts what a provider does with
# what it is sent — that can only be established by firing at the real thing,
# never by scripting a stand-in whose replies we wrote ourselves.

import pytest

from configs.channels.thread_gist import ThreadGistConfig
from configs.enums.thinking_level import ThinkingLevel


# ---------------------------------------------------------------------------
# 1. Anthropic native thinking flag mapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicThinkingNative:

    def test_all_five_levels_map_correctly(self) -> None:
        from services.llm_clients.anthropic import AnthropicClient

        client = AnthropicClient({"platform": "anthropic", "model": "claude-sonnet-4-20250514"})

        none_result = client._thinking_native(ThinkingLevel.NONE, max_tokens=9000)
        assert none_result == {"thinking": {"type": "disabled"}}

        low_result = client._thinking_native(ThinkingLevel.LOW, max_tokens=9000)
        assert low_result == {}

        med_result = client._thinking_native(ThinkingLevel.MEDIUM, max_tokens=9000)
        assert med_result == {"thinking": {"type": "enabled", "budget_tokens": 4096}}

        high_result = client._thinking_native(ThinkingLevel.HIGH, max_tokens=9000)
        assert high_result == {"thinking": {"type": "enabled", "budget_tokens": 16384}}

        max_result = client._thinking_native(ThinkingLevel.MAX, max_tokens=9000)
        assert max_result == {"thinking": {"type": "enabled", "budget_tokens": 9000}}


# ---------------------------------------------------------------------------
# 2. OpenAI native thinking flag mapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAIThinkingNative:

    def test_all_five_levels_map_correctly(self) -> None:
        from services.llm_clients.openai import OpenAIClient

        client = OpenAIClient({"platform": "openai", "model": "gpt-4o", "api_key": "k"})

        assert client._thinking_native(ThinkingLevel.NONE) == "none"
        assert client._thinking_native(ThinkingLevel.LOW) is None
        assert client._thinking_native(ThinkingLevel.MEDIUM) == "medium"
        assert client._thinking_native(ThinkingLevel.HIGH) == "high"
        assert client._thinking_native(ThinkingLevel.MAX) == "high"


# ---------------------------------------------------------------------------
# 3. Ollama think payload
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaThinkPayload:

    def test_think_flag_per_level_when_supported(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        # The host is never dialled — _build_payload assembles a dict in process.
        client = OllamaClient({"host": "http://127.0.0.1:1", "model": "m"})
        client._show_payload = {"capabilities": ["thinking"]}

        none_payload = client._build_payload("s", [], None, ThinkingLevel.NONE)
        assert none_payload.get("think") is False

        low_payload = client._build_payload("s", [], None, ThinkingLevel.LOW)
        assert "think" not in low_payload

        for level in (ThinkingLevel.MEDIUM, ThinkingLevel.HIGH, ThinkingLevel.MAX):
            payload = client._build_payload("s", [], None, level)
            assert payload.get("think") is True

    def test_think_absent_when_not_supported(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        client = OllamaClient({"host": "http://127.0.0.1:1", "model": "m"})
        client._show_payload = {}

        for level in (ThinkingLevel.NONE, ThinkingLevel.LOW,
                      ThinkingLevel.MEDIUM, ThinkingLevel.HIGH, ThinkingLevel.MAX):
            payload = client._build_payload("s", [], None, level)
            assert "think" not in payload, f"Unexpected 'think' for level={level}"


# ---------------------------------------------------------------------------
# 4. ThreadGist pins thinking_mode to 'none'
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestThreadGistPinsNone:

    def test_thread_gist_thinking_mode_is_none(self) -> None:
        assert ThreadGistConfig.thinking_mode == "none"
        assert ThinkingLevel("none") is ThinkingLevel.NONE


# ---------------------------------------------------------------------------
# 5. _is_thinking_rejection recognition
#
# A pure predicate over an exception and the kwargs that produced it. What is
# proven here is the classifier's own logic — that it keys off the kwarg being
# present, not the message alone — never that a given provider emits a given
# string.
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsThinkingRejection:

    def test_reasoning_effort_rejection_with_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("unsupported value for reasoning_effort")
        assert _is_thinking_rejection(exc, {"reasoning_effort": "none"}) is True

    def test_reasoning_effort_error_without_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("unsupported value for reasoning_effort")
        assert _is_thinking_rejection(exc, {}) is False

    def test_extra_body_thinking_rejection(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("Extra inputs are not permitted extra_forbidden")
        kwargs: dict[str, object] = {"extra_body": {"thinking": {"type": "disabled"}}}
        assert _is_thinking_rejection(exc, kwargs) is True

    def test_unrelated_error_with_both_kwargs(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("boom")
        kwargs: dict[str, object] = {
            "reasoning_effort": "none",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        assert _is_thinking_rejection(exc, kwargs) is False

    def test_extra_body_without_thinking_key(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = Exception("Extra inputs are not permitted extra_forbidden")
        kwargs: dict[str, object] = {"extra_body": {"something_else": "value"}}
        assert _is_thinking_rejection(exc, kwargs) is False
