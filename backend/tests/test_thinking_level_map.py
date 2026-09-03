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
# 2b. A platform states its own scale rather than inheriting OpenAI's
#
# Most clients subclass OpenAICompatibleClient, so an absent map is not an
# absent behaviour — it is OpenAI's vocabulary, silently. vLLM's scale has no
# 'high' in it at all, which is why the inherited row cost a 400 on every HIGH
# and MAX request. What the server accepts was established by firing at it;
# what the client *sends* is what these assert.
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPerPlatformReasoningScale:

    def test_vllm_never_sends_a_value_its_scale_lacks(self) -> None:
        from services.llm_clients.vllm import VllmClient

        client = VllmClient({"platform": "vllm", "model": "m", "host": "http://127.0.0.1:1/v1"})
        sent = {level: client._thinking_native(level) for level in ThinkingLevel}

        assert sent[ThinkingLevel.HIGH] == "xhigh"
        assert sent[ThinkingLevel.MAX] == "xhigh"
        assert "high" not in [v for v in sent.values() if v is not None]

    def test_vllm_spells_low_out_because_its_default_is_the_ceiling(self) -> None:
        """Sending no flag to vLLM buys 'xhigh' — its default. LOW must be said."""
        from services.llm_clients.vllm import VllmClient

        client = VllmClient({"platform": "vllm", "model": "m", "host": "http://127.0.0.1:1/v1"})
        assert client._thinking_native(ThinkingLevel.LOW) == "low"

    def test_base_client_still_answers_openai_vocabulary(self) -> None:
        """The escape hatch serves uncharacterised hosts, so it keeps the
        protocol originator's spelling — and LOW stays absent there."""
        from services.llm_clients.openai_compatible import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            {"platform": "openai_compatible", "model": "m", "host": "http://127.0.0.1:1/v1", "api_key": "k"},
        )
        assert client._thinking_native(ThinkingLevel.HIGH) == "high"
        assert client._thinking_native(ThinkingLevel.LOW) is None

    def test_every_declared_scale_is_reachable_from_its_platform(self) -> None:
        """A map declared but never wired to a class is the same as no map."""
        from services.llm_clients import thinking_map
        from services.llm_clients.openai_compatible import OpenAICompatibleClient
        from services.llm_clients.registry import PROVIDERS_BY_PLATFORM

        declared = {
            name for name in dir(thinking_map) if name.endswith("_REASONING_EFFORTS")
        }
        wired = set()
        for cls in PROVIDERS_BY_PLATFORM.values():
            if not issubclass(cls, OpenAICompatibleClient):
                continue
            row = cls.REASONING_EFFORTS
            for name in declared:
                if getattr(thinking_map, name) is row:
                    wired.add(name)

        # CODEX_REASONING_EFFORTS belongs to a non-OpenAI-protocol client and is
        # read by CodexCliClient directly, so it is not reachable through this
        # class attribute. Everything else here must be.
        assert declared - wired == {"CODEX_REASONING_EFFORTS"}, (
            f"declared but wired to no platform: {sorted(declared - wired)}"
        )

    # Five vendors default their reasoning effort to the TOP of their own
    # scale. On those, this module's usual "LOW = send no flag, take the
    # default" convention hands the *most* thinking to the level named least —
    # silently, with no error to notice. Every one of them must spell LOW out.
    TOP_DEFAULT_PLATFORMS = ("vllm", "xai", "deepseek", "zhipu", "moonshot")

    @pytest.mark.parametrize("platform", TOP_DEFAULT_PLATFORMS)
    def test_low_is_spelled_out_where_the_vendor_default_is_the_ceiling(
        self, platform: str,
    ) -> None:
        from services.llm_clients.openai_compatible import OpenAICompatibleClient
        from services.llm_clients.registry import PROVIDERS_BY_PLATFORM

        cls = PROVIDERS_BY_PLATFORM[platform]
        assert issubclass(cls, OpenAICompatibleClient)
        client = cls({
            "platform": platform, "model": "m",
            "host": "http://127.0.0.1:1/v1", "api_key": "k",
        })
        sent = client._thinking_native(ThinkingLevel.LOW)

        assert sent is not None, (
            f"{platform} sends no flag at LOW, so it takes the vendor default — "
            "which on this vendor is the top of the scale."
        )
        assert sent != client._thinking_native(ThinkingLevel.MAX), (
            f"{platform} sends the same value at LOW and MAX"
        )

    def test_xai_never_sends_a_disable_value(self) -> None:
        """xAI documents that reasoning cannot be disabled, so 'none' — which
        the inherited OpenAI row sent at NONE — is not a value it accepts."""
        from services.llm_clients.xai import XaiClient

        client = XaiClient({"platform": "xai", "model": "m", "api_key": "k"})
        sent = {client._thinking_native(level) for level in ThinkingLevel}

        assert not sent & {"none", "minimal"}
        assert client._thinking_native(ThinkingLevel.NONE) == "low"


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
# proven here is the classifier's own logic — that it keys off the thinking
# param having been SENT and the provider answering 400 — never that a given
# provider emits a given string. The errors are the openai client's real
# exception classes, not a stand-in: that the status is readable off what the
# ladder actually catches is the whole assumption the classifier rests on.
# ---------------------------------------------------------------------------

def _api_error(message: str, status: int) -> Exception:
    """The openai client's own error for *status*, carrying *message*.

    ``httpx2`` is deliberate, not a typo: the installed openai SDK imports that
    name, and an error built on the other httpx is a different class to the one
    the ladder catches — which would make this whole section prove nothing.
    """
    import httpx2
    from openai import APIStatusError

    request = httpx2.Request("POST", "http://provider.invalid/v1/chat/completions")
    return APIStatusError(message, response=httpx2.Response(status, request=request), body=None)


@pytest.mark.unit
class TestIsThinkingRejection:

    def test_reasoning_effort_rejected_with_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = _api_error("unsupported value for reasoning_effort", 400)
        assert _is_thinking_rejection(exc, {"reasoning_effort": "none"}) is True

    def test_rejection_wording_the_classifier_must_not_depend_on(self) -> None:
        """The refusal that used to kill the turn: the parameter is spelled with
        a space and the prose says "Unexpected"/"Supported", so a substring match
        on 'reasoning_effort' or 'unsupported' finds neither and the recovery
        ladder never runs."""
        from services.llm_service import _is_thinking_rejection
        exc = _api_error(
            "Unexpected reasoning effort high. "
            "Supported types are xhigh (default), medium, and low.",
            400,
        )
        assert _is_thinking_rejection(exc, {"reasoning_effort": "high"}) is True

    def test_reasoning_effort_error_without_kwarg(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = _api_error("unsupported value for reasoning_effort", 400)
        assert _is_thinking_rejection(exc, {}) is False

    def test_extra_body_thinking_rejection(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = _api_error("Extra inputs are not permitted extra_forbidden", 400)
        kwargs: dict[str, object] = {"extra_body": {"thinking": {"type": "disabled"}}}
        assert _is_thinking_rejection(exc, kwargs) is True

    def test_server_fault_is_not_a_thinking_rejection(self) -> None:
        """Only a 400 says the request's shape was refused. Stripping the
        thinking params off a 500 would retry into the same server fault."""
        from services.llm_service import _is_thinking_rejection
        exc = _api_error("internal server error", 500)
        kwargs: dict[str, object] = {
            "reasoning_effort": "none",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        assert _is_thinking_rejection(exc, kwargs) is False

    def test_transport_failure_carries_no_status(self) -> None:
        """A connection error never reached the provider, so nothing was refused."""
        from services.llm_service import _is_thinking_rejection
        assert _is_thinking_rejection(Exception("connection reset"), {"reasoning_effort": "high"}) is False

    def test_extra_body_without_thinking_key(self) -> None:
        from services.llm_service import _is_thinking_rejection
        exc = _api_error("Extra inputs are not permitted extra_forbidden", 400)
        kwargs: dict[str, object] = {"extra_body": {"something_else": "value"}}
        assert _is_thinking_rejection(exc, kwargs) is False
