"""Regression tests for CompactionMessageProcessor family.

Minimal — only asserts invariants that catch real regressions:
1. Recursion guard: _check_threshold is hard-disabled on both subclasses.
2. Class constants: JOB, ALWAYS_AVAILABLE, DISCOVERABLE, SKIP_TRANSCRIPT_WRITE.
3. compaction_service module is gone (ModuleNotFoundError on import).
4. MetricsAccumulator.merge() sums token and tool counts correctly.
"""

import json
import pytest

pytestmark = pytest.mark.unit


class TestCompactionProcessorConstants:
    def test_continuity_check_threshold_always_false(self):
        from services.compaction_message_processor import ContinuityCompactionProcessor
        p = ContinuityCompactionProcessor(raw_input='x')
        assert p._check_threshold('system prompt', [{'name': 'tool'}], 'a very long string ' * 1000) is False

    def test_processor_constants(self):
        from services.compaction_message_processor import (
            ContinuityCompactionProcessor, SubagentTrailCompactionProcessor,
        )
        for cls in (ContinuityCompactionProcessor, SubagentTrailCompactionProcessor):
            assert cls.LOG_LABEL == 'compaction'
            assert cls.ALWAYS_AVAILABLE == []
            assert cls.DISCOVERABLE == []
            assert cls.SKIP_TRANSCRIPT_WRITE is True


class TestCompactionServiceGone:
    def test_compaction_service_not_importable(self):
        with pytest.raises(ModuleNotFoundError):
            import services.compaction_service  # noqa: F401


class TestMetricsAccumulatorMerge:
    def test_merge_sums_token_fields(self):
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator(
            tokens_input=100, tokens_output=50,
            tokens_thinking=10, tokens_cache_read=5, tokens_cache_create=3,
        )
        b = MetricsAccumulator(
            tokens_input=200, tokens_output=75,
            tokens_thinking=20, tokens_cache_read=8, tokens_cache_create=4,
        )
        a.merge(b)
        assert a.tokens_input == 300
        assert a.tokens_output == 125
        assert a.tokens_thinking == 30
        assert a.tokens_cache_read == 13
        assert a.tokens_cache_create == 7

    def test_merge_sums_tool_counts(self):
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator()
        a.record_tool('memory')
        a.record_tool('search')
        b = MetricsAccumulator()
        b.record_tool('memory')
        b.record_tool('memory')
        a.merge(b)
        assert a.tool_counts['memory'] == 3
        assert a.tool_counts['search'] == 1

    def test_merge_propagates_incomplete_flag(self):
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator(tokens_total_complete=True)
        b = MetricsAccumulator(tokens_total_complete=False)
        a.merge(b)
        assert a.tokens_total_complete is False

class TestCheckThresholdFullPayload:
    """Regression: _check_threshold must measure the full payload, not just user_body.

    Bug: compact_at is 80% of the provider's max_tokens (the full context window).
    Before this fix, only user_body was measured, leaving system_prompt and tools
    schema unmeasured — together they typically dwarf user_body.

    This test verifies that a small user_body combined with a large system_prompt
    and large tools schema DOES exceed compact_at, which the old code would have
    missed (returning False and skipping compaction).
    """

    def _make_processor(self):
        from services.message_processor import MessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt

        class _MinimalProcessor(MessageProcessor):
            CHANNEL = 'subagent'
            ROLE = 'subagent'
            SKIP_TRANSCRIPT_WRITE = True
            SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt

            def get_user_prompt(self):
                return 'hi'

            def get_user_definition(self):
                return 'test user'

        return _MinimalProcessor(raw_input='hi')

    def test_full_payload_exceeds_threshold_when_system_and_tools_are_large(self):
        """A tiny user_body must NOT hide a large system_prompt + tools schema."""
        from unittest.mock import patch, MagicMock
        from services.llm_service import estimate_tokens

        # 8000-word system prompt + large tools schema; user_body is tiny.
        large_system = 'word ' * 8_000        # ~10 400 estimated tokens
        large_tools = [{'name': f'tool_{i}', 'description': 'long description ' * 20}
                       for i in range(50)]    # 50 tools × ~20 words each
        small_user = 'hi'

        # compact_at set to something smaller than the combined estimate
        # but larger than user_body alone (proving the old code would have
        # returned False while the new code returns True).
        tools_json = json.dumps(large_tools, separators=(',', ':'))
        total_est = (
            estimate_tokens(large_system)
            + estimate_tokens(tools_json)
            + estimate_tokens(small_user)
        )
        user_only_est = estimate_tokens(small_user)
        compact_at = user_only_est + 1   # just above user_body alone

        # Sanity: total must exceed compact_at (otherwise test is meaningless)
        assert total_est > compact_at, (
            f"test setup error: total_est={total_est} must exceed compact_at={compact_at}"
        )

        proc = self._make_processor()
        fake_provider = MagicMock()
        fake_provider.get_compact_at.return_value = compact_at
        # estimate_payload_tokens delegates to provider.build_request_body —
        # in production it returns the literal HTTP body's token estimate.
        # Here we return our pre-computed `total_est` so the test asserts the
        # comparison logic, not the provider's serialisation details.
        fake_provider.estimate_payload_tokens.return_value = total_est

        with patch('services.providers.Providers') as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider
            result = proc._check_threshold(large_system, large_tools, small_user)

        assert result is True, (
            f"_check_threshold should return True when system+tools+user "
            f"({total_est} tokens) > compact_at ({compact_at}), "
            f"but user_body alone ({user_only_est} tokens) is below it"
        )




# ─────────────────────────────────────────────────────────────────────────────
# build_request_body — each provider must include system_prompt, messages,
# and tools in its serialised output.
#
# Regression: if a provider's build_request_body omits the system_prompt or
# tools schema from its serialised payload, estimate_payload_tokens will
# under-count and _check_threshold will never fire, silently disabling
# compaction.  No network call is needed — build_request_body is a pure
# serialisation function.
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildRequestBodyIncludesAllComponents:
    """Each provider's build_request_body must serialise system_prompt,
    messages, AND tools.  If any are missing, the token estimate is wrong
    and compaction can be silently disabled.
    """

    _SYSTEM = 'You are a helpful assistant with important instructions here.'
    _MESSAGES = [{'role': 'user', 'content': 'What is the capital of France?'}]
    _TOOLS = [
        {
            'name': 'search',
            'description': 'Search the web for information',
            'input_schema': {
                'type': 'object',
                'properties': {'query': {'type': 'string', 'description': 'Search query'}},
                'required': ['query'],
            },
        }
    ]

    def test_ollama_includes_system_message_and_tools(self):
        from services.ollama_service import OllamaService
        svc = OllamaService({'platform': 'ollama', 'host': 'http://localhost:11434', 'model': 'llama3'})
        body = svc.build_request_body(self._SYSTEM, self._MESSAGES, self._TOOLS)
        assert self._SYSTEM in body
        assert 'What is the capital of France?' in body
        assert 'search' in body

    def test_anthropic_includes_system_message_and_tools(self):
        from services.llm_service import AnthropicService
        svc = AnthropicService({'api_key': 'test-key', 'model': 'claude-haiku-4-5-20251001'})
        body = svc.build_request_body(self._SYSTEM, self._MESSAGES, self._TOOLS)
        assert self._SYSTEM in body
        assert 'What is the capital of France?' in body
        assert 'search' in body

    def test_build_request_body_without_tools_smaller_than_with_tools(self):
        """Serialised body WITH tools must be strictly larger than WITHOUT tools.
        If they are equal the tools schema is silently dropped and the token
        estimate will under-count for tool-heavy turns.
        """
        from services.ollama_service import OllamaService
        svc = OllamaService({'platform': 'ollama', 'host': 'http://localhost:11434', 'model': 'llama3'})
        body_no_tools = svc.build_request_body(self._SYSTEM, self._MESSAGES, tools=None)
        body_with_tools = svc.build_request_body(self._SYSTEM, self._MESSAGES, self._TOOLS)
        assert len(body_with_tools) > len(body_no_tools), (
            "build_request_body with tools must produce a larger payload than without tools. "
            "If they are equal, tools are being silently dropped and compaction token "
            "estimates will be wrong."
        )
