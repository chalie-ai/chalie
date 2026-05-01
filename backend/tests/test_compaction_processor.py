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

    def test_subagent_trail_check_threshold_always_false(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        p = SubagentTrailCompactionProcessor(raw_input='x')
        assert p._check_threshold('system prompt', [{'name': 'tool'}], 'a very long string ' * 1000) is False

    def test_continuity_job_is_frontal_cortex_unified(self):
        from services.compaction_message_processor import ContinuityCompactionProcessor
        assert ContinuityCompactionProcessor.JOB == 'frontal-cortex-unified'

    def test_subagent_trail_job_is_frontal_cortex_unified(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        assert SubagentTrailCompactionProcessor.JOB == 'frontal-cortex-unified'

    def test_continuity_always_available_empty(self):
        from services.compaction_message_processor import ContinuityCompactionProcessor
        assert ContinuityCompactionProcessor.ALWAYS_AVAILABLE == []

    def test_subagent_trail_always_available_empty(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        assert SubagentTrailCompactionProcessor.ALWAYS_AVAILABLE == []

    def test_continuity_discoverable_empty(self):
        from services.compaction_message_processor import ContinuityCompactionProcessor
        assert ContinuityCompactionProcessor.DISCOVERABLE == []

    def test_subagent_trail_discoverable_empty(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        assert SubagentTrailCompactionProcessor.DISCOVERABLE == []

    def test_continuity_skip_transcript_write(self):
        from services.compaction_message_processor import ContinuityCompactionProcessor
        assert ContinuityCompactionProcessor.SKIP_TRANSCRIPT_WRITE is True

    def test_subagent_trail_skip_transcript_write(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        assert SubagentTrailCompactionProcessor.SKIP_TRANSCRIPT_WRITE is True


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

    def test_merge_preserves_start_time(self):
        import time
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator()
        original_start = a.start_time
        time.sleep(0.01)
        b = MetricsAccumulator()
        a.merge(b)
        assert a.start_time == original_start

    def test_merge_propagates_incomplete_flag(self):
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator(tokens_total_complete=True)
        b = MetricsAccumulator(tokens_total_complete=False)
        a.merge(b)
        assert a.tokens_total_complete is False

    def test_merge_complete_stays_complete_when_both_complete(self):
        from services.metrics_accumulator import MetricsAccumulator
        a = MetricsAccumulator(tokens_total_complete=True)
        b = MetricsAccumulator(tokens_total_complete=True)
        a.merge(b)
        assert a.tokens_total_complete is True


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

            def getUserPrompt(self):
                return 'hi'

            def getUserDefinition(self):
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
        import json
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

    def test_small_full_payload_does_not_exceed_threshold(self):
        """When the entire payload is well within compact_at, return False."""
        from unittest.mock import patch, MagicMock

        proc = self._make_processor()
        fake_provider = MagicMock()
        fake_provider.get_compact_at.return_value = 128_000
        fake_provider.estimate_payload_tokens.return_value = 100  # well below

        with patch('services.providers.Providers') as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider
            result = proc._check_threshold('short system', [], 'short body')

        assert result is False

    def test_missing_compact_at_returns_false_and_logs_error(self):
        """compact_at=None must return False (skip compaction) and log an error."""
        from unittest.mock import patch, MagicMock
        import logging

        proc = self._make_processor()
        fake_provider = MagicMock()
        fake_provider.get_compact_at.return_value = None

        with patch('services.providers.Providers') as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider
            with patch('services.message_processor.logger') as mock_logger:
                result = proc._check_threshold('system', [], 'body')

        assert result is False
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert 'compact_at missing' in call_args[0]
