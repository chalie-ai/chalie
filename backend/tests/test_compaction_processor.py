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
        assert p._check_threshold('a very long string ' * 1000) is False

    def test_subagent_trail_check_threshold_always_false(self):
        from services.compaction_message_processor import SubagentTrailCompactionProcessor
        p = SubagentTrailCompactionProcessor(raw_input='x')
        assert p._check_threshold('a very long string ' * 1000) is False

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
