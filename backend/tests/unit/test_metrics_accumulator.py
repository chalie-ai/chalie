import time

import pytest

from services.metrics_accumulator import MetricsAccumulator


@pytest.mark.unit
class TestMetricsAccumulator:
    def test_construction_defaults(self):
        acc = MetricsAccumulator()
        assert acc.tokens_input == 0
        assert acc.tokens_output == 0
        assert acc.tokens_thinking == 0
        assert acc.tokens_cache_read == 0
        assert acc.tokens_cache_create == 0
        assert acc.tool_counts == {}
        assert acc.tokens_total_complete is True
        assert acc.start_time > 0

    def test_accumulate_all_fields(self):
        class FakeResponse:
            tokens_input = 100
            tokens_output = 50
            tokens_thinking = 20
            tokens_cache_read = 10
            tokens_cache_create = 5

        acc = MetricsAccumulator()
        acc.accumulate(FakeResponse())
        acc.accumulate(FakeResponse())

        assert acc.tokens_input == 200
        assert acc.tokens_output == 100
        assert acc.tokens_thinking == 40
        assert acc.tokens_cache_read == 20
        assert acc.tokens_cache_create == 10
        assert acc.tokens_total_complete is True

    def test_accumulate_none_response(self):
        acc = MetricsAccumulator()
        acc.accumulate(None)
        assert acc.tokens_input == 0
        assert acc.tokens_total_complete is True

    def test_accumulate_marks_incomplete_when_primary_missing(self):
        class PartialResponse:
            tokens_input = None
            tokens_output = None
            tokens_thinking = 10

        acc = MetricsAccumulator()
        acc.accumulate(PartialResponse())
        assert acc.tokens_total_complete is False
        assert acc.tokens_thinking == 10

    def test_accumulate_marks_incomplete_when_input_missing(self):
        class OneSided:
            tokens_input = None
            tokens_output = 50

        acc = MetricsAccumulator()
        acc.accumulate(OneSided())
        assert acc.tokens_total_complete is False
        assert acc.tokens_output == 50

    def test_record_tool_counts(self):
        acc = MetricsAccumulator()
        acc.record_tool("Grep")
        acc.record_tool("Grep")
        acc.record_tool("Read")
        assert acc.tool_counts == {"Grep": 2, "Read": 1}

    def test_record_tool_empty_name_ignored(self):
        acc = MetricsAccumulator()
        acc.record_tool("")
        assert acc.tool_counts == {}

    def test_snapshot_shape(self):
        acc = MetricsAccumulator()
        acc.record_tool("code_eval")

        class R:
            tokens_input = 200
            tokens_output = 100
            tokens_thinking = None
            tokens_cache_read = None
            tokens_cache_create = None

        acc.accumulate(R())
        snap = acc.snapshot()

        assert snap["tokens_total"] == 300
        assert snap["tools"] == {"code_eval": 1}
        assert snap["response_time_s"] >= 0
        assert "tokens_total_complete" not in snap

    def test_snapshot_response_time_positive(self):
        acc = MetricsAccumulator()
        acc.start_time = time.time() - 1.5
        snap = acc.snapshot()
        assert snap["response_time_s"] >= 1.5

    def test_snapshot_incomplete_flag_present(self):
        acc = MetricsAccumulator()
        acc.tokens_total_complete = False
        snap = acc.snapshot()
        assert snap.get("tokens_total_complete") is False

    def test_snapshot_tools_empty_dict_not_omitted(self):
        acc = MetricsAccumulator()
        snap = acc.snapshot()
        assert "tools" in snap
        assert snap["tools"] == {}

    def test_set_turn_start_via_direct_assignment(self):
        acc = MetricsAccumulator()
        fixed = time.time() - 10.0
        acc.start_time = fixed
        snap = acc.snapshot()
        assert snap["response_time_s"] >= 10.0

    # ── Timing extensions ────────────────────────────────────────────────

    def test_snapshot_emits_timing_block(self):
        acc = MetricsAccumulator()
        snap = acc.snapshot()
        assert "timing" in snap
        t = snap["timing"]
        for key in ("wall_ms", "pre_llm_ms", "llm_ms", "post_llm_ms",
                    "llm_calls", "stages", "iterations"):
            assert key in t

    def test_stage_records_into_stages_dict(self):
        acc = MetricsAccumulator()
        with acc.stage("pre_act"):
            time.sleep(0.01)
        snap = acc.snapshot()
        assert snap["timing"]["stages"]["pre_act"] >= 10
        assert snap["timing"]["pre_llm_ms"] >= 10

    def test_stage_accumulates_repeated_calls(self):
        acc = MetricsAccumulator()
        with acc.stage("prompt_assembly"):
            time.sleep(0.005)
        with acc.stage("prompt_assembly"):
            time.sleep(0.005)
        snap = acc.snapshot()
        assert snap["timing"]["stages"]["prompt_assembly"] >= 10

    def test_record_llm_call_sums_into_llm_ms(self):
        acc = MetricsAccumulator()
        acc.record_llm_call(120)
        acc.record_llm_call(80)
        snap = acc.snapshot()
        assert snap["timing"]["llm_ms"] == 200
        assert snap["timing"]["llm_calls"] == 2

    def test_record_llm_call_negative_or_none_ignored(self):
        acc = MetricsAccumulator()
        acc.record_llm_call(None)
        acc.record_llm_call(-5)
        snap = acc.snapshot()
        assert snap["timing"]["llm_ms"] == 0
        assert snap["timing"]["llm_calls"] == 0

    def test_iteration_attributes_llm_calls(self):
        acc = MetricsAccumulator()
        with acc.iteration(0):
            acc.record_llm_call(150)
        with acc.iteration(1):
            acc.record_llm_call(75)
            acc.record_llm_call(50)
        snap = acc.snapshot()
        iters = snap["timing"]["iterations"]
        assert len(iters) == 2
        assert iters[0]["iter"] == 0
        assert iters[0]["llm_ms"] == 150
        assert iters[0]["llm_calls"] == 1
        assert iters[1]["iter"] == 1
        assert iters[1]["llm_ms"] == 125
        assert iters[1]["llm_calls"] == 2
        assert snap["timing"]["llm_ms"] == 275

    def test_iteration_captures_stages_per_iter(self):
        acc = MetricsAccumulator()
        with acc.iteration(0):
            with acc.stage("prompt_assembly"):
                time.sleep(0.005)
        snap = acc.snapshot()
        iter_stages = snap["timing"]["iterations"][0]["stages"]
        assert "prompt_assembly" in iter_stages
        assert iter_stages["prompt_assembly"] >= 5

    def test_post_llm_ms_derived(self):
        acc = MetricsAccumulator()
        # Simulate 100ms wall, 30ms pre, 50ms llm → post = 20ms
        acc.start_time = time.time() - 0.1
        with acc.stage("pre_act"):
            time.sleep(0.03)
        acc.record_llm_call(50)
        snap = acc.snapshot()
        # Wall is now > 100ms because we slept inside the test, so post is whatever's left.
        assert snap["timing"]["post_llm_ms"] >= 0
        assert (
            snap["timing"]["pre_llm_ms"]
            + snap["timing"]["llm_ms"]
            + snap["timing"]["post_llm_ms"]
            == snap["timing"]["wall_ms"]
        )

    def test_add_stage_ms_external_record(self):
        acc = MetricsAccumulator()
        acc.add_stage_ms("external_stage", 250)
        snap = acc.snapshot()
        assert snap["timing"]["stages"]["external_stage"] == 250
        assert snap["timing"]["pre_llm_ms"] == 250

    def test_merge_absorbs_llm_time(self):
        parent = MetricsAccumulator()
        child = MetricsAccumulator()
        child.record_llm_call(400)
        parent.merge(child)
        snap = parent.snapshot()
        assert snap["timing"]["llm_ms"] == 400
        assert snap["timing"]["llm_calls"] == 1
