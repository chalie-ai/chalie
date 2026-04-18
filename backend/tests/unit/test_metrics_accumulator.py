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
