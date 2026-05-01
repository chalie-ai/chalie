# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for _filter_seed_to_high_relevance in user_message_processor.

Verifies line-level relevance filtering of the pre_act memory seed block.
Only relevance:high body lines are retained; the header results=N is rewritten
to match the kept count. The block is dropped (returns None) when zero
high-relevance lines survive or the block is structurally malformed.
"""

import pytest

pytestmark = pytest.mark.unit


def _filter(block: str):
    from services.user_message_processor import _filter_seed_to_high_relevance
    return _filter_seed_to_high_relevance(block)


class TestFilterSeedToHighRelevance:
    """Tests for the line-level relevance filter applied in pre_act()."""

    def test_all_high_passthrough(self):
        """Three relevance:high lines all survive; results count stays 3."""
        block = (
            "[memory(query=test, results=3)]\n"
            "[id:key1,relevance:high] fact one\n"
            "[id:key2,relevance:high] fact two\n"
            "[id:key3,relevance:high] fact three\n"
            "[end:memory]"
        )
        result = _filter(block)

        assert result is not None
        lines = result.splitlines()
        assert lines[0] == "[memory(query=test, results=3)]"
        assert lines[-1] == "[end:memory]"
        assert "[id:key1,relevance:high] fact one" in result
        assert "[id:key2,relevance:high] fact two" in result
        assert "[id:key3,relevance:high] fact three" in result
        assert "results=3" in lines[0]

    def test_mixed_keeps_only_high(self):
        """Only the relevance:high line survives from a mixed block; results=1."""
        block = (
            "[memory(query=test, results=3)]\n"
            "[id:key1,relevance:high] important fact\n"
            "[id:key2,relevance:medium] meh fact\n"
            "[id:key3,relevance:low] weak fact\n"
            "[end:memory]"
        )
        result = _filter(block)

        assert result is not None
        lines = result.splitlines()
        assert "results=1" in lines[0]
        assert "[id:key1,relevance:high] important fact" in result
        assert "relevance:medium" not in result
        assert "relevance:low" not in result

    def test_all_low_returns_none(self):
        """A block with no relevance:high lines must return None."""
        block = (
            "[memory(query=test, results=2)]\n"
            "[id:key1,relevance:medium] medium fact\n"
            "[id:key2,relevance:low] low fact\n"
            "[end:memory]"
        )
        assert _filter(block) is None

    def test_malformed_missing_footer_returns_none(self):
        """A block missing [end:memory] is malformed and must return None."""
        block = (
            "[memory(query=test, results=1)]\n"
            "[id:key1,relevance:high] some fact\n"
        )
        assert _filter(block) is None

    def test_empty_string_returns_none(self):
        """Empty input must return None without raising."""
        assert _filter("") is None

    def test_results_count_rewritten_correctly(self):
        """results=N in header is rewritten to match the number of kept lines."""
        block = (
            "[memory(query=weather, results=5)]\n"
            "[id:a,relevance:high] sunny\n"
            "[id:b,relevance:low] rainy\n"
            "[id:c,relevance:high] cloudy\n"
            "[id:d,relevance:medium] windy\n"
            "[id:e,relevance:high] foggy\n"
            "[end:memory]"
        )
        result = _filter(block)

        assert result is not None
        header = result.splitlines()[0]
        assert "results=3" in header
        assert "results=5" not in header
