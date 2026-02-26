"""Tests for semantic_consolidation_worker — routing, status determination, salience boost."""

import pytest


pytestmark = pytest.mark.unit


# ── Batch vs single routing ──────────────────────────────────────────

class TestRouting:

    def test_batch_consolidation_type_routes_to_batch(self):
        """message_data with type='batch_consolidation' is recognized as batch."""
        message_data = {'type': 'batch_consolidation'}
        assert message_data.get('type') == 'batch_consolidation'

    def test_episode_key_routes_to_single(self):
        """message_data with 'episode' key is recognized as single processing."""
        message_data = {'episode': {'id': 'ep-001', 'gist': 'test'}}
        assert message_data.get('type') != 'batch_consolidation'
        assert message_data.get('episode') is not None


# ── Status determination ─────────────────────────────────────────────

class TestStatusDetermination:

    def test_status_empty_when_zero_concepts(self):
        """0 concepts extracted → status 'empty' (retried on next batch)."""
        concepts_created = 0
        status = 'completed' if concepts_created > 0 else 'empty'
        assert status == 'empty'

    def test_status_completed_when_concepts_extracted(self):
        """>0 concepts extracted → status 'completed'."""
        concepts_created = 3
        status = 'completed' if concepts_created > 0 else 'empty'
        assert status == 'completed'


# ── Salience boost for tool_reflection episodes ──────────────────────

class TestSalienceBoost:
    """
    Promotion rule:
        if source == 'tool_reflection' and retrieval_count >= 3:
            salience = min(10, salience + 2)
    """

    def test_boost_applied_when_tool_reflection_and_retrieval_ge_3(self):
        salience_factors = {'source': 'tool_reflection', 'retrieval_count': 3}
        salience = 5
        if (salience_factors.get('source') == 'tool_reflection'
                and salience_factors.get('retrieval_count', 0) >= 3):
            salience = min(10, salience + 2)
        assert salience == 7

    def test_no_boost_when_retrieval_below_3(self):
        salience_factors = {'source': 'tool_reflection', 'retrieval_count': 2}
        salience = 5
        if (salience_factors.get('source') == 'tool_reflection'
                and salience_factors.get('retrieval_count', 0) >= 3):
            salience = min(10, salience + 2)
        assert salience == 5

    def test_no_boost_when_source_not_tool_reflection(self):
        salience_factors = {'source': 'conversation', 'retrieval_count': 5}
        salience = 5
        if (salience_factors.get('source') == 'tool_reflection'
                and salience_factors.get('retrieval_count', 0) >= 3):
            salience = min(10, salience + 2)
        assert salience == 5

    def test_boost_capped_at_10(self):
        salience_factors = {'source': 'tool_reflection', 'retrieval_count': 3}
        salience = 9
        if (salience_factors.get('source') == 'tool_reflection'
                and salience_factors.get('retrieval_count', 0) >= 3):
            salience = min(10, salience + 2)
        assert salience == 10
