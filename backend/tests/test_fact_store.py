"""Tests for FactStoreService — confidence conflict, eviction, formatting."""

import pytest
from services.fact_store_service import FactStoreService


pytestmark = pytest.mark.unit


class TestFactStore:

    def test_store_and_retrieve_fact(self, mock_redis):
        svc = FactStoreService(ttl_minutes=1440, max_facts_per_topic=50)
        stored = svc.store_fact("topic-a", "name", "Dylan", confidence=0.9)
        assert stored is True

        fact = svc.get_fact("topic-a", "name")
        assert fact is not None
        assert fact['value'] == 'Dylan'
        assert fact['confidence'] == 0.9

    def test_higher_confidence_wins_conflict(self, mock_redis):
        """Same key, higher confidence should overwrite."""
        svc = FactStoreService()
        svc.store_fact("topic-a", "name", "Alice", confidence=0.5)
        svc.store_fact("topic-a", "name", "Bob", confidence=0.9)

        fact = svc.get_fact("topic-a", "name")
        assert fact['value'] == 'Bob'
        assert fact['confidence'] == 0.9

    def test_lower_confidence_rejected(self, mock_redis):
        """Same key, lower confidence should keep existing."""
        svc = FactStoreService()
        svc.store_fact("topic-a", "name", "Alice", confidence=0.9)
        stored = svc.store_fact("topic-a", "name", "Bob", confidence=0.3)

        assert stored is False
        fact = svc.get_fact("topic-a", "name")
        assert fact['value'] == 'Alice'

    def test_max_facts_eviction(self, mock_redis):
        """51st fact evicts oldest."""
        svc = FactStoreService(max_facts_per_topic=3)

        svc.store_fact("topic-a", "fact1", "v1", confidence=0.5)
        svc.store_fact("topic-a", "fact2", "v2", confidence=0.5)
        svc.store_fact("topic-a", "fact3", "v3", confidence=0.5)
        svc.store_fact("topic-a", "fact4", "v4", confidence=0.5)

        # fact1 should be evicted (oldest)
        fact1 = svc.get_fact("topic-a", "fact1")
        assert fact1 is None

        # fact4 should exist
        fact4 = svc.get_fact("topic-a", "fact4")
        assert fact4 is not None
        assert fact4['value'] == 'v4'

    def test_get_all_facts_ordered(self, mock_redis):
        """get_all_facts returns newest first."""
        svc = FactStoreService()
        svc.store_fact("topic-a", "first", "v1", confidence=0.5)
        svc.store_fact("topic-a", "second", "v2", confidence=0.5)
        svc.store_fact("topic-a", "third", "v3", confidence=0.5)

        facts = svc.get_all_facts("topic-a")
        assert len(facts) == 3
        # Newest first (reverse order of insertion by timestamp)
        assert facts[0]['key'] == 'third'

    def test_formatted_facts_output(self, mock_redis):
        """get_facts_formatted returns readable string."""
        svc = FactStoreService()
        svc.store_fact("topic-a", "language", "Python", confidence=0.9)

        formatted = svc.get_facts_formatted("topic-a")
        assert "## Known Facts" in formatted
        assert "language: Python" in formatted
        assert "0.9" in formatted
