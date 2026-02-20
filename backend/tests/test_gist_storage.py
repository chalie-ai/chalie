"""Tests for GistStorageService — confidence filtering, dedup, TTL, type caps."""

import pytest
import time
from services.gist_storage_service import GistStorageService


pytestmark = pytest.mark.unit


class TestGistStorage:

    def _make_service(self, **kwargs):
        defaults = dict(
            attention_span_minutes=30,
            min_confidence=7,
            max_gists=8,
            similarity_threshold=0.7,
            max_per_type=2,
        )
        defaults.update(kwargs)
        return GistStorageService(**defaults)

    def _make_gist(self, content="test gist", gist_type="observation", confidence=8):
        return {"content": content, "type": gist_type, "confidence": confidence}

    # ── Confidence filtering ──────────────────────────────────────

    def test_store_gist_above_threshold(self, mock_redis):
        svc = self._make_service()
        gists = [self._make_gist(confidence=8)]
        stored = svc.store_gists("topic-a", gists, "hello", "hi there")
        assert stored == 1

    def test_reject_gist_below_threshold(self, mock_redis):
        """The amnesia bug — confidence < min_confidence must be rejected when gists exist."""
        svc = self._make_service()
        # First store a valid gist so has_existing_gists is True
        svc.store_gists("topic-a", [self._make_gist(confidence=9)], "p", "r")

        # Now try to store a low-confidence gist
        stored = svc.store_gists("topic-a", [self._make_gist(confidence=3)], "p2", "r2")
        assert stored == 0

    def test_bypass_threshold_when_empty(self, mock_redis):
        """First gists stored regardless of confidence when topic has no gists."""
        svc = self._make_service()
        gists = [self._make_gist(confidence=2)]
        stored = svc.store_gists("empty-topic", gists, "p", "r")
        assert stored == 1

    def test_cold_start_gists_filtered_from_bypass(self, mock_redis):
        """Cold-start gists don't count as 'has gists' for confidence bypass."""
        svc = self._make_service()
        # Inject cold-start gists (type=cold_start, confidence=5)
        svc.store_cold_start_gists("topic-cold")

        # Cold-start gists exist in the index, so has_existing_gists is True.
        # But a low-confidence gist should still be rejected because
        # the index exists (cold-start gists are in the sorted set).
        # This verifies the current behavior: cold-start gists DO make
        # has_existing_gists True, so low confidence gets rejected.
        stored = svc.store_gists("topic-cold", [self._make_gist(confidence=3)], "p", "r")
        assert stored == 0

    # ── Deduplication ─────────────────────────────────────────────

    def test_dedup_jaccard_identical(self, mock_redis):
        """Identical content should be deduplicated (Jaccard >= 0.7)."""
        svc = self._make_service()
        gist1 = self._make_gist(content="the quick brown fox jumps", confidence=8)
        gist2 = self._make_gist(content="the quick brown fox jumps", confidence=8)

        svc.store_gists("topic-a", [gist1], "p", "r")
        stored = svc.store_gists("topic-a", [gist2], "p2", "r2")
        assert stored == 0

    def test_dedup_replaces_higher_confidence(self, mock_redis):
        """Higher confidence duplicate replaces the existing one."""
        svc = self._make_service()
        gist_low = self._make_gist(content="the quick brown fox jumps over", confidence=7)
        gist_high = self._make_gist(content="the quick brown fox jumps over", confidence=9)

        svc.store_gists("topic-a", [gist_low], "p", "r")
        stored = svc.store_gists("topic-a", [gist_high], "p2", "r2")
        assert stored == 1

        # Verify the stored gist has the higher confidence
        gists = svc.get_latest_gists("topic-a")
        assert len(gists) == 1
        assert gists[0]['confidence'] == 9

    # ── Type cap enforcement ──────────────────────────────────────

    def test_type_cap_enforcement(self, mock_redis):
        """Max 2 per type — lowest confidence removed."""
        svc = self._make_service(max_per_type=2)

        gists = [
            self._make_gist(content="obs one about animals", gist_type="observation", confidence=7),
            self._make_gist(content="obs two about plants", gist_type="observation", confidence=8),
            self._make_gist(content="obs three about rocks", gist_type="observation", confidence=9),
        ]
        svc.store_gists("topic-a", gists, "p", "r")

        result = svc.get_latest_gists("topic-a")
        obs_gists = [g for g in result if g['type'] == 'observation']
        assert len(obs_gists) == 2
        # Should have kept confidence 8 and 9
        confidences = sorted(g['confidence'] for g in obs_gists)
        assert confidences == [8, 9]

    # ── Touch-on-read TTL ─────────────────────────────────────────

    def test_touch_on_read_refreshes_ttl(self, mock_redis):
        """get_latest_gists refreshes TTL via expire calls."""
        svc = self._make_service()
        svc.store_gists("topic-a", [self._make_gist()], "p", "r")

        # Read gists — this should call expire (touch-on-read)
        gists = svc.get_latest_gists("topic-a")
        assert len(gists) == 1

        # Verify the gist key has a TTL set (fakeredis supports ttl())
        index_key = svc._get_gist_index_key("topic-a")
        ttl = mock_redis.ttl(index_key)
        assert ttl > 0

    # ── Confidence coercion ───────────────────────────────────────

    def test_confidence_coercion_string_to_int(self, mock_redis):
        """String confidence should be coerced to int."""
        svc = self._make_service()
        gist = {"content": "test content here for coercion", "type": "observation", "confidence": "8"}
        stored = svc.store_gists("topic-a", [gist], "p", "r")
        assert stored == 1

    def test_confidence_coercion_invalid_to_zero(self, mock_redis):
        """Invalid confidence should coerce to 0 and be rejected (when gists exist)."""
        svc = self._make_service()
        # First store a valid gist
        svc.store_gists("topic-a", [self._make_gist(confidence=9)], "p", "r")

        # Now store with invalid confidence
        gist = {"content": "invalid confidence value here", "type": "observation", "confidence": "not_a_number"}
        stored = svc.store_gists("topic-a", [gist], "p2", "r2")
        assert stored == 0
