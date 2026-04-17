"""Tests for the concept LUT lookup path in DataGraphService.

Covers KNN threshold gating, canonical key + rule retrieval, and the miss path.
Uses the real concept_lut.sqlite asset so tests reflect production behaviour.
Marked as integration because they require the sqlite-vec extension.
"""

import os
import sqlite3

import pytest

from services.data_graph_service import (
    _CONCEPT_LUT_PATH,
    _CONCEPT_LUT_THRESHOLD,
    _get_lut_conn,
    _l2_dist_to_cosine,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def reset_lut_singleton():
    """Reset the module-level LUT connection singleton before each test.

    Needed because _get_lut_conn() caches the connection after the first call,
    and tests that intentionally break the path would poison subsequent tests.
    """
    import services.data_graph_service as _mod
    original_conn = _mod._lut_conn
    original_loaded = _mod._lut_loaded
    _mod._lut_conn = None
    _mod._lut_loaded = False
    yield
    _mod._lut_conn = original_conn
    _mod._lut_loaded = original_loaded


class TestLutConnLoad:
    """Verify the lazy-load mechanism and logging."""

    def test_lut_file_exists(self):
        """concept_lut.sqlite must be present on disk (committed artifact)."""
        assert os.path.exists(_CONCEPT_LUT_PATH), (
            f"concept_lut.sqlite not found at {_CONCEPT_LUT_PATH}. "
            "Run: cd backend && python -m utils.generate_concept_lut"
        )

    def test_get_lut_conn_returns_connection(self):
        """_get_lut_conn() returns a live sqlite3 connection on a real build."""
        conn = _get_lut_conn()
        assert conn is not None, "LUT connection is None — sqlite not found or failed to load"



class TestL2DistToCosine:
    """Unit tests for the distance-to-similarity conversion helper."""

    def test_zero_distance_is_1(self):
        """L2 distance of 0 corresponds to identical vectors → cosine = 1.0."""
        assert _l2_dist_to_cosine(0.0) == pytest.approx(1.0)

    def test_large_distance_clamps_to_zero(self):
        """Distances that would produce negative similarity are clamped to 0."""
        assert _l2_dist_to_cosine(10.0) == 0.0

    def test_midpoint_distance(self):
        """sqrt(2) L2 distance → cosine = 0.0 for orthogonal unit vectors."""
        import math
        # For unit-norm vectors: cos = 1 - dist^2/2.  dist = sqrt(2) → cos = 0.
        assert _l2_dist_to_cosine(math.sqrt(2.0)) == pytest.approx(0.0, abs=1e-6)


class TestLutKnnLookup:
    """End-to-end KNN lookup tests against the real LUT asset."""

    def _embed(self, text: str):
        """Generate embedding via the production service (requires running model)."""
        from services.embedding_service import EmbeddingService
        return EmbeddingService().generate_embedding(text)

    def test_canonical_key_matches_itself(self):
        """Embedding of a canonical key should hit that same key with cos ≥ threshold."""
        conn = _get_lut_conn()
        if conn is None:
            pytest.skip("LUT not available")

        from services.embedding_utils import pack_embedding
        emb = self._embed("residence")
        blob = pack_embedding(emb)

        hits = conn.execute(
            "SELECT rowid, distance FROM lut_embeddings WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
            (blob,),
        ).fetchall()
        assert hits, "No hits returned for 'residence' — KNN broken"
        rowid, distance = hits[0]
        cos = _l2_dist_to_cosine(distance)

        row = conn.execute(
            "SELECT canonical_key, rule FROM lut_concepts WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == "residence", f"Expected 'residence', got {row[0]!r}"
        assert cos >= _CONCEPT_LUT_THRESHOLD, f"Self-match too low: cos={cos:.4f}"

    def test_alias_routes_to_correct_canonical(self):
        """'residency' (an alias) should resolve to canonical 'residence' as top-1 KNN hit.

        With 27 high-level concepts the alias may fall below the production threshold
        (LLM is the primary canonicalizer, LUT is a validator). This test verifies
        correct routing, not threshold gating.
        """
        conn = _get_lut_conn()
        if conn is None:
            pytest.skip("LUT not available")

        from services.embedding_utils import pack_embedding
        emb = self._embed("residency")
        blob = pack_embedding(emb)

        hits = conn.execute(
            "SELECT rowid, distance FROM lut_embeddings WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
            (blob,),
        ).fetchall()
        assert hits
        rowid, distance = hits[0]
        cos = _l2_dist_to_cosine(distance)

        row = conn.execute(
            "SELECT canonical_key, rule FROM lut_concepts WHERE id = ?", (rowid,)
        ).fetchone()
        assert row is not None
        assert row[0] == "residence", (
            f"Alias 'residency' should map to 'residence', got {row[0]!r} (cos={cos:.4f})"
        )
        assert cos >= 0.70, f"Alias cos suspiciously low: {cos:.4f}"

    def test_unrelated_key_below_threshold(self):
        """A niche domain-specific key should not hit any concept at or above the threshold."""
        conn = _get_lut_conn()
        if conn is None:
            pytest.skip("LUT not available")

        from services.embedding_utils import pack_embedding
        emb = self._embed("dryer_streak_bowling_score")
        blob = pack_embedding(emb)

        hits = conn.execute(
            "SELECT rowid, distance FROM lut_embeddings WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
            (blob,),
        ).fetchall()
        if not hits:
            return  # No hits at all — clearly a miss, test passes

        _, distance = hits[0]
        cos = _l2_dist_to_cosine(distance)
        assert cos < _CONCEPT_LUT_THRESHOLD, (
            f"Niche key 'dryer_streak_bowling_score' scored {cos:.4f} — threshold too low or LUT too broad"
        )

