"""
Unit tests for ContradictionClassifierService.

Tests cover:
  - check_concept_conflict (without LLM — using monkeypatching)
  - reconcile_memory_batch pairwise comparison
  - pair_already_tracked deduplication
  - sample_memories_for_reconcile DB query
"""

import json
import struct
import sqlite3
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.contradiction_classifier_service import (
    ContradictionClassifierService,
    _cosine_similarity,
    _unpack_embedding,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _schema_sql():
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema.sql')
    with open(schema_path, 'r') as f:
        return f.read()


def _pack_embedding(values: list) -> bytes:
    return struct.pack(f'{len(values)}f', *values)


class FakeDB:
    """In-memory SQLite wrapper matching the database_service interface."""

    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_schema_sql())

    def connection(self):
        return _Ctx(self.conn)


class _Ctx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
        self._conn.commit()


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCosineHelper:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_length(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


@pytest.mark.unit
class TestUnpackEmbedding:
    def test_bytes_round_trip(self):
        vals = [0.1, 0.2, 0.3]
        packed = _pack_embedding(vals)
        unpacked = _unpack_embedding(packed)
        assert len(unpacked) == 3
        for a, b in zip(vals, unpacked):
            assert abs(a - b) < 1e-5

    def test_none_returns_none(self):
        assert _unpack_embedding(None) is None

    def test_list_pass_through(self):
        vals = [0.5, 0.6]
        assert _unpack_embedding(vals) == vals


@pytest.mark.unit
class TestPairAlreadyTracked:
    def test_no_prior_uncertainty(self):
        db = FakeDB()
        svc = ContradictionClassifierService(db_service=db)
        assert svc.pair_already_tracked('trait-a', 'trait-b') is False

    def test_detects_existing_open_uncertainty(self):
        db = FakeDB()
        db.conn.execute("""
            INSERT INTO uncertainties
              (id, memory_a_type, memory_a_id, memory_b_type, memory_b_id,
               uncertainty_type, severity, detection_context)
            VALUES ('unc-1', 'trait', 'trait-a', 'trait', 'trait-b',
                    'contradiction', 'high', 'ingestion')
        """)
        db.conn.commit()
        svc = ContradictionClassifierService(db_service=db)
        assert svc.pair_already_tracked('trait-a', 'trait-b') is True

    def test_ignores_reversed_id_order(self):
        db = FakeDB()
        db.conn.execute("""
            INSERT INTO uncertainties
              (id, memory_a_type, memory_a_id, memory_b_type, memory_b_id,
               uncertainty_type, severity, detection_context)
            VALUES ('unc-2', 'trait', 'trait-b', 'trait', 'trait-a',
                    'contradiction', 'high', 'ingestion')
        """)
        db.conn.commit()
        svc = ContradictionClassifierService(db_service=db)
        assert svc.pair_already_tracked('trait-a', 'trait-b') is True

    def test_resolved_uncertainty_not_counted(self):
        db = FakeDB()
        db.conn.execute("""
            INSERT INTO uncertainties
              (id, memory_a_type, memory_a_id, memory_b_type, memory_b_id,
               uncertainty_type, severity, detection_context, state)
            VALUES ('unc-3', 'trait', 'trait-a', 'trait', 'trait-b',
                    'contradiction', 'high', 'ingestion', 'resolved')
        """)
        db.conn.commit()
        svc = ContradictionClassifierService(db_service=db)
        assert svc.pair_already_tracked('trait-a', 'trait-b') is False


@pytest.mark.unit
class TestReconcileMemoryBatch:
    def test_no_matches_below_threshold(self):
        """Orthogonal embeddings — no contradiction candidates."""
        svc = ContradictionClassifierService(db_service=None)
        memories = [
            {'id': 'a', 'type': 'trait', 'text': 'job: teacher', 'embedding': [1.0, 0.0], 'meta': {}},
            {'id': 'b', 'type': 'trait', 'text': 'pet: dog', 'embedding': [0.0, 1.0], 'meta': {}},
        ]
        # Should return empty — cosine similarity is 0, below threshold
        results = svc.reconcile_memory_batch(memories)
        assert results == []

    def test_skips_pairs_without_embeddings(self):
        svc = ContradictionClassifierService(db_service=None)
        memories = [
            {'id': 'a', 'type': 'trait', 'text': 'job: teacher', 'embedding': None, 'meta': {}},
            {'id': 'b', 'type': 'trait', 'text': 'job: doctor', 'embedding': None, 'meta': {}},
        ]
        results = svc.reconcile_memory_batch(memories)
        assert results == []

    def test_high_similarity_triggers_llm(self, monkeypatch):
        """Near-identical embeddings should trigger LLM call."""
        called = {}

        def fake_classify(self_inner, text_a, text_b, context_hint, meta_a, meta_b):
            called['invoked'] = True
            return {
                'classification': 'true_contradiction',
                'confidence': 0.9,
                'temporal_signal': False,
                'reasoning': 'cannot both be true',
                'surface_context': 'discuss career',
                'recommended_resolution': 'flag_response',
            }

        monkeypatch.setattr(
            ContradictionClassifierService,
            '_classify_pair_llm',
            fake_classify,
        )

        svc = ContradictionClassifierService(db_service=None)
        # High cosine similarity
        memories = [
            {'id': 'a', 'type': 'trait', 'text': 'job: teacher', 'embedding': [1.0, 0.0], 'meta': {}},
            {'id': 'b', 'type': 'trait', 'text': 'job: doctor', 'embedding': [0.99, 0.14], 'meta': {}},
        ]
        results = svc.reconcile_memory_batch(memories)
        assert called.get('invoked') is True
        assert len(results) == 1
        assert results[0]['classification'] == 'true_contradiction'


@pytest.mark.unit
class TestCheckConceptConflict:
    def test_compatible_returns_none(self, monkeypatch):
        def fake_classify(self_inner, text_a, text_b, context_hint, meta_a, meta_b):
            return {'classification': 'compatible', 'confidence': 0.8,
                    'temporal_signal': False, 'reasoning': 'ok', 'surface_context': None,
                    'recommended_resolution': 'ignore'}

        monkeypatch.setattr(ContradictionClassifierService, '_classify_pair_llm', fake_classify)
        svc = ContradictionClassifierService(db_service=None)
        result = svc.check_concept_conflict(
            concept_name='python',
            concept_definition='a programming language',
            existing={'id': 'c1', 'concept_name': 'python', 'definition': 'a snake'},
        )
        assert result is None

    def test_contradiction_returns_dict(self, monkeypatch):
        def fake_classify(self_inner, text_a, text_b, context_hint, meta_a, meta_b):
            return {'classification': 'true_contradiction', 'confidence': 0.9,
                    'temporal_signal': False, 'reasoning': 'conflict', 'surface_context': 'discussing python',
                    'recommended_resolution': 'flag_response'}

        monkeypatch.setattr(ContradictionClassifierService, '_classify_pair_llm', fake_classify)
        svc = ContradictionClassifierService(db_service=None)
        result = svc.check_concept_conflict(
            concept_name='python',
            concept_definition='a snake',
            existing={'id': 'c1', 'concept_name': 'python', 'definition': 'a programming language'},
        )
        assert result is not None
        assert result['classification'] == 'true_contradiction'


@pytest.mark.unit
class TestSampleMemoriesForReconcile:
    def test_returns_empty_without_data(self):
        db = FakeDB()
        svc = ContradictionClassifierService(db_service=db)
        result = svc.sample_memories_for_reconcile(n_traits=5, n_concepts=5)
        # May return empty list — no error
        assert isinstance(result, list)

    def test_none_db_returns_empty(self):
        svc = ContradictionClassifierService(db_service=None)
        assert svc.sample_memories_for_reconcile() == []
