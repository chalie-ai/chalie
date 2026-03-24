"""Tests for world_awareness_service — interest extraction and news scanning."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.news_service import NewsArticle
from services.world_awareness_service import WorldAwarenessService


def _make_article(title="AI Breakthrough", **kwargs):
    defaults = {
        "title": title,
        "description": "New discovery",
        "url": "https://example.com/article",
        "published_at": "2026-03-24T10:00:00+00:00",
        "source": "TechCrunch",
        "source_id": "techcrunch",
        "category": "tech",
    }
    defaults.update(kwargs)
    return NewsArticle(**defaults)


def _make_db_mock(trait_rows=None, topic_rows=None):
    """Create a mock DB that returns specified trait and topic rows."""
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.get_connection.return_value = mock_conn

    call_count = [0]
    def execute_side_effect(*args, **kwargs):
        cursor = MagicMock()
        if call_count[0] == 0:
            cursor.fetchall.return_value = trait_rows or []
        else:
            cursor.fetchall.return_value = topic_rows or []
        call_count[0] += 1
        return cursor

    mock_conn.execute.side_effect = execute_side_effect
    return mock_db


# ── Trait extraction tests ────────────────────────────────────

@pytest.mark.unit
class TestTraitExtraction:

    def test_extracts_high_confidence_traits(self):
        trait_rows = [
            ("interest_ai", "artificial intelligence", 0.9, 5),
            ("profession", "software engineering", 0.85, 4),
        ]
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)
        result = svc._extract_trait_interests()

        assert len(result) == 2
        assert result[0]["term"] == "artificial intelligence"
        assert result[0]["score"] == 0.9 * 5
        assert result[0]["source"] == "trait"

    def test_skips_short_trait_values(self):
        trait_rows = [("key", "x", 0.9, 5)]  # too short
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)
        result = svc._extract_trait_interests()
        assert len(result) == 0

    def test_skips_long_trait_values(self):
        trait_rows = [("key", "x" * 101, 0.9, 5)]  # too long
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)
        result = svc._extract_trait_interests()
        assert len(result) == 0

    def test_db_error_returns_empty(self):
        mock_db = MagicMock()
        mock_db.get_connection.side_effect = Exception("DB down")
        svc = WorldAwarenessService(mock_db)
        result = svc._extract_trait_interests()
        assert result == []


# ── Topic extraction tests ────────────────────────────────────

@pytest.mark.unit
class TestTopicExtraction:

    def test_extracts_topic_frequency(self):
        topic_rows = [
            ("machine_learning", 10, "2026-03-24T10:00:00"),
            ("docker_compose", 5, "2026-03-23T10:00:00"),
        ]
        # For topic-only tests, put data in trait_rows position since
        # _extract_topic_interests is called independently
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = topic_rows
        mock_conn.execute.return_value = mock_cursor

        svc = WorldAwarenessService(mock_db)
        result = svc._extract_topic_interests()

        assert len(result) == 2
        assert result[0]["term"] == "machine learning"  # underscores replaced
        assert result[0]["score"] == 1.0  # highest freq
        assert result[1]["score"] == 0.5  # 5/10

    def test_normalizes_topic_names(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("natural_language_processing", 3, "2026-03-24T10:00:00")]
        mock_conn.execute.return_value = mock_cursor

        svc = WorldAwarenessService(mock_db)
        result = svc._extract_topic_interests()
        assert result[0]["term"] == "natural language processing"

    def test_db_error_returns_empty(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("DB down")
        svc = WorldAwarenessService(mock_db)
        result = svc._extract_topic_interests()
        assert result == []


# ── Embedding dedup tests ─────────────────────────────────────

@pytest.mark.unit
class TestEmbeddingDedup:

    def test_removes_near_duplicates(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)

        # Two identical vectors = similarity 1.0 > 0.85 threshold
        vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec, vec]
        svc._embedding_svc = mock_emb

        candidates = [
            {"term": "AI", "score": 5.0, "source": "trait"},
            {"term": "artificial intelligence", "score": 3.0, "source": "trait"},
        ]
        result = svc._deduplicate_by_embedding(candidates)
        assert len(result) == 1
        assert result[0]["term"] == "AI"  # highest score kept

    def test_keeps_different_interests(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)

        # Orthogonal vectors = similarity 0.0
        vec_a = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(768, dtype=np.float32)
        vec_b[1] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec_a, vec_b]
        svc._embedding_svc = mock_emb

        candidates = [
            {"term": "AI", "score": 5.0, "source": "trait"},
            {"term": "cooking", "score": 3.0, "source": "trait"},
        ]
        result = svc._deduplicate_by_embedding(candidates)
        assert len(result) == 2

    def test_single_candidate_unchanged(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)
        candidates = [{"term": "AI", "score": 5.0, "source": "trait"}]
        result = svc._deduplicate_by_embedding(candidates)
        assert len(result) == 1

    def test_empty_returns_empty(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)
        assert svc._deduplicate_by_embedding([]) == []

    def test_embedding_failure_returns_all(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.side_effect = Exception("Model error")
        svc._embedding_svc = mock_emb

        candidates = [
            {"term": "AI", "score": 5.0, "source": "trait"},
            {"term": "cooking", "score": 3.0, "source": "trait"},
        ]
        result = svc._deduplicate_by_embedding(candidates)
        assert len(result) == 2


# ── extract_interests integration ─────────────────────────────

@pytest.mark.unit
class TestExtractInterests:

    def test_combines_traits_and_topics(self):
        trait_rows = [("interest", "artificial intelligence", 0.9, 5)]
        topic_rows = [("cooking", 10, "2026-03-24T10:00:00")]
        db = _make_db_mock(trait_rows=trait_rows, topic_rows=topic_rows)
        svc = WorldAwarenessService(db)

        # Distinct embeddings to avoid dedup
        vec_a = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(768, dtype=np.float32)
        vec_b[1] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec_a, vec_b]
        svc._embedding_svc = mock_emb

        result = svc.extract_interests()
        assert len(result) == 2
        # Sorted by score desc
        assert result[0]["score"] >= result[1]["score"]

    def test_returns_empty_when_no_data(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)
        result = svc.extract_interests()
        assert result == []

    def test_capped_at_max_interests(self):
        # 10 traits + 10 topics = 20 candidates, should cap at 8
        trait_rows = [(f"key_{i}", f"interest {i}", 0.9, 5) for i in range(10)]
        topic_rows = [(f"topic_{i}", 10, "2026-03-24T10:00:00") for i in range(10)]
        db = _make_db_mock(trait_rows=trait_rows, topic_rows=topic_rows)
        svc = WorldAwarenessService(db)

        # All distinct embeddings
        vecs = []
        for i in range(20):
            v = np.zeros(768, dtype=np.float32)
            v[i % 768] = 1.0
            vecs.append(v)
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = vecs
        svc._embedding_svc = mock_emb

        result = svc.extract_interests()
        assert len(result) <= 8


# ── Scan tests ────────────────────────────────────────────────

@pytest.mark.unit
class TestScan:

    def test_scan_writes_signals(self):
        trait_rows = [("interest", "AI", 0.9, 5)]
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)

        # Mock embedding
        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec]
        svc._embedding_svc = mock_emb

        # Mock news service
        mock_news = MagicMock()
        mock_news.search.return_value = [_make_article()]
        svc._news_service = mock_news

        # Mock world state
        mock_world = MagicMock()
        svc._world_state_svc = mock_world

        count = svc.scan()
        assert count == 1
        mock_world.notify_external_signal.assert_called_once()

        # Check signal content
        call_kwargs = mock_world.notify_external_signal.call_args[1]
        assert call_kwargs["signal_type"] == "news"
        assert call_kwargs["source"] == "world_awareness"
        assert "AI Breakthrough" in call_kwargs["content"]

    def test_scan_skips_empty_interests(self):
        db = _make_db_mock()
        svc = WorldAwarenessService(db)
        count = svc.scan()
        assert count == 0

    def test_scan_skips_interests_with_no_articles(self):
        trait_rows = [("interest", "obscure topic", 0.9, 5)]
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)

        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec]
        svc._embedding_svc = mock_emb

        mock_news = MagicMock()
        mock_news.search.return_value = []
        svc._news_service = mock_news

        mock_world = MagicMock()
        svc._world_state_svc = mock_world

        count = svc.scan()
        assert count == 0
        mock_world.notify_external_signal.assert_not_called()

    def test_scan_continues_on_search_failure(self):
        trait_rows = [
            ("key1", "AI", 0.9, 5),
            ("key2", "cooking", 0.8, 4),
        ]
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)

        vec_a = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(768, dtype=np.float32)
        vec_b[1] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec_a, vec_b]
        svc._embedding_svc = mock_emb

        # First search fails, second succeeds
        mock_news = MagicMock()
        mock_news.search.side_effect = [Exception("Network error"), [_make_article()]]
        svc._news_service = mock_news

        mock_world = MagicMock()
        svc._world_state_svc = mock_world

        count = svc.scan()
        assert count == 1  # Only second interest succeeded

    def test_scan_activation_energy_capped(self):
        trait_rows = [("interest", "AI", 0.99, 100)]  # score = 99
        db = _make_db_mock(trait_rows=trait_rows)
        svc = WorldAwarenessService(db)

        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0
        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [vec]
        svc._embedding_svc = mock_emb

        mock_news = MagicMock()
        mock_news.search.return_value = [_make_article()]
        svc._news_service = mock_news

        mock_world = MagicMock()
        svc._world_state_svc = mock_world

        svc.scan()
        call_kwargs = mock_world.notify_external_signal.call_args[1]
        assert call_kwargs["activation_energy"] <= 0.6
