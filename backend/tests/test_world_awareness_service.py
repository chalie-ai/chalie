"""Tests for world_awareness_service — interest extraction and news scanning."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.database_service import get_shared_db_service
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


def _seed_trait(db, key, value, confidence, evidence_count, category='preference'):
    """Insert a knowledge row for a high-confidence user trait."""
    db.execute(
        """INSERT INTO knowledge (kind, entity, key, value, confidence, evidence_count,
                                  data, deleted_at)
           VALUES ('trait', 'user', ?, ?, ?, ?, ?, NULL)""",
        (key, value, confidence, evidence_count, json.dumps({"category": category})),
    )
    db.commit()


def _seed_topic_transcript(db, topic, count, last_ts="2026-03-24T10:00:00"):
    """Insert `count` user transcript rows for a given channel."""
    for i in range(count):
        db.execute(
            """INSERT INTO transcript (channel, role, content, created_at)
               VALUES (?, 'user', ?, ?)""",
            (topic, f"message {i}", last_ts),
        )
    db.commit()


def _make_service(db):
    """Create a WorldAwarenessService backed by the real test DB."""
    db_service = get_shared_db_service()
    return WorldAwarenessService(db_service)


# ── Trait extraction tests ────────────────────────────────────

@pytest.mark.unit
class TestTraitExtraction:

    def test_extracts_high_confidence_traits(self, db):
        _seed_trait(db, "interest_ai", "artificial intelligence", 0.9, 5)
        _seed_trait(db, "profession", "software engineering", 0.85, 4)

        svc = _make_service(db)
        result = svc._extract_trait_interests()

        assert len(result) == 2
        assert result[0]["term"] == "artificial intelligence"
        assert result[0]["score"] == 0.9 * 5
        assert result[0]["source"] == "trait"

    def test_skips_short_trait_values(self, db):
        _seed_trait(db, "key_short", "x", 0.9, 5)

        svc = _make_service(db)
        result = svc._extract_trait_interests()
        assert len(result) == 0

    def test_skips_long_trait_values(self, db):
        _seed_trait(db, "key_long", "x" * 101, 0.9, 5)

        svc = _make_service(db)
        result = svc._extract_trait_interests()
        assert len(result) == 0

    def test_db_error_returns_empty(self, db):
        # Use a broken DB service to simulate failure
        broken_db = MagicMock()
        broken_db.get_connection.side_effect = Exception("DB down")
        svc = WorldAwarenessService(broken_db)
        result = svc._extract_trait_interests()
        assert result == []


# ── Topic extraction tests ────────────────────────────────────

@pytest.mark.unit
class TestTopicExtraction:

    def test_extracts_topic_frequency(self, db):
        _seed_topic_transcript(db, "machine_learning", 10)
        _seed_topic_transcript(db, "docker_compose", 5)

        svc = _make_service(db)
        result = svc._extract_topic_interests()

        assert len(result) == 2
        assert result[0]["term"] == "machine learning"  # underscores replaced
        assert result[0]["score"] == 1.0  # highest freq
        assert result[1]["score"] == 0.5  # 5/10

    def test_normalizes_topic_names(self, db):
        _seed_topic_transcript(db, "natural_language_processing", 3)

        svc = _make_service(db)
        result = svc._extract_topic_interests()
        assert result[0]["term"] == "natural language processing"

    def test_db_error_returns_empty(self, db):
        broken_db = MagicMock()
        broken_conn = MagicMock()
        broken_db.get_connection.return_value = broken_conn
        broken_conn.execute.side_effect = Exception("DB down")
        svc = WorldAwarenessService(broken_db)
        result = svc._extract_topic_interests()
        assert result == []


# ── Embedding dedup tests ─────────────────────────────────────

@pytest.mark.unit
class TestEmbeddingDedup:

    def test_removes_near_duplicates(self, db):
        svc = _make_service(db)

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

    def test_keeps_different_interests(self, db):
        svc = _make_service(db)

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

    def test_single_candidate_unchanged(self, db):
        svc = _make_service(db)
        candidates = [{"term": "AI", "score": 5.0, "source": "trait"}]
        result = svc._deduplicate_by_embedding(candidates)
        assert len(result) == 1

    def test_empty_returns_empty(self, db):
        svc = _make_service(db)
        assert svc._deduplicate_by_embedding([]) == []

    def test_embedding_failure_returns_all(self, db):
        svc = _make_service(db)
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

    def test_combines_traits_and_topics(self, db):
        _seed_trait(db, "interest", "artificial intelligence", 0.9, 5)
        _seed_topic_transcript(db, "cooking", 10)

        svc = _make_service(db)

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

    def test_returns_empty_when_no_data(self, db):
        svc = _make_service(db)
        result = svc.extract_interests()
        assert result == []

    def test_capped_at_max_interests(self, db):
        # 10 traits + 10 topics = 20 candidates, should cap at 8
        for i in range(10):
            _seed_trait(db, f"key_{i}", f"interest number {i}", 0.9, 5, category='preference')
        for i in range(10):
            _seed_topic_transcript(db, f"topic_{i}", 10)

        svc = _make_service(db)

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

    def test_scan_writes_signals(self, db):
        _seed_trait(db, "interest", "AI", 0.9, 5)

        svc = _make_service(db)

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

    def test_scan_skips_empty_interests(self, db):
        svc = _make_service(db)
        count = svc.scan()
        assert count == 0

    def test_scan_skips_interests_with_no_articles(self, db):
        _seed_trait(db, "interest", "obscure topic", 0.9, 5)

        svc = _make_service(db)

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

    def test_scan_continues_on_search_failure(self, db):
        _seed_trait(db, "key1", "AI", 0.9, 5)
        _seed_trait(db, "key2", "cooking", 0.8, 4)

        svc = _make_service(db)

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

    def test_scan_activation_energy_capped(self, db):
        _seed_trait(db, "interest", "AI", 0.99, 100)  # score = 99

        svc = _make_service(db)

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
