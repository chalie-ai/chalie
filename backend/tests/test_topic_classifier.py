"""Tests for TopicClassifierService — classification, switch scoring."""

import struct
import uuid

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


pytestmark = pytest.mark.unit

# The test DB template uses 256-dim vec tables (see conftest._db_template).
_TEST_EMBED_DIM = 256


def _random_unit_vector(dim=_TEST_EMBED_DIM):
    v = np.random.randn(dim)
    return v / np.linalg.norm(v)


def _pack_embedding(embedding):
    """Pack a numpy or list embedding into a binary blob for topics_vec."""
    if hasattr(embedding, 'tolist'):
        data = embedding.tolist()
    else:
        data = list(embedding)
    return struct.pack(f'{len(data)}f', *data)


def _seed_topic(db, name, embedding, avg_salience, last_updated, message_count):
    """Insert a topic row and its companion topics_vec embedding."""
    topic_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO topics (topic_id, name, avg_salience, last_updated, message_count) VALUES (?, ?, ?, ?, ?)",
        (topic_id, name, avg_salience, last_updated.isoformat() if hasattr(last_updated, 'isoformat') else last_updated, message_count),
    )
    db.commit()
    # Retrieve the rowid for the companion vec table
    row = db.execute("SELECT rowid FROM topics WHERE name = ?", (name,)).fetchone()
    rowid = row[0]
    blob = _pack_embedding(embedding)
    db.execute("INSERT INTO topics_vec(rowid, embedding) VALUES (?, ?)", (rowid, blob))
    db.commit()


def _make_classifier(db):
    """Create a TopicClassifierService with the real test DB already patched."""
    with patch('services.topic_classifier_service.EmbeddingService'):
        from services.topic_classifier_service import TopicClassifierService
        svc = TopicClassifierService()
        return svc


class TestTopicClassifier:

    def test_new_topic_created_when_no_match(self, db):
        """No topics in DB -> creates new topic."""
        svc = _make_classifier(db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            mock_embed.return_value = _random_unit_vector()
            result = svc.classify("Tell me about machine learning algorithms")

        assert result['is_new_topic'] is True
        assert 'topic' in result
        assert result['confidence'] == 1.0

    def test_existing_topic_matched(self, db):
        """High cosine similarity -> returns existing topic."""
        topic_embedding = _random_unit_vector()

        _seed_topic(
            db,
            name='python-programming',
            embedding=topic_embedding,
            avg_salience=0.6,
            last_updated=datetime.now(timezone.utc) - timedelta(minutes=5),
            message_count=10,
        )
        svc = _make_classifier(db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            near_identical = topic_embedding + np.random.randn(_TEST_EMBED_DIM) * 0.01
            mock_embed.return_value = near_identical / np.linalg.norm(near_identical)

            result = svc.classify("Python programming basics")

        assert result['is_new_topic'] is False
        assert result['topic'] == 'python-programming'
        assert result['confidence'] > 0.9

    def test_switch_score_ranking(self, db):
        """Multiple candidates ranked correctly by switch_score."""
        emb1 = _random_unit_vector()
        emb2 = _random_unit_vector()
        now = datetime.now(timezone.utc)

        _seed_topic(db, 'topic-fresh', emb1, 0.5, now - timedelta(minutes=1), 5)
        _seed_topic(db, 'topic-stale', emb2, 0.5, now - timedelta(hours=2), 20)
        svc = _make_classifier(db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            near_emb1 = emb1 + np.random.randn(_TEST_EMBED_DIM) * 0.01
            mock_embed.return_value = near_emb1 / np.linalg.norm(near_emb1)

            result = svc.classify("related to topic fresh")

        assert result['topic'] == 'topic-fresh'

    def test_classification_returns_expected_keys(self, db):
        """Result dict has required keys."""
        svc = _make_classifier(db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            mock_embed.return_value = _random_unit_vector()
            result = svc.classify("Hello world")

        expected_keys = {'topic', 'confidence', 'switch_score', 'is_new_topic',
                         'classification_time', 'boundary_diagnostics',
                         'just_reset_from_silence', 'message_embedding'}
        assert expected_keys == set(result.keys())

    def test_two_signal_boundary_service_used_with_thread_id(self, db):
        """With thread_id, TwoSignalBoundaryService is consulted for boundary detection."""
        topic_embedding = _random_unit_vector()

        _seed_topic(
            db,
            name='existing-topic',
            embedding=topic_embedding,
            avg_salience=0.5,
            last_updated=datetime.now(timezone.utc) - timedelta(minutes=2),
            message_count=5,
        )
        svc = _make_classifier(db)

        mock_result = MagicMock()
        mock_result.is_boundary = True
        mock_result.just_reset_from_silence = False
        mock_result.trigger = 'marker'
        mock_result.confidence = 0.8
        mock_result.consec_sim = 0.3
        mock_result.window_sim = 0.35
        mock_result.marker_found = 'by the way'

        mock_detector = MagicMock()
        mock_detector.update.return_value = mock_result

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed, \
             patch('services.two_signal_boundary_service.TwoSignalBoundaryService',
                   return_value=mock_detector) as mock_cls:
            mock_embed.return_value = _random_unit_vector()
            result = svc.classify("By the way, what's the weather like?", thread_id='thread-abc')

        mock_cls.assert_called_once_with(thread_id='thread-abc')
        mock_detector.update.assert_called_once()
        mock_detector.save_state.assert_called_once()

        # Boundary fired -> new topic created
        assert result['is_new_topic'] is True
        assert result['just_reset_from_silence'] is False
        assert result['boundary_diagnostics']['trigger'] == 'marker'
        assert result['boundary_diagnostics']['marker_found'] == 'by the way'

    def test_two_signal_boundary_service_no_boundary(self, db):
        """With thread_id and no boundary fired, existing topic is matched."""
        topic_embedding = _random_unit_vector()

        _seed_topic(
            db,
            name='existing-topic',
            embedding=topic_embedding,
            avg_salience=0.5,
            last_updated=datetime.now(timezone.utc) - timedelta(minutes=2),
            message_count=5,
        )
        svc = _make_classifier(db)

        mock_result = MagicMock()
        mock_result.is_boundary = False
        mock_result.just_reset_from_silence = False
        mock_result.trigger = 'none'
        mock_result.confidence = 0.0
        mock_result.consec_sim = 0.85
        mock_result.window_sim = 0.88
        mock_result.marker_found = None

        mock_detector = MagicMock()
        mock_detector.update.return_value = mock_result

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed, \
             patch('services.two_signal_boundary_service.TwoSignalBoundaryService',
                   return_value=mock_detector):
            # Near-identical embedding -> high similarity -> same topic
            near_identical = topic_embedding + np.random.randn(_TEST_EMBED_DIM) * 0.01
            mock_embed.return_value = near_identical / np.linalg.norm(near_identical)
            result = svc.classify("More about existing topic", thread_id='thread-abc')

        assert result['is_new_topic'] is False
        assert result['topic'] == 'existing-topic'
        assert result['boundary_diagnostics']['trigger'] == 'none'
