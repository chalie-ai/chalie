"""Tests for TopicClassifierService — classification, switch scoring."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


pytestmark = pytest.mark.unit


def _make_db_mock(fetchall_return=None, fetchone_return=None):
    """Create a mock DB with proper connection -> cursor chain."""
    mock_db = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return

    conn = MagicMock()
    conn.cursor.return_value = cursor

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.connection.return_value = ctx

    return mock_db, cursor


def _make_classifier(mock_db):
    """Create a TopicClassifierService with mocked DB."""
    with patch('services.topic_classifier_service.get_shared_db_service', return_value=mock_db), \
         patch('services.topic_classifier_service.EmbeddingService'):

        from services.topic_classifier_service import TopicClassifierService
        svc = TopicClassifierService()
        return svc


def _random_unit_vector(dim=768):
    v = np.random.randn(dim)
    return v / np.linalg.norm(v)


class TestTopicClassifier:

    def test_new_topic_created_when_no_match(self):
        """No topics in DB → creates new topic."""
        mock_db, cursor = _make_db_mock(fetchall_return=[])
        svc = _make_classifier(mock_db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            mock_embed.return_value = _random_unit_vector()
            result = svc.classify("Tell me about machine learning algorithms")

        assert result['is_new_topic'] is True
        assert 'topic' in result
        assert result['confidence'] == 1.0

    def test_existing_topic_matched(self):
        """High cosine similarity → returns existing topic."""
        topic_embedding = _random_unit_vector()

        mock_db, cursor = _make_db_mock(fetchall_return=[
            ('python-programming', topic_embedding.tolist(), 0.6,
             datetime.now(timezone.utc) - timedelta(minutes=5), 10),
        ])
        svc = _make_classifier(mock_db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            near_identical = topic_embedding + np.random.randn(768) * 0.01
            mock_embed.return_value = near_identical / np.linalg.norm(near_identical)

            result = svc.classify("Python programming basics")

        assert result['is_new_topic'] is False
        assert result['topic'] == 'python-programming'
        assert result['confidence'] > 0.9

    def test_switch_score_ranking(self):
        """Multiple candidates ranked correctly by switch_score."""
        emb1 = _random_unit_vector()
        emb2 = _random_unit_vector()
        now = datetime.now(timezone.utc)

        mock_db, cursor = _make_db_mock(fetchall_return=[
            ('topic-fresh', emb1.tolist(), 0.5, now - timedelta(minutes=1), 5),
            ('topic-stale', emb2.tolist(), 0.5, now - timedelta(hours=2), 20),
        ])
        svc = _make_classifier(mock_db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            near_emb1 = emb1 + np.random.randn(768) * 0.01
            mock_embed.return_value = near_emb1 / np.linalg.norm(near_emb1)

            result = svc.classify("related to topic fresh")

        assert result['topic'] == 'topic-fresh'

    def test_classification_returns_expected_keys(self):
        """Result dict has required keys."""
        mock_db, cursor = _make_db_mock(fetchall_return=[])
        svc = _make_classifier(mock_db)

        with patch('services.topic_classifier_service.generate_embedding') as mock_embed:
            mock_embed.return_value = _random_unit_vector()
            result = svc.classify("Hello world")

        expected_keys = {'topic', 'confidence', 'switch_score', 'is_new_topic',
                         'classification_time', 'boundary_diagnostics',
                         'just_reset_from_silence', 'message_embedding'}
        assert expected_keys == set(result.keys())

    def test_two_signal_boundary_service_used_with_thread_id(self):
        """With thread_id, TwoSignalBoundaryService is consulted for boundary detection."""
        topic_embedding = _random_unit_vector()
        mock_db, cursor = _make_db_mock(fetchall_return=[
            ('existing-topic', topic_embedding.tolist(), 0.5,
             datetime.now(timezone.utc) - timedelta(minutes=2), 5),
        ])
        svc = _make_classifier(mock_db)

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

        # Boundary fired → new topic created
        assert result['is_new_topic'] is True
        assert result['just_reset_from_silence'] is False
        assert result['boundary_diagnostics']['trigger'] == 'marker'
        assert result['boundary_diagnostics']['marker_found'] == 'by the way'

    def test_two_signal_boundary_service_no_boundary(self):
        """With thread_id and no boundary fired, existing topic is matched."""
        topic_embedding = _random_unit_vector()
        mock_db, cursor = _make_db_mock(fetchall_return=[
            ('existing-topic', topic_embedding.tolist(), 0.5,
             datetime.now(timezone.utc) - timedelta(minutes=2), 5),
        ])
        svc = _make_classifier(mock_db)

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
            # Near-identical embedding → high similarity → same topic
            near_identical = topic_embedding + np.random.randn(768) * 0.01
            mock_embed.return_value = near_identical / np.linalg.norm(near_identical)
            result = svc.classify("More about existing topic", thread_id='thread-abc')

        assert result['is_new_topic'] is False
        assert result['topic'] == 'existing-topic'
        assert result['boundary_diagnostics']['trigger'] == 'none'
