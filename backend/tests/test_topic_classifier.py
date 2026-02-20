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
    reg_instance = MagicMock()
    reg_instance.get_current_parameters.return_value = {
        'switch_threshold': 0.65,
        'decay_constant': 300,
        'w_semantic': 0.6,
        'w_freshness': 0.3,
        'w_salience': 0.1,
    }

    with patch('services.topic_classifier_service.DatabaseService') as mock_db_cls, \
         patch('services.topic_classifier_service.get_merged_db_config', return_value={}), \
         patch('services.topic_stability_regulator_service.TopicStabilityRegulator', return_value=reg_instance):

        mock_db_cls.return_value = mock_db

        from services.topic_classifier_service import TopicClassifierService
        svc = TopicClassifierService()
        svc.db = mock_db
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

        expected_keys = {'topic', 'confidence', 'switch_score', 'is_new_topic', 'classification_time'}
        assert expected_keys == set(result.keys())
