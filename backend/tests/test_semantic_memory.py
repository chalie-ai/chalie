"""Tests for semantic memory system — storage, retrieval, relationships."""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.semantic_service import SemanticService


pytestmark = pytest.mark.unit


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def storage_env():
    """SemanticService with mocked DatabaseService.

    Returns (service, mock_cursor) so tests can set cursor behavior.
    """
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    with patch('services.config_service.ConfigService.get_agent_config', return_value={}):
        svc = SemanticService(mock_db, config={})
    return svc, mock_cursor


# ── Store concept ────────────────────────────────────────────────────

class TestStoreConcept:

    def test_requires_concept_name(self, storage_env):
        """Missing concept_name raises ValueError."""
        svc, _ = storage_env
        with pytest.raises(ValueError, match="Missing required field: concept_name"):
            svc.store_concept({'concept_type': 'knowledge', 'definition': 'test'})

    def test_requires_concept_type(self, storage_env):
        """Missing concept_type raises ValueError."""
        svc, _ = storage_env
        with pytest.raises(ValueError, match="Missing required field: concept_type"):
            svc.store_concept({'concept_name': 'test', 'definition': 'test'})

    def test_requires_definition(self, storage_env):
        """Missing definition raises ValueError."""
        svc, _ = storage_env
        with pytest.raises(ValueError, match="Missing required field: definition"):
            svc.store_concept({'concept_name': 'test', 'concept_type': 'knowledge'})

    def test_returns_uuid_on_success(self, storage_env):
        """Successful store returns a UUID string."""
        svc, mock_cursor = storage_env

        concept_data = {
            'concept_name': 'Python Programming',
            'concept_type': 'knowledge',
            'definition': 'A high-level programming language',
            'domain': 'programming',
            'confidence': 0.9,
        }

        result = svc.store_concept(concept_data)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 36  # UUID format


# ── Get concept ──────────────────────────────────────────────────────

class TestGetConcept:

    def test_returns_none_for_missing_concept(self, storage_env):
        """get_concept returns None when concept not found."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = None

        result = svc.get_concept('nonexistent-id')
        assert result is None

    def test_returns_concept_dict_on_success(self, storage_env):
        """get_concept returns a dict with expected keys."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = (
            'id-1', 'Python', 'knowledge', 'A language',
            3, 'programming', 0.5, 1.0,
            0, 0, 0.9, '[]',
            'unverified', '{}', '[]',
            None, None, None,
            0.0, 0.5, '2026-01-01', '2026-01-01'
        )

        result = svc.get_concept('id-1')
        assert result is not None
        assert result['concept_name'] == 'Python'
        assert result['concept_type'] == 'knowledge'
        assert result['confidence'] == 0.9


# ── Strengthen concept ───────────────────────────────────────────────

class TestStrengthenConcept:

    def test_returns_false_when_not_found(self, storage_env):
        """strengthen_concept returns False when concept doesn't exist."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = None

        result = svc.strengthen_concept('missing-id', 'ep-1')
        assert result is False

    def test_increases_strength(self, storage_env):
        """strengthen_concept multiplies strength by 1.1."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = (
            5.0,   # current strength
            '[]',  # source_episodes
            3,     # consolidation_count
            0.5,   # decay_resistance
        )

        result = svc.strengthen_concept('concept-1', 'ep-new')
        assert result is True

        # Verify UPDATE was called with new_strength ~5.5
        update_call = mock_cursor.execute.call_args_list[-1]
        params = update_call[0][1]
        assert abs(params[0] - 5.5) < 0.01  # 5.0 * 1.1


# ── Store relationship ───────────────────────────────────────────────

class TestStoreRelationship:

    def test_requires_fields(self, storage_env):
        """Missing required fields raise ValueError."""
        svc, _ = storage_env
        with pytest.raises(ValueError, match="Missing required field"):
            svc.store_relationship({'source_concept_id': 'a'})

    def test_creates_new_relationship(self, storage_env):
        """New relationship returns a UUID."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = None  # No existing relationship

        rel_data = {
            'source_concept_id': 'concept-1',
            'target_concept_id': 'concept-2',
            'relationship_type': 'related_to',
            'strength': 0.8,
        }

        result = svc.store_relationship(rel_data)
        assert result is not None
        assert isinstance(result, str)

    def test_strengthens_existing_relationship(self, storage_env):
        """Existing relationship gets strengthened instead of duplicated."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchone.return_value = ('rel-1', 0.5, '[]')

        rel_data = {
            'source_concept_id': 'concept-1',
            'target_concept_id': 'concept-2',
            'relationship_type': 'related_to',
        }

        result = svc.store_relationship(rel_data)
        assert result == 'rel-1'


# ── Get relationships ────────────────────────────────────────────────

class TestGetRelationships:

    def test_returns_empty_when_none(self, storage_env):
        """No relationships returns empty list."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchall.return_value = []

        result = svc.get_relationships('concept-1')
        assert result == []

    def test_returns_relationships_with_target_id_alias(self, storage_env):
        """Relationships include target_id alias for spreading activation compat."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchall.return_value = [
            ('rel-1', 'src-1', 'tgt-1', 'related_to', 0.8, False, '[]', 0.9)
        ]

        result = svc.get_relationships('src-1')
        assert len(result) == 1
        assert result[0]['target_id'] == 'tgt-1'
        assert result[0]['target_concept_id'] == 'tgt-1'


# ── Retrieval with confidence filtering ──────────────────────────────

class TestRetrieveConcepts:

    def test_returns_empty_when_no_concepts(self, storage_env):
        """Empty concept store returns []."""
        svc, mock_cursor = storage_env
        mock_cursor.fetchall.return_value = []

        result = svc.retrieve_concepts('test query', query_embedding=[0.0] * 256)
        assert result == []
