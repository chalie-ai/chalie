"""
Unit tests for AutobiographyService.

Tests autobiography narrative retrieval, synthesis thresholds, and input gathering.
Mocks database connections.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from services.autobiography_service import AutobiographyService


@pytest.mark.unit
class TestAutobiographyService:
    """Test AutobiographyService methods."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database service."""
        db = Mock()
        db.get_session = MagicMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create an AutobiographyService instance with mocked database."""
        return AutobiographyService(mock_db)

    def test_service_initialization(self, mock_db):
        """Service should initialize without errors."""
        service = AutobiographyService(mock_db)
        assert service.db is mock_db

    def test_get_current_narrative_returns_none_when_empty(self, mock_db, service):
        """get_current_narrative should return None when no autobiography exists."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None
        mock_session.execute.return_value.fetchone.return_value = None

        result = service.get_current_narrative()

        assert result is None

    def test_get_current_narrative_returns_latest_version(self, mock_db, service):
        """get_current_narrative should return the latest version when it exists."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        # Mock the row returned from database
        test_narrative = "Test narrative content"
        test_created = datetime.now()
        mock_session.execute.return_value.fetchone.return_value = (
            "test-id-123",
            2,  # version
            test_narrative,
            test_created,
            5  # episodes_since
        )

        result = service.get_current_narrative()

        assert result is not None
        assert result["version"] == 2
        assert result["narrative"] == test_narrative
        assert result["episodes_since"] == 5

    def test_build_synthesis_prompt_includes_all_inputs(self, mock_db, service):
        """_build_synthesis_prompt should format all input data."""
        inputs = {
            "episodes": [
                {"gist": "Test episode", "emotion": "positive", "topic": "work"}
            ],
            "traits": [
                {"key": "trait1", "value": "value1", "confidence": 0.9, "category": "cat1"}
            ],
            "concepts": [
                {"name": "concept1", "definition": "def1", "strength": 0.8, "domain": "dom1"}
            ],
            "relationships": []
        }

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "Test episode" in prompt
        assert "trait1" in prompt
        assert "concept1" in prompt
        assert "positive" in prompt

    def test_build_synthesis_prompt_includes_current_narrative_for_incremental(self, mock_db, service):
        """_build_synthesis_prompt should include current narrative for incremental updates."""
        inputs = {"episodes": [], "traits": [], "concepts": [], "relationships": []}
        current = {
            "narrative": "Current narrative text",
            "version": 1
        }

        prompt = service._build_synthesis_prompt(inputs, current)

        assert "Current Narrative" in prompt
        assert "New Episodes" in prompt

    def test_gather_synthesis_inputs_returns_structure(self, mock_db, service):
        """gather_synthesis_inputs should return dict with expected keys."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        # Mock all the fetchall() calls for different result sets
        mock_session.execute.return_value.fetchall.side_effect = [
            [],  # episodes
            [],  # traits
            [],  # concepts
            [],  # relationships
            [],  # constraint_episodes
        ]

        result = service.gather_synthesis_inputs()

        assert "episodes" in result
        assert "traits" in result
        assert "concepts" in result
        assert "relationships" in result
        assert "constraint_episodes" in result
        assert isinstance(result["episodes"], list)

    def test_gather_synthesis_inputs_truncates_long_gists(self, mock_db, service):
        """gather_synthesis_inputs should truncate gists to 500 chars."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        long_gist = "x" * 600  # 600 chars
        episode_rows = [
            (long_gist, "action", "outcome", "positive", 0.8, "topic", datetime.now()),
        ]

        mock_session.execute.return_value.fetchall.side_effect = [
            episode_rows,
            [],  # traits
            [],  # concepts
            [],  # relationships
            [],  # constraint_episodes
        ]

        result = service.gather_synthesis_inputs()

        assert len(result["episodes"]) == 1
        assert len(result["episodes"][0]["gist"]) == 503  # 500 + "..."
        assert result["episodes"][0]["gist"].endswith("...")

    def test_gather_synthesis_inputs_concepts_row_mapping(self, mock_db, service):
        """Concept rows should map to correct dict keys/values."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        mock_session.execute.return_value.fetchall.side_effect = [
            [],  # episodes
            [],  # traits
            [("Python", "knowledge", "A programming language", "tech", 0.9)],  # concepts
            [],  # relationships
            [],  # constraint_episodes
        ]

        result = service.gather_synthesis_inputs()

        assert len(result["concepts"]) == 1
        assert result["concepts"][0] == {
            "name": "Python",
            "type": "knowledge",
            "definition": "A programming language",
            "domain": "tech",
            "strength": 0.9,
        }

    def test_gather_synthesis_inputs_relationships_row_mapping(self, mock_db, service):
        """Relationship rows should map to correct dict keys/values."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        mock_session.execute.return_value.fetchall.side_effect = [
            [],  # episodes
            [],  # traits
            [],  # concepts
            [("Python", "Flask", "uses", 0.85)],  # relationships
            [],  # constraint_episodes
        ]

        result = service.gather_synthesis_inputs()

        assert len(result["relationships"]) == 1
        assert result["relationships"][0] == {
            "source": "Python",
            "target": "Flask",
            "type": "uses",
            "strength": 0.85,
        }

    def test_gather_synthesis_inputs_sql_column_names(self, mock_db, service):
        """SQL queries must use correct column names from the actual schema."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        mock_session.execute.return_value.fetchall.side_effect = [
            [],  # episodes
            [],  # traits
            [],  # concepts
            [],  # relationships
            [],  # constraint_episodes
        ]

        service.gather_synthesis_inputs()

        # Capture all SQL strings passed to session.execute()
        sql_calls = [
            str(call.args[0]) for call in mock_session.execute.call_args_list
        ]

        # Find the concepts query (selects from semantic_concepts)
        concepts_sql = [s for s in sql_calls if 'semantic_concepts' in s and 'JOIN' not in s]
        assert len(concepts_sql) == 1, "Expected exactly one semantic_concepts query"
        assert 'concept_type' in concepts_sql[0], "Concepts query must use 'concept_type', not 'type'"
        assert 'user_id' not in concepts_sql[0], "Concepts query must not reference 'user_id'"

        # Find the relationships query (joins semantic_relationships)
        rels_sql = [s for s in sql_calls if 'semantic_relationships' in s]
        assert len(rels_sql) == 1, "Expected exactly one semantic_relationships query"
        assert 'source_concept_id' in rels_sql[0], "Relationships query must join on 'source_concept_id'"
        assert 'target_concept_id' in rels_sql[0], "Relationships query must join on 'target_concept_id'"
        assert 'relationship_type' in rels_sql[0], "Relationships query must select 'relationship_type'"
        assert 'user_id' not in rels_sql[0], "Relationships query must not reference 'user_id'"

    def test_gather_synthesis_inputs_includes_constraint_episodes(self, mock_db, service):
        """gather_synthesis_inputs should query constraint_learned episodes."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_session.return_value.__exit__.return_value = None

        constraint_rows = [
            ("Attempted communicate 15 times; blocked by timing_gate",
             "learned constraint: communicate blocked by timing_gate",
             "2026-03-07T12:00:00+00:00", 3),
        ]

        mock_session.execute.return_value.fetchall.side_effect = [
            [],  # episodes
            [],  # traits
            [],  # concepts
            [],  # relationships
            constraint_rows,  # constraint_episodes
        ]

        result = service.gather_synthesis_inputs()

        assert "constraint_episodes" in result
        assert len(result["constraint_episodes"]) == 1
        assert "communicate" in result["constraint_episodes"][0]["gist"]
        assert result["constraint_episodes"][0]["activation_score"] == 3

    def test_build_synthesis_prompt_includes_constraint_episodes(self, mock_db, service):
        """_build_synthesis_prompt should include constraint episodes when present."""
        inputs = {
            "episodes": [],
            "traits": [],
            "concepts": [],
            "relationships": [],
            "constraint_episodes": [
                {
                    "gist": "Attempted communicate 15 times; blocked by timing_gate",
                    "action": "learned constraint",
                    "created_at": "2026-03-07T12:00:00",
                    "activation_score": 3,
                }
            ],
        }

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "Learned Constraints" in prompt
        assert "communicate" in prompt
        assert "timing_gate" in prompt

    def test_build_synthesis_prompt_omits_constraints_when_empty(self, mock_db, service):
        """_build_synthesis_prompt should omit constraints section when no data."""
        inputs = {
            "episodes": [],
            "traits": [],
            "concepts": [],
            "relationships": [],
            "constraint_episodes": [],
        }

        prompt = service._build_synthesis_prompt(inputs, None)

        assert "Learned Constraints" not in prompt
