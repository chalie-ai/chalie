import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_associations — pure JSON parser, no IO
# ---------------------------------------------------------------------------


class TestParseAssociations:
    """_parse_associations(text) converts LLM JSON text into a list of dicts."""

    def test_valid_json_array_returns_list_of_dicts(self) -> None:
        """Plain JSON array is parsed correctly."""
        from services.skill_association_service import SkillAssociationService

        payload = json.dumps([
            {"skill_id": 1, "pattern_name": "dawn_meditation", "rule": "Schedule mindfulness tasks early morning."},
            {"skill_id": 3, "pattern_name": "evening_walk", "rule": "Plan physical activities post-work."},
        ])
        result = SkillAssociationService()._parse_associations(payload)

        assert result is not None
        assert len(result) == 2
        assert result[0]["skill_id"] == 1
        assert result[1]["pattern_name"] == "evening_walk"

    def test_fenced_code_block_is_stripped_before_parse(self) -> None:
        """LLM sometimes wraps output in ```json ... ``` — should parse cleanly."""
        from services.skill_association_service import SkillAssociationService

        payload = '```json\n[{"skill_id": 2, "pattern_name": "morning_run", "rule": "Prioritise physical tasks."}]\n```'
        result = SkillAssociationService()._parse_associations(payload)

        assert result is not None
        assert len(result) == 1
        assert result[0]["skill_id"] == 2

    def test_empty_array_returns_empty_list(self) -> None:
        """LLM returning [] is valid — means no patterns match any skills."""
        from services.skill_association_service import SkillAssociationService

        result = SkillAssociationService()._parse_associations("[]")

        assert result is not None
        assert result == []

    def test_non_list_json_returns_none(self) -> None:
        """If the LLM returns a dict instead of a list, return None."""
        from services.skill_association_service import SkillAssociationService

        result = SkillAssociationService()._parse_associations('{"error": "unexpected"}')

        assert result is None

    def test_malformed_json_returns_none(self) -> None:
        """Unparseable text must return None without raising."""
        from services.skill_association_service import SkillAssociationService

        result = SkillAssociationService()._parse_associations("this is not json at all")

        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string input must return None."""
        from services.skill_association_service import SkillAssociationService

        result = SkillAssociationService()._parse_associations("")

        assert result is None