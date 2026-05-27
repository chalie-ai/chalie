"""Feature tests for TKT-675: SkillSuggestionMessageProcessor subconscious invariants.

The spec for TKT-675 mandates six observable properties of the reworked
skill-suggestion system.  If any regress, this suite catches the failure
before the 300-second nightly test runs.

All tests are pure attribute / method checks on real production classes.
No mocks, no patches, no stubs.
"""

import pytest

pytestmark = pytest.mark.unit


def test_skill_suggestion_processor_channel():
    """CHANNEL must be 'skills_building' so transcript rows stay off the user channel."""
    from services.skill_suggestion_message_processor import SkillSuggestionMessageProcessor

    assert SkillSuggestionMessageProcessor.CHANNEL == 'skills_building'


def test_skill_suggestion_processor_skip_transcript_write_is_false():
    """SKIP_TRANSCRIPT_WRITE must be False so the processor writes a real transcript record."""
    from services.skill_suggestion_message_processor import SkillSuggestionMessageProcessor

    assert SkillSuggestionMessageProcessor.SKIP_TRANSCRIPT_WRITE is False


def test_skill_suggestion_processor_always_available_is_skill_manager():
    """ALWAYS_AVAILABLE must be ['skill_manager'] — skill_builder must NOT appear."""
    from services.skill_suggestion_message_processor import SkillSuggestionMessageProcessor

    assert SkillSuggestionMessageProcessor.ALWAYS_AVAILABLE == ['skill_manager']


def test_skill_suggestion_processor_get_previous_messages_returns_empty():
    """get_previous_messages() must return '' — each run is independent, no prior context."""
    from services.skill_suggestion_message_processor import SkillSuggestionMessageProcessor

    proc = SkillSuggestionMessageProcessor(
        original_trail=[],
        original_input='',
        iteration_count=0,
    )
    assert proc.get_previous_messages() == ''


def test_skill_manager_ability_name():
    """SkillManagerAbility.NAME must be 'skill_manager'."""
    from abilities.skill_manager import SkillManagerAbility

    assert SkillManagerAbility.NAME == 'skill_manager'


def test_skill_manager_ability_system_is_true():
    """SkillManagerAbility.SYSTEM must be True so policy enforcement is bypassed."""
    from abilities.skill_manager import SkillManagerAbility

    assert SkillManagerAbility.SYSTEM is True
