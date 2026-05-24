"""Unit test — personality voice injection boundary.

UMP starts with 'When responding;' — background processors do not.
"""

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _cold_cache():
    from services.personality.personality_service import personality_service
    personality_service.invalidate()
    yield
    personality_service.invalidate()


class TestVoiceInjectionBoundary:

    def test_ump_injects_voice(self, db):
        from services.user_message_processor import UserMessageProcessor
        assert UserMessageProcessor(raw_input='hi').get_system_prompt().startswith("When responding;")

    @pytest.mark.parametrize('cls_name,mod,kw', [
        ('DMNMessageProcessor', 'services.dmn_message_processor', {'raw_input': 'x'}),
        ('SubagentProcessor', 'services.subagent_processor', {'raw_input': 'x'}),
        ('UserSummaryProcessor', 'services.user_summary_processor', {}),
        ('EpisodeEncoderProcessor', 'services.episode_encoder_processor',
         {'transcript_window': 'w', 'referenced_episodes': ''}),
        ('SuperEpisodeEncoderProcessor', 'services.super_episode_encoder_processor',
         {'channel': 'c'}),
    ], ids=['DMN', 'Subagent', 'UserSummary', 'EpisodeEncoder', 'SuperEpisodeEncoder'])
    def test_background_excludes_voice(self, db, cls_name, mod, kw):
        import importlib
        cls = getattr(importlib.import_module(mod), cls_name)
        assert 'When responding;' not in cls(**kw).get_system_prompt()
