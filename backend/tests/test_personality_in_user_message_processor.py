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

