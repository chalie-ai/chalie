import itertools
import pytest

pytestmark = pytest.mark.unit


class TestVoiceCorpus:
    def test_all_3125_tuples_return_non_trivial_voice(self):
        from services.personality.personality_service import personality_service
        for tup in itertools.product((-2, -1, 0, 1, 2), repeat=5):
            voice = personality_service.voice_for(tup)
            assert isinstance(voice, str) and len(voice) >= 60, f"Tuple {tup}: {voice!r}"
