"""Feature tests — personality voice in UserMessageProcessor.getSystemPrompt().

All tests are @pytest.mark.unit — real SQLite via the ``db`` fixture,
real voices.jsonl corpus, zero mocks for service internals.

Verifies:
  - UserMessageProcessor.getSystemPrompt() starts with the personality voice line.
  - Background processors (DMN, GoalPursuit, Scheduled, UserSummary,
    EpisodeEncoder, SuperEpisodeEncoder) do NOT emit the personality voice.
"""

import json
import os

import pytest

pytestmark = pytest.mark.unit


# ── Shared corpus helper ───────────────────────────────────────────────────────

_VOICES_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'services', 'personality', 'voices.jsonl',
)


def _load_corpus() -> dict:
    index = {}
    with open(_VOICES_PATH, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index[tuple(obj['tuple'])] = obj['voice']
    return index


# ── TestUserMessageProcessorVoiceLine ─────────────────────────────────────────


@pytest.mark.unit
class TestUserMessageProcessorVoiceLine:
    """getSystemPrompt() for UserMessageProcessor includes the personality voice line."""

    def test_user_system_prompt_starts_with_personality_voice_line(self, db):
        """Set personality to all-cool; system prompt starts with 'When responding; <voice>\\n\\n'."""
        corpus = _load_corpus()
        target = (-2, -2, -2, -2, -2)
        expected_voice = corpus[target]

        from services.personality.personality_service import set_current_tuple
        from services.user_message_processor import UserMessageProcessor

        set_current_tuple(target)

        proc = UserMessageProcessor(raw_input='hello')
        result = proc.getSystemPrompt()

        expected_prefix = f"When responding; {expected_voice}\n\n"
        assert result.startswith(expected_prefix), (
            f"System prompt does not start with the expected voice line.\n"
            f"Expected prefix: {expected_prefix!r}\n"
            f"Actual start:    {result[:len(expected_prefix) + 40]!r}"
        )

    def test_user_system_prompt_defaults_to_neutral_voice_when_unset(self, db):
        """With no personality setting, system prompt starts with the neutral voice line."""
        corpus = _load_corpus()
        neutral_voice = corpus[(0, 0, 0, 0, 0)]

        from services.user_message_processor import UserMessageProcessor

        proc = UserMessageProcessor(raw_input='hello')
        result = proc.getSystemPrompt()

        expected_prefix = f"When responding; {neutral_voice}\n\n"
        assert result.startswith(expected_prefix), (
            f"System prompt does not start with the neutral voice line.\n"
            f"Expected prefix: {expected_prefix!r}\n"
            f"Actual start:    {result[:len(expected_prefix) + 40]!r}"
        )

    def test_user_system_prompt_does_not_contain_old_static_peer_string(self, db):
        """The old static fallback 'Engage naturally as a peer.' must not appear in the prompt body."""
        from services.personality.personality_service import set_current_tuple
        from services.user_message_processor import UserMessageProcessor

        # Set a non-neutral personality so the fallback string is clearly wrong
        set_current_tuple((2, 2, 2, 2, 2))

        proc = UserMessageProcessor(raw_input='hello')
        result = proc.getSystemPrompt()

        # The static fallback should not be present as content in the system prompt.
        # Note: the fallback string only appears inside personality_service.py as a
        # defensive sentinel for unknown tuples, never in the assembled prompt.
        assert 'Engage naturally as a peer.' not in result, (
            "Old static voice string 'Engage naturally as a peer.' still present in system prompt"
        )


# ── TestBackgroundProcessorsExcludePersonality ────────────────────────────────


_BACKGROUND_PROCESSOR_CASES = [
    (
        'DMNMessageProcessor',
        'services.dmn_message_processor',
        {'raw_input': 'review context'},
    ),
    (
        'GoalPursuitProcessor',
        'services.goal_pursuit_processor',
        {'raw_input': 'pursue goal'},
    ),
    (
        'ScheduledMessageProcessor',
        'services.scheduled_message_processor',
        {'raw_input': 'run scheduled task'},
    ),
    (
        'UserSummaryProcessor',
        'services.user_summary_processor',
        {},  # no raw_input — takes metadata only
    ),
    (
        'EpisodeEncoderProcessor',
        'services.episode_encoder_processor',
        {'transcript_window': 'window text', 'referenced_episodes': ''},
    ),
    (
        'SuperEpisodeEncoderProcessor',
        'services.super_episode_encoder_processor',
        {'channel': 'test-channel'},
    ),
]


@pytest.mark.unit
class TestBackgroundProcessorsExcludePersonality:
    """Background processors must NOT include the personality voice in their system prompt."""

    @pytest.mark.parametrize(
        'cls_name,module_path,init_kwargs',
        _BACKGROUND_PROCESSOR_CASES,
        ids=[c[0] for c in _BACKGROUND_PROCESSOR_CASES],
    )
    def test_background_processor_does_not_include_personality_voice(
        self, db, cls_name, module_path, init_kwargs
    ):
        """Set personality to max-warm; background processor system prompt must not contain it."""
        corpus = _load_corpus()
        all_warm_voice = corpus[(2, 2, 2, 2, 2)]

        from services.personality.personality_service import set_current_tuple

        set_current_tuple((2, 2, 2, 2, 2))

        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        proc = cls(**init_kwargs)

        result = proc.getSystemPrompt()

        assert 'When responding;' not in result, (
            f"{cls_name}.getSystemPrompt() contains 'When responding;' — "
            f"personality voice line must only appear in UserMessageProcessor"
        )

        # No 40-char substring of the all-warm voice should appear in the background prompt.
        # Sample every 40-char window from the voice paragraph.
        for offset in range(0, len(all_warm_voice) - 39, 20):
            fragment = all_warm_voice[offset:offset + 40]
            assert fragment not in result, (
                f"{cls_name}.getSystemPrompt() contains a fragment of the all-warm voice: "
                f"{fragment!r}"
            )
