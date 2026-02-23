# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for AdaptiveLayerService.

All tests are pure-unit (no external DB/Redis calls).  External dependencies
are patched at the method level so the rules-engine logic is exercised in
complete isolation.

Coverage:
  - Cold-start gate (observation_count < 2 → empty)
  - Directive rules: low / mid / high threshold firing
  - Slot selection: pacing, cognitive (top-2), emotional (salience gate)
  - Challenge tolerance tier mapping + directive selection
  - Cognitive load estimation (LOW/NORMAL/HIGH/OVERLOAD)
  - Energy mirror directives (baseline mismatch)
  - Growth reflection (cooldown + consecutive_cycles gate)
  - Fork directive (mid-range ambiguity + cooldown)
  - Micro-preferences (label lookup)
  - generate_directives integration (happy path, empty-state path)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.adaptive_layer_service import (
    AdaptiveLayerService,
    DIRECTIVE_RULES,
    CHALLENGE_STYLE_TIERS,
    LOAD_DIRECTIVES,
    GROWTH_REFLECTIONS,
    PREF_LABELS,
    _MIN_OBSERVATION_COUNT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_style(**kwargs) -> dict:
    """Build a minimal valid communication style dict."""
    defaults = {
        '_observation_count': 5,
        'verbosity': 5,
        'directness': 5,
        'formality': 5,
        'abstraction_level': 5,
        'emotional_valence': 5,
        'certainty_level': 5,
        'challenge_appetite': 5,
        'depth_preference': 5,
        'pacing': 5,
    }
    defaults.update(kwargs)
    return defaults


def _make_turns(*contents, role='user') -> list:
    """Build a list of working-memory turn dicts."""
    return [{'role': role, 'content': c} for c in contents]


def _service() -> AdaptiveLayerService:
    return AdaptiveLayerService()


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start gate
# ─────────────────────────────────────────────────────────────────────────────

class TestColdStartGate:

    def test_no_style_returns_empty(self):
        svc = _service()
        with patch.object(svc, '_get_communication_style', return_value={}):
            result = svc.generate_directives()
        assert result == ""

    def test_observation_count_zero_returns_empty(self):
        svc = _service()
        style = _make_style(_observation_count=0)
        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None):
            result = svc.generate_directives()
        assert result == ""

    def test_observation_count_one_returns_empty(self):
        svc = _service()
        style = _make_style(_observation_count=1)
        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None):
            result = svc.generate_directives()
        assert result == ""

    def test_observation_count_at_threshold_passes(self):
        svc = _service()
        style = _make_style(_observation_count=_MIN_OBSERVATION_COUNT, pacing=2)
        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives()
        assert result != ""


# ─────────────────────────────────────────────────────────────────────────────
# Directive rule thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveDirective:

    @pytest.mark.parametrize("dim,value,expect_side", [
        ('verbosity',         2,   'low'),
        ('verbosity',         8,   'high'),
        ('verbosity',         5,   'none'),
        ('directness',        3,   'low'),
        ('directness',        9,   'high'),
        ('directness',        5.5, 'none'),
        ('pacing',            1,   'low'),
        ('pacing',            10,  'high'),
        ('depth_preference',  4,   'low'),
        ('depth_preference',  7,   'high'),
        ('emotional_valence', 4,   'low'),
        ('emotional_valence', 7,   'high'),
        ('certainty_level',   4,   'low'),
        ('certainty_level',   7,   'high'),
        ('challenge_appetite',4,   'low'),
        ('challenge_appetite',7,   'high'),
    ])
    def test_resolve_directive_thresholds(self, dim, value, expect_side):
        svc = _service()
        style = {dim: value}
        result = svc._resolve_directive(dim, style)
        rule = DIRECTIVE_RULES[dim]
        low_thresh, high_thresh, low_text, high_text = rule

        if expect_side == 'low':
            assert result == low_text
        elif expect_side == 'high':
            assert result == high_text
        else:
            assert result == ""

    def test_resolve_directive_missing_dim_returns_empty(self):
        svc = _service()
        assert svc._resolve_directive('verbosity', {}) == ""

    def test_resolve_directive_unknown_dim_returns_empty(self):
        svc = _service()
        assert svc._resolve_directive('nonexistent_dim', {'nonexistent_dim': 5}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Emotional slot
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionalSlot:

    def test_emotional_valence_high_salience_fires(self):
        svc = _service()
        # val=8, midpoint=5.5, salience=2.5 > 1.5 threshold
        style = _make_style(emotional_valence=8, certainty_level=5)
        result = svc._resolve_emotional_slot(style)
        assert result == DIRECTIVE_RULES['emotional_valence'][3]  # high_text

    def test_certainty_level_low_salience_fires(self):
        svc = _service()
        style = _make_style(certainty_level=2, emotional_valence=5)
        result = svc._resolve_emotional_slot(style)
        assert result == DIRECTIVE_RULES['certainty_level'][2]  # low_text

    def test_neutral_emotional_slot_returns_empty(self):
        svc = _service()
        style = _make_style(emotional_valence=5, certainty_level=5)
        result = svc._resolve_emotional_slot(style)
        assert result == ""

    def test_picks_higher_salience_when_both_extreme(self):
        svc = _service()
        # emotional_valence=9 (salience 3.5) vs certainty_level=2 (salience 3.5)
        style = _make_style(emotional_valence=9, certainty_level=2)
        result = svc._resolve_emotional_slot(style)
        # Both have same salience — either is valid; should not be empty
        assert result != ""


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive load estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveLoadEstimation:

    def test_empty_turns_returns_low(self):
        # score = 0 < 2 → LOW (no signals to raise it)
        svc = _service()
        assert svc._estimate_cognitive_load([]) == 'LOW'

    def test_single_turn_no_signal_returns_low(self):
        # No question marks, single turn → score 0 → LOW
        svc = _service()
        turns = _make_turns("Hello there.")
        assert svc._estimate_cognitive_load(turns) == 'LOW'

    def test_single_turn_with_question_marks_returns_normal(self):
        # "Hello, how are you today?" → 1 question mark, 5 words → ratio high → +2 → NORMAL
        svc = _service()
        turns = _make_turns("Hello, how are you today?")
        assert svc._estimate_cognitive_load(turns) == 'NORMAL'

    def test_declining_length_trend_produces_normal(self):
        # declining trend (+2) + short density (+1) = score 3 → NORMAL
        svc = _service()
        turns = _make_turns(
            "This is a really long message with a lot of words in it.",  # 12 words
            "Shorter message here with less.",                             # 5 words
            "Ok."                                                          # 1 word
        )
        result = svc._estimate_cognitive_load(turns)
        assert result == 'NORMAL'

    def test_declining_plus_question_density_produces_high(self):
        # 11 > 4 > 2 (declining, +2) + question density in last turn (+2) + short density (+1) = 5 → HIGH
        svc = _service()
        turns = _make_turns(
            "This is a very long message with many words in it here.",  # 11 words
            "Much shorter reply here.",                                    # 4 words
            "Why? How?",                                                   # 2 words, 2 question marks
        )
        result = svc._estimate_cognitive_load(turns)
        assert result in ('HIGH', 'OVERLOAD')

    def test_all_signals_combined_produces_high(self):
        # Same as above: 5 signals → HIGH
        svc = _service()
        turns = _make_turns(
            "This is a very long message with many words in it here.",
            "Much shorter reply here.",
            "Why? How?",
        )
        result = svc._estimate_cognitive_load(turns)
        assert result in ('HIGH', 'OVERLOAD')

    def test_all_signals_plus_micro_pref_produces_overload(self):
        # 5 signals + concise micro-pref (+1) = 6 → OVERLOAD
        svc = _service()
        turns = _make_turns(
            "This is a very long message with many words in it here.",
            "Much shorter reply here.",
            "Why? How?",
        )
        micro_prefs = ["User prefers concise, to-the-point responses."]
        result = svc._estimate_cognitive_load(turns, micro_prefs)
        assert result == 'OVERLOAD'

    def test_assistant_turns_are_ignored(self):
        # Long assistant turns contribute no score → LOW
        svc = _service()
        assistant_turns = [
            {'role': 'assistant', 'content': 'word ' * 100}
            for _ in range(5)
        ]
        result = svc._estimate_cognitive_load(assistant_turns)
        assert result == 'LOW'

    def test_stable_verbose_turns_returns_low(self):
        svc = _service()
        long_turns = _make_turns(*["This is a medium length sentence here with words." for _ in range(4)])
        assert svc._estimate_cognitive_load(long_turns) == 'LOW'


# ─────────────────────────────────────────────────────────────────────────────
# Challenge tier mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestChallengeTier:

    @pytest.mark.parametrize("tolerance,expected_tier", [
        (1.0,  'low'),
        (3.0,  'low'),
        (4.0,  'medium'),
        (5.5,  'medium'),
        (7.0,  'medium'),
        (8.0,  'high'),
        (10.0, 'high'),
    ])
    def test_tier_mapping(self, tolerance, expected_tier):
        assert AdaptiveLayerService._challenge_tier(tolerance) == expected_tier

    def test_tier_directive_content(self):
        for tier in ('low', 'medium', 'high'):
            assert CHALLENGE_STYLE_TIERS[tier]  # non-empty string


# ─────────────────────────────────────────────────────────────────────────────
# Energy mirror
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMirror:

    def test_negative_feedback_triggers_acknowledge(self):
        svc = _service()
        style = _make_style(verbosity=7)
        result = svc._get_energy_mirror_directive(style, {'explicit_feedback': 'negative'})
        assert 'friction' in result.lower()

    def test_verbose_user_sending_short_message(self):
        svc = _service()
        style = _make_style(verbosity=8)
        # ~8 words = ~10 tokens, which is < 10 word approximate trigger
        result = svc._get_energy_mirror_directive(style, {'prompt_token_count': 8})
        assert 'brief' in result.lower() or 'match' in result.lower()

    def test_terse_user_sending_long_message(self):
        svc = _service()
        style = _make_style(verbosity=2)
        result = svc._get_energy_mirror_directive(style, {'prompt_token_count': 200})
        assert 'deeper' in result.lower() or 'room' in result.lower()

    def test_no_deviation_returns_empty(self):
        svc = _service()
        style = _make_style(verbosity=5)
        result = svc._get_energy_mirror_directive(style, {'prompt_token_count': 30})
        assert result == ""

    def test_empty_style_returns_empty(self):
        svc = _service()
        result = svc._get_energy_mirror_directive({}, {'prompt_token_count': 5})
        assert result == ""

    def test_missing_token_count_returns_empty(self):
        svc = _service()
        style = _make_style(verbosity=8)
        result = svc._get_energy_mirror_directive(style, {})
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# Fork directive
# ─────────────────────────────────────────────────────────────────────────────

class TestForkDirective:

    def test_no_thread_id_returns_empty(self):
        svc = _service()
        style = _make_style(depth_preference=5)
        result = svc._get_fork_directive(style, None)
        assert result == ""

    def test_extreme_dimensions_do_not_fork(self):
        """When ALL fork-trigger dimensions are outside 4-7, no fork fires."""
        svc = _service()
        # Set all three FORK_TRIGGERS dims to extreme values
        style = _make_style(depth_preference=2, challenge_appetite=2, abstraction_level=1)

        mock_redis = MagicMock()
        mock_redis.exists.return_value = False

        with patch('services.redis_client.RedisClientService') as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            result = svc._get_fork_directive(style, 'thread-123')

        assert result == ""

    def test_ambiguous_dimension_triggers_fork(self):
        """A dimension at exactly 5.5 (closest to mid) should fire."""
        svc = _service()
        style = _make_style(depth_preference=5.5)  # dead center, most ambiguous

        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_redis.set = MagicMock()

        with patch('services.redis_client.RedisClientService') as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            result = svc._get_fork_directive(style, 'thread-123')

        assert result != ""
        # Should have set the pending key
        mock_redis.set.assert_called_once()

    def test_cooldown_blocks_fork(self):
        svc = _service()
        style = _make_style(depth_preference=5.5)

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True  # cooldown active

        with patch('services.redis_client.RedisClientService') as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            result = svc._get_fork_directive(style, 'thread-123')

        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# Growth reflection
# ─────────────────────────────────────────────────────────────────────────────

class TestGrowthReflection:

    def _mock_db_with_signals(self, signals: list):
        """signals: list of (trait_key, json_value_dict)"""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        rows = [(k, json.dumps(v)) for k, v in signals]
        mock_cursor.fetchall.return_value = rows
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn
        return mock_db

    # Patch paths for lazy imports inside the methods:
    # _get_growth_reflection imports from services.redis_client and services.database_service
    _REDIS_PATCH = 'services.redis_client.RedisClientService'
    _DB_PATCH = 'services.database_service.get_shared_db_service'

    def test_cooldown_active_returns_empty(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH):
            mock_cls.create_connection.return_value = mock_redis
            result = svc._get_growth_reflection('primary')

        assert result == ""

    def test_no_signals_returns_empty(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH) as mock_db_fn:
            mock_cls.create_connection.return_value = mock_redis
            mock_db_fn.return_value = self._mock_db_with_signals([])
            result = svc._get_growth_reflection('primary')

        assert result == ""

    def test_signal_below_threshold_returns_empty(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False

        signals = [
            ('growth_signal:certainty_level', {'consecutive_cycles': 3, 'direction': 'increasing'}),
        ]

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH) as mock_db_fn:
            mock_cls.create_connection.return_value = mock_redis
            mock_db_fn.return_value = self._mock_db_with_signals(signals)
            result = svc._get_growth_reflection('primary')

        assert result == ""

    def test_signal_at_threshold_returns_reflection(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_redis.set = MagicMock()

        signals = [
            ('growth_signal:certainty_level', {'consecutive_cycles': 6, 'direction': 'increasing'}),
        ]

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH) as mock_db_fn:
            mock_cls.create_connection.return_value = mock_redis
            mock_db_fn.return_value = self._mock_db_with_signals(signals)
            result = svc._get_growth_reflection('primary')

        assert result in GROWTH_REFLECTIONS['certainty_level']
        mock_redis.set.assert_called_once()

    def test_picks_strongest_signal(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_redis.set = MagicMock()

        signals = [
            ('growth_signal:certainty_level', {'consecutive_cycles': 7}),
            ('growth_signal:depth_preference', {'consecutive_cycles': 10}),  # stronger
        ]

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH) as mock_db_fn:
            mock_cls.create_connection.return_value = mock_redis
            mock_db_fn.return_value = self._mock_db_with_signals(signals)
            result = svc._get_growth_reflection('primary')

        assert result in GROWTH_REFLECTIONS['depth_preference']

    def test_unknown_dimension_returns_empty(self):
        svc = _service()
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False

        signals = [
            ('growth_signal:unknown_dim', {'consecutive_cycles': 99}),
        ]

        with patch(self._REDIS_PATCH) as mock_cls, \
             patch(self._DB_PATCH) as mock_db_fn:
            mock_cls.create_connection.return_value = mock_redis
            mock_db_fn.return_value = self._mock_db_with_signals(signals)
            result = svc._get_growth_reflection('primary')

        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# Micro-preference labels
# ─────────────────────────────────────────────────────────────────────────────

class TestMicroPreferences:

    def test_known_pref_key_returns_label(self):
        for key, label in PREF_LABELS.items():
            assert isinstance(label, str) and len(label) > 0

    def test_get_micro_preferences_maps_correctly(self):
        svc = _service()

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('prefers_bullet_format', 0.85),
            ('prefers_concise', 0.70),
            ('unknown_pref_key', 0.90),  # not in PREF_LABELS → excluded
        ]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = svc._get_micro_preferences('primary')

        assert PREF_LABELS['prefers_bullet_format'] in result
        assert PREF_LABELS['prefers_concise'] in result
        assert len(result) == 2  # unknown_pref_key excluded


# ─────────────────────────────────────────────────────────────────────────────
# generate_directives integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateDirectivesIntegration:

    def _patch_svc(self, svc, style=None, micro_prefs=None, challenge_tol=None,
                   fork="", growth=""):
        style = style or _make_style()
        return {
            '_get_communication_style': style,
            '_get_micro_preferences': micro_prefs or [],
            '_get_challenge_tolerance': challenge_tol,
            '_get_fork_directive': fork,
            '_get_growth_reflection': growth,
        }

    def test_output_starts_with_header(self):
        svc = _service()
        style = _make_style(pacing=2)  # extreme pacing → pacing directive fires
        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives()
        assert result.startswith("## Adaptive Response Style")

    def test_priority_note_always_appended(self):
        svc = _service()
        style = _make_style(pacing=2)
        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives()
        assert "identity voice" in result

    def test_high_load_directive_takes_first_slot(self):
        svc = _service()
        # 11 > 4 > 2 (declining +2) + question density +2 + short density +1 = 5 → HIGH
        turns = _make_turns(
            "This is a very long message with many words in it here.",
            "Much shorter reply here.",
            "Why? How?",
        )
        style = _make_style(_observation_count=5, pacing=5)  # neutral pacing so load wins first slot

        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives(working_memory_turns=turns)

        # HIGH load directive should appear
        assert any(kw in result for kw in ('summary', 'bullet', 'Compress', 'essentials'))

    def test_growth_reflection_appears_in_output(self):
        svc = _service()
        style = _make_style(pacing=2)
        reflection = "Your thinking is going deeper — exploring more layers."

        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=reflection):
            result = svc.generate_directives()

        assert reflection in result
        assert "If natural, weave in" in result

    def test_micro_prefs_capped_at_two(self):
        svc = _service()
        style = _make_style(pacing=2)
        prefs = [
            PREF_LABELS['prefers_bullet_format'],
            PREF_LABELS['prefers_concise'],
            PREF_LABELS['prefers_depth'],
        ]

        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=prefs), \
             patch.object(svc, '_get_challenge_tolerance', return_value=None), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives()

        # Only first 2 prefs should appear
        assert PREF_LABELS['prefers_bullet_format'] in result
        assert PREF_LABELS['prefers_concise'] in result
        assert PREF_LABELS['prefers_depth'] not in result

    def test_challenge_tolerance_supersedes_appetite(self):
        svc = _service()
        # challenge_appetite=9 (extreme high) + tolerance=2 (low tier)
        # → low tier directive should win
        style = _make_style(challenge_appetite=9)

        with patch.object(svc, '_get_communication_style', return_value=style), \
             patch.object(svc, '_get_micro_preferences', return_value=[]), \
             patch.object(svc, '_get_challenge_tolerance', return_value=2.0), \
             patch.object(svc, '_get_fork_directive', return_value=""), \
             patch.object(svc, '_get_growth_reflection', return_value=""):
            result = svc.generate_directives()

        # Low tier text should be present; raw high-appetite text should not
        assert CHALLENGE_STYLE_TIERS['low'] in result
        assert DIRECTIVE_RULES['challenge_appetite'][3] not in result  # high_text absent

    def test_exception_in_sub_method_returns_empty_gracefully(self):
        svc = _service()
        with patch.object(svc, '_get_communication_style', side_effect=RuntimeError("db down")):
            result = svc.generate_directives()
        assert result == ""
