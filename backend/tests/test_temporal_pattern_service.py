"""
Tests for TemporalPatternService — temporal behavioral pattern mining.

All tests are unit tests with no external dependencies (DB calls are mocked).
Tests cover statistical detection logic, privacy-preserving label generation,
and deduplication via store_trait().
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestDetectPeakHours:

    def setup_method(self):
        from services.temporal_pattern_service import TemporalPatternService
        self.service = TemporalPatternService(MagicMock())

    def test_detects_significant_peak(self):
        """Hour with 2x average frequency should be detected."""
        dist = {h: 5 for h in range(24)}  # average = 5
        dist[10] = 30  # 6x average, well above threshold
        dist[11] = 25
        patterns = self.service._detect_peak_hours(dist)
        assert len(patterns) > 0

    def test_peak_label_is_generalized(self):
        """Peak label should be a broad time-of-day label, never a raw hour."""
        dist = {h: 2 for h in range(24)}
        dist[10] = 40  # morning peak
        dist[11] = 35
        patterns = self.service._detect_peak_hours(dist)
        assert len(patterns) > 0
        assert 'morning' in patterns[0]['value']

    def test_insufficient_data_returns_empty(self):
        """Less than MIN_OBSERVATIONS total should return no patterns."""
        dist = {10: 2, 11: 1}  # total = 3 < 10
        patterns = self.service._detect_peak_hours(dist)
        assert patterns == []

    def test_no_significant_peak_returns_empty(self):
        """Uniform distribution should return no patterns."""
        dist = {h: 10 for h in range(24)}  # no peaks
        patterns = self.service._detect_peak_hours(dist)
        assert patterns == []

    def test_confidence_capped_at_0_9(self):
        """Confidence should never exceed 0.9 (avoid false certainty)."""
        dist = {h: 1 for h in range(24)}
        dist[10] = 500  # extreme peak
        patterns = self.service._detect_peak_hours(dist)
        if patterns:
            assert patterns[0]['confidence'] <= 0.9

    def test_key_is_active_hours(self):
        """Pattern key should be 'active_hours'."""
        dist = {h: 2 for h in range(24)}
        dist[10] = 40
        dist[11] = 35
        patterns = self.service._detect_peak_hours(dist)
        if patterns:
            assert patterns[0]['key'] == 'active_hours'


@pytest.mark.unit
class TestDetectPeakDays:

    def setup_method(self):
        from services.temporal_pattern_service import TemporalPatternService
        self.service = TemporalPatternService(MagicMock())

    def test_detects_peak_day(self):
        """Day with 2x average frequency should be detected."""
        dist = {d: 5 for d in range(7)}
        dist[0] = 20  # Monday, 4x average
        patterns = self.service._detect_peak_days(dist)
        assert len(patterns) > 0
        assert 'Monday' in patterns[0]['value']

    def test_insufficient_data_returns_empty(self):
        dist = {0: 2, 1: 1}  # total = 3 < 10
        patterns = self.service._detect_peak_days(dist)
        assert patterns == []

    def test_uniform_distribution_returns_empty(self):
        dist = {d: 10 for d in range(7)}
        patterns = self.service._detect_peak_days(dist)
        assert patterns == []

    def test_key_is_active_days(self):
        dist = {d: 5 for d in range(7)}
        dist[4] = 25  # Friday peak
        patterns = self.service._detect_peak_days(dist)
        if patterns:
            assert patterns[0]['key'] == 'active_days'

    def test_multiple_peak_days_combined(self):
        dist = {d: 3 for d in range(7)}
        dist[5] = 20  # Saturday
        dist[6] = 18  # Sunday
        patterns = self.service._detect_peak_days(dist)
        assert len(patterns) == 1  # Combined into one pattern
        assert 'Saturday' in patterns[0]['value']
        assert 'Sunday' in patterns[0]['value']


@pytest.mark.unit
class TestHourToLabel:

    def test_morning(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(10) == "morning"

    def test_night(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(22) == "night"

    def test_late_night(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(2) == "late night"

    def test_all_hours_map_to_label(self):
        """Every hour (0-23) should return a named label — never expose raw numbers."""
        from services.temporal_pattern_service import TemporalPatternService
        valid_labels = {"early morning", "morning", "midday", "afternoon",
                        "evening", "night", "late night", "daytime"}
        for h in range(24):
            label = TemporalPatternService._hour_to_label(h)
            assert label in valid_labels, f"Hour {h} mapped to unknown label '{label}'"

    def test_no_colon_in_label(self):
        """Labels must never contain raw time formats like '10:00'."""
        from services.temporal_pattern_service import TemporalPatternService
        for h in range(24):
            assert ":" not in TemporalPatternService._hour_to_label(h)


@pytest.mark.unit
class TestGroupContiguous:

    def test_single_element(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([5]) == [[5]]

    def test_contiguous_group(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([9, 10, 11]) == [[9, 10, 11]]

    def test_two_groups(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([9, 10, 11, 14, 15]) == [[9, 10, 11], [14, 15]]

    def test_empty(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([]) == []

    def test_non_contiguous_single_elements(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._group_contiguous([1, 3, 5])
        assert result == [[1], [3], [5]]


@pytest.mark.unit
class TestSlugifyTopic:

    def test_basic_slugification(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._slugify_topic("My Cooking Adventures!") == "my_cooking_adventures"

    def test_max_length_capped(self):
        from services.temporal_pattern_service import TemporalPatternService
        long = "a" * 100
        result = TemporalPatternService._slugify_topic(long)
        assert len(result) <= 40

    def test_special_chars_removed(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._slugify_topic("hello/world#test")
        assert '/' not in result
        assert '#' not in result

    def test_leading_trailing_underscores_removed(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._slugify_topic("  hello world  ")
        assert not result.startswith('_')
        assert not result.endswith('_')


@pytest.mark.unit
class TestStorePatternsAsTraits:

    def test_calls_store_trait_for_each_pattern(self):
        """Each pattern should be stored via store_trait()."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()
        mock_trait_svc.store_trait.return_value = True

        patterns = [
            {'key': 'active_hours', 'value': 'Most active in the evenings', 'confidence': 0.7},
            {'key': 'active_days', 'value': 'Most active on Monday', 'confidence': 0.7},
        ]

        # UserTraitService is a lazy import inside _store_patterns_as_traits,
        # so patch the source module, not the temporal_pattern_service module.
        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits(patterns, 'primary')

        assert mock_trait_svc.store_trait.call_count == 2

    def test_passes_correct_arguments_to_store_trait(self):
        """store_trait() should be called with category='behavioral_pattern' and source='inferred'."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()
        mock_trait_svc.store_trait.return_value = True

        patterns = [{'key': 'active_hours', 'value': 'Most active in the mornings', 'confidence': 0.7}]

        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits(patterns, 'primary')

        call_kwargs = mock_trait_svc.store_trait.call_args.kwargs
        assert call_kwargs['category'] == 'behavioral_pattern'
        assert call_kwargs['source'] == 'inferred'
        assert call_kwargs['trait_key'] == 'active_hours'
        assert call_kwargs['user_id'] == 'primary'

    def test_empty_patterns_does_not_call_store(self):
        """No patterns → store_trait should never be called."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits([], 'primary')

        mock_trait_svc.store_trait.assert_not_called()

    def test_error_does_not_raise(self):
        """Storage failure should be swallowed (logged, not raised)."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        with patch('services.user_trait_service.UserTraitService', side_effect=Exception("db error")):
            # Should not raise
            service._store_patterns_as_traits(
                [{'key': 'k', 'value': 'v', 'confidence': 0.5}], 'primary'
            )


@pytest.mark.unit
class TestBehavioralPatternDecay:

    def test_behavioral_pattern_in_category_decay(self):
        """behavioral_pattern should be a registered decay category."""
        from services.user_trait_service import CATEGORY_DECAY
        assert 'behavioral_pattern' in CATEGORY_DECAY

    def test_behavioral_pattern_has_slow_decay(self):
        """Activity time patterns are stable — base_decay should be low."""
        from services.user_trait_service import CATEGORY_DECAY
        assert CATEGORY_DECAY['behavioral_pattern']['base_decay'] <= 0.01

    def test_behavioral_pattern_has_floor(self):
        """Behavioral patterns should have a floor to prevent total erasure."""
        from services.user_trait_service import CATEGORY_DECAY
        assert CATEGORY_DECAY['behavioral_pattern']['floor'] > 0
