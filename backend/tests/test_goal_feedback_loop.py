"""Tests for H1.3 — Goal feedback loop wiring."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestClassifyEngagement:
    """Test _classify_engagement() deterministic classifier."""

    def test_empty_text_is_ignored(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement('') == 'ignored'
        assert _classify_engagement('  ') == 'ignored'
        assert _classify_engagement('hi') == 'ignored'  # < 3 chars

    def test_rejection_patterns(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("stop doing that") == 'rejected'
        assert _classify_engagement("no thanks") == 'rejected'
        assert _classify_engagement("not interested") == 'rejected'
        assert _classify_engagement("leave me alone") == 'rejected'
        assert _classify_engagement("that's annoying") == 'rejected'

    def test_acknowledgment_patterns(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("ok") == 'acknowledged'
        assert _classify_engagement("thanks") == 'acknowledged'
        assert _classify_engagement("sure") == 'acknowledged'
        assert _classify_engagement("cool") == 'acknowledged'
        assert _classify_engagement("got it") == 'acknowledged'
        assert _classify_engagement("yep") == 'acknowledged'
        assert _classify_engagement("noted.") == 'acknowledged'

    def test_engaged_long_response(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("That's really interesting, tell me more about it") == 'engaged'

    def test_engaged_question(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("How does that work?") == 'engaged'

    def test_engaged_action_words(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("yes please") == 'engaged'
        assert _classify_engagement("show me that") == 'engaged'
        assert _classify_engagement("go ahead") == 'engaged'
        assert _classify_engagement("tell me more") == 'engaged'

    def test_short_unknown_is_acknowledged(self):
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("hmm") == 'acknowledged'

    def test_interesting_is_engaged(self):
        """'interesting' shows active engagement, not mere acknowledgment."""
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("interesting") == 'engaged'

    def test_ambiguous_rejection_with_engagement_is_engaged(self):
        """'I don't think that's right but tell me more' is engaged, not rejected."""
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("I don't think that's right but tell me more") == 'engaged'

    def test_short_rejection_still_rejected(self):
        """Short unambiguous rejections still work."""
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("stop") == 'rejected'
        assert _classify_engagement("don't do that") == 'rejected'

    def test_long_with_but_clause_is_engaged(self):
        """Long message with rejection + 'but' + engagement = engaged."""
        from workers.digest_worker import _classify_engagement
        assert _classify_engagement("I'm not interested in that approach however tell me about alternatives") == 'engaged'


@pytest.mark.unit
class TestProactiveEngagementCorrelation:
    """Test _try_proactive_engagement_correlation() end-to-end."""

    def test_no_tag_is_noop(self, mock_store):
        """No proactive_response_tag means nothing happens."""
        from workers.digest_worker import _try_proactive_engagement_correlation
        _try_proactive_engagement_correlation("some response", "test-topic")
        # No errors, no calls

    def test_engaged_response_records_outcome(self, mock_store):
        """Engaged response calls record_outcome and clears tag."""
        mock_store.setex("proactive_response_tag:cooking", 14400, "goal-123")

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value

            from workers.digest_worker import _try_proactive_engagement_correlation
            _try_proactive_engagement_correlation(
                "That's really interesting, tell me more about the recipe",
                "cooking"
            )

            svc.record_outcome.assert_called_once_with("goal-123", "engaged")

        # Tag should be consumed
        assert mock_store.get("proactive_response_tag:cooking") is None

    def test_rejected_response_records_outcome(self, mock_store):
        """Rejected response calls record_outcome with 'rejected'."""
        mock_store.setex("proactive_response_tag:topic1", 14400, "goal-456")

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value

            from workers.digest_worker import _try_proactive_engagement_correlation
            _try_proactive_engagement_correlation("stop doing that", "topic1")

            svc.record_outcome.assert_called_once_with("goal-456", "rejected")

    def test_ignored_increments_counter(self, mock_store):
        """Ignored response increments goal:recent_ignored_count."""
        mock_store.setex("proactive_response_tag:topic2", 14400, "goal-789")

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            from workers.digest_worker import _try_proactive_engagement_correlation
            _try_proactive_engagement_correlation("", "topic2")

        count = mock_store.get("goal:recent_ignored_count")
        assert count is not None
        assert int(count) == 1

    def test_engaged_clears_ignored_counter(self, mock_store):
        """Engaged response clears the ignored counter."""
        mock_store.setex("proactive_response_tag:topic3", 14400, "goal-abc")
        mock_store.setex("goal:recent_ignored_count", 86400, "3")

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            from workers.digest_worker import _try_proactive_engagement_correlation
            _try_proactive_engagement_correlation(
                "yes please, tell me more about this",
                "topic3"
            )

        assert mock_store.get("goal:recent_ignored_count") is None

    def test_tag_consumed_only_once(self, mock_store):
        """Tag is consumed on first correlation — second call is noop."""
        mock_store.setex("proactive_response_tag:topic4", 14400, "goal-def")

        with patch('services.goal_ecology_service.GoalEcologyService') as MockEcology:
            svc = MockEcology.return_value

            from workers.digest_worker import _try_proactive_engagement_correlation
            _try_proactive_engagement_correlation("thanks", "topic4")
            _try_proactive_engagement_correlation("more thoughts", "topic4")

            # record_outcome called exactly once (first call consumed the tag)
            svc.record_outcome.assert_called_once()
