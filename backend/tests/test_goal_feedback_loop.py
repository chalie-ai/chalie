"""Tests for H1.3 — Goal feedback loop wiring."""

import pytest


@pytest.mark.unit
class TestClassifyEngagement:
    """Test _classify_engagement() deterministic classifier."""

    def test_empty_text_is_ignored(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement('') == 'ignored'
        assert _classify_engagement('  ') == 'ignored'
        assert _classify_engagement('hi') == 'ignored'  # < 3 chars

    def test_rejection_patterns(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("stop doing that") == 'rejected'
        assert _classify_engagement("no thanks") == 'rejected'
        assert _classify_engagement("not interested") == 'rejected'
        assert _classify_engagement("leave me alone") == 'rejected'
        assert _classify_engagement("that's annoying") == 'rejected'

    def test_acknowledgment_patterns(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("ok") == 'acknowledged'
        assert _classify_engagement("thanks") == 'acknowledged'
        assert _classify_engagement("sure") == 'acknowledged'
        assert _classify_engagement("cool") == 'acknowledged'
        assert _classify_engagement("got it") == 'acknowledged'
        assert _classify_engagement("yep") == 'acknowledged'
        assert _classify_engagement("noted.") == 'acknowledged'

    def test_engaged_long_response(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("That's really interesting, tell me more about it") == 'engaged'

    def test_engaged_question(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("How does that work?") == 'engaged'

    def test_engaged_action_words(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("yes please") == 'engaged'
        assert _classify_engagement("show me that") == 'engaged'
        assert _classify_engagement("go ahead") == 'engaged'
        assert _classify_engagement("tell me more") == 'engaged'

    def test_short_unknown_is_acknowledged(self):
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("hmm") == 'acknowledged'

    def test_interesting_is_engaged(self):
        """'interesting' shows active engagement, not mere acknowledgment."""
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("interesting") == 'engaged'

    def test_ambiguous_rejection_with_engagement_is_engaged(self):
        """'I don't think that's right but tell me more' is engaged, not rejected."""
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("I don't think that's right but tell me more") == 'engaged'

    def test_short_rejection_still_rejected(self):
        """Short unambiguous rejections still work."""
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("stop") == 'rejected'
        assert _classify_engagement("don't do that") == 'rejected'

    def test_long_with_but_clause_is_engaged(self):
        """Long message with rejection + 'but' + engagement = engaged."""
        from workers.post_exchange_hooks import _classify_engagement
        assert _classify_engagement("I'm not interested in that approach however tell me about alternatives") == 'engaged'


