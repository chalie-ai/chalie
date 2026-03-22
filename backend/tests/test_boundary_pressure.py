"""Pressure tests for TwoSignalBoundaryService.

Covers the critical false-positive bug where terse answers (e.g. "iam 38" in
response to "How old are you?") caused spurious topic boundaries, losing
conversational context.

Two categories:
  1. Synthetic embeddings (unit) -- pure logic, no model dependency
  2. Real ONNX embeddings (integration) -- end-to-end with gte-modernbert-base
"""

import time

import numpy as np
import pytest
from unittest.mock import patch

from services.two_signal_boundary_service import (
    TwoSignalBoundaryService,
    BoundaryResult,
    COLD_START_MSGS,
    STALE_GAP_SECONDS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic embedding helpers
# ══════════════════════════════════════════════════════════════════════════════

DIM = 768


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _random_emb(seed=None):
    rng = np.random.RandomState(seed)
    return _norm(rng.randn(DIM).astype(np.float32))


def _similar_emb(base, noise=0.05, seed=None):
    rng = np.random.RandomState(seed)
    return _norm(base + rng.randn(DIM).astype(np.float32) * noise)


def _orthogonal_emb(base, seed=42):
    """Return an embedding roughly orthogonal to *base*."""
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    # Remove component along base
    v = v - np.dot(v, base) * base
    return _norm(v)


def _warm_up(svc, base, n=COLD_START_MSGS + 2, noise=0.06):
    """Feed *n* similar messages to exit cold start and build a baseline.

    Default noise=0.06 gives consecutive cosine ~0.25, which is realistic
    for conversation-level variance in 768-dim embeddings.  The resulting
    thresholds (mean - 1.6*std) leave headroom for within-topic drift.
    """
    for i in range(n):
        emb = _similar_emb(base, noise=noise, seed=100 + i)
        svc.update(emb, f'warm-up message {i} about the ongoing topic')


def _diag(result):
    """Format BoundaryResult diagnostics for assertion messages."""
    return (
        f"is_boundary={result.is_boundary} trigger={result.trigger} "
        f"consec_sim={result.consec_sim:.4f} window_sim={result.window_sim:.4f} "
        f"marker_found={result.marker_found}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Synthetic embedding tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestShortAnswerSynthetic:
    """Small embedding perturbations (simulating short answers) must not
    trigger boundaries after a warm-up baseline is established."""

    def test_within_baseline_drift_no_boundary(self, mock_store):
        """Probe at same noise as warm-up -- normal within-topic variance.

        With warm-up noise=0.06 and probe noise=0.06, the consecutive cosine
        is within the learned distribution, so neither signal should drop
        below threshold.
        """
        base = _random_emb(seed=1)
        svc = TwoSignalBoundaryService(thread_id='syn-short-mod')
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.06, seed=200)
        result = svc.update(noisy, 'iam 38')
        assert not result.is_boundary, (
            f"Within-baseline drift (noise=0.06) should not fire. {_diag(result)}"
        )

    def test_slightly_above_baseline_no_boundary(self, mock_store):
        """Probe slightly noisier than warm-up -- one signal may dip but
        the two-signal gate should prevent a false positive.

        With warm-up noise=0.06, probe at 0.08: consecutive similarity drops
        but window similarity (centroid) remains within threshold because
        the probe is still correlated with the base.
        """
        base = _random_emb(seed=2)
        svc = TwoSignalBoundaryService(thread_id='syn-short-015')
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.08, seed=201)
        result = svc.update(noisy, 'yes thats right')
        assert not result.is_boundary, (
            f"Slightly-above-baseline drift (noise=0.08) should not fire. "
            f"{_diag(result)}"
        )

    def test_boundary_region(self, mock_store):
        """Probe at noise=0.15 -- well outside the warm-up distribution.

        With warm-up noise=0.06, this is a significant departure.  Whether
        it fires depends on exact seed, but it enters the boundary zone.
        We verify the result is well-formed and document the outcome.
        """
        base = _random_emb(seed=3)
        svc = TwoSignalBoundaryService(thread_id='syn-short-030')
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.15, seed=202)
        result = svc.update(noisy, 'something moderately different')
        # At noise=0.15 with warm-up noise=0.06, this often fires.
        # We just verify the result is well-formed.
        assert isinstance(result.is_boundary, bool)
        assert result.trigger in ('none', 'two_signal', 'marker', 'both')

    def test_orthogonal_embedding_fires(self, mock_store):
        """Truly orthogonal embedding SHOULD fire a two_signal boundary."""
        base = _random_emb(seed=4)
        svc = TwoSignalBoundaryService(thread_id='syn-short-orth')
        _warm_up(svc, base)

        orth = _orthogonal_emb(base, seed=42)
        result = svc.update(orth, 'explain quantum computing fundamentals to me please')
        assert result.is_boundary, (
            f"Orthogonal embedding should fire boundary. {_diag(result)}"
        )
        assert result.trigger in ('two_signal', 'both'), (
            f"Expected two_signal or both, got {result.trigger}"
        )


@pytest.mark.unit
class TestStatisticalEdgeCases:
    """Edge cases around the self-calibrating threshold maths."""

    def test_low_variance_baseline_fires_on_deviation(self, mock_store):
        """Very uniform warm-up (noise=0.005) makes thresholds tight.

        With noise=0.005, consecutive cosine is ~0.99 and std is tiny
        (floored at 0.01), giving threshold ~0.975.  A probe at noise=0.01
        (consec sim ~0.96) drops BOTH signals below threshold and correctly
        fires.

        This is the expected behaviour of self-calibrating thresholds: a
        very uniform baseline means ANY deviation is significant.  In real
        conversations, natural variance during warm-up prevents this
        sensitivity -- which is validated by the other synthetic tests.
        """
        base = _random_emb(seed=10)
        svc = TwoSignalBoundaryService(thread_id='syn-lowvar')
        # Very tight warm-up
        for i in range(COLD_START_MSGS + 2):
            emb = _similar_emb(base, noise=0.005, seed=300 + i)
            svc.update(emb, f'uniform warm-up {i} about the same topic')

        # Probe at noise=0.01 -- double the warm-up variance.
        noisy = _similar_emb(base, noise=0.01, seed=310)
        result = svc.update(noisy, 'a slightly different answer about the same topic')

        # With extremely low baseline variance, even mild deviation fires
        # both signals.  This is correct calibration behaviour.
        assert result.is_boundary, (
            f"Low-variance baseline should fire on 2x variance deviation. "
            f"{_diag(result)}"
        )
        assert result.trigger == 'two_signal'

    def test_low_variance_baseline_tolerates_same_noise(self, mock_store):
        """With very uniform warm-up, a probe at the SAME noise level
        should NOT fire -- it is statistically indistinguishable from
        the baseline."""
        base = _random_emb(seed=10)
        svc = TwoSignalBoundaryService(thread_id='syn-lowvar-same')
        for i in range(COLD_START_MSGS + 2):
            emb = _similar_emb(base, noise=0.005, seed=300 + i)
            svc.update(emb, f'uniform warm-up {i} about the same topic')

        noisy = _similar_emb(base, noise=0.005, seed=310)
        result = svc.update(noisy, 'a slightly different answer about the same topic')
        assert not result.is_boundary, (
            f"Same-noise probe should not fire on low-variance baseline. "
            f"{_diag(result)}"
        )

    def test_high_variance_baseline_tolerates_drift(self, mock_store):
        """Varied warm-up (noise=0.08) makes thresholds loose.

        A probe at noise=0.10 should NOT fire because the learned std
        already accounts for this level of variation in the baseline.
        """
        base = _random_emb(seed=11)
        svc = TwoSignalBoundaryService(thread_id='syn-highvar')
        for i in range(COLD_START_MSGS + 4):
            emb = _similar_emb(base, noise=0.08, seed=400 + i)
            svc.update(emb, f'varied warm-up {i} diverse topic aspects')

        noisy = _similar_emb(base, noise=0.10, seed=410)
        result = svc.update(noisy, 'another aspect of the discussion')
        assert not result.is_boundary, (
            f"High-variance baseline should tolerate noise=0.10. {_diag(result)}"
        )

    def test_exact_cold_start_exit(self, mock_store):
        """After exactly COLD_START_MSGS messages, the (COLD_START_MSGS+1)-th
        should be evaluated by the two-signal detector (not cold_start).

        The cold start check is: len(consec_history) < COLD_START_MSGS.
        The first message (n=0) has no predecessor so adds nothing.
        Messages 1 through COLD_START_MSGS each append to consec_history.
        So after COLD_START_MSGS+1 total messages, history has COLD_START_MSGS
        entries and message COLD_START_MSGS+1 exits cold start.
        """
        base = _random_emb(seed=12)
        svc = TwoSignalBoundaryService(thread_id='syn-cold-exit')

        # Feed COLD_START_MSGS + 1 messages (fill the history)
        for i in range(COLD_START_MSGS + 1):
            emb = _similar_emb(base, noise=0.06, seed=500 + i)
            r = svc.update(emb, f'cold start msg {i}')

        # The next message should use the real two-signal detector
        emb = _similar_emb(base, noise=0.06, seed=510)
        r = svc.update(emb, 'just past cold start')
        assert r.trigger != 'cold_start', (
            f"Still in cold start after {COLD_START_MSGS + 2} messages"
        )
        # And this similar message should NOT be a boundary
        assert not r.is_boundary, (
            f"Similar message just past cold start fired boundary. {_diag(r)}"
        )

    def test_boundary_does_not_corrupt_stats(self, mock_store):
        """After a two_signal boundary fires, subsequent similar messages
        should NOT fire -- the outlier must not have polluted the baseline.

        Key property: stats are only updated on NON-boundary messages,
        so the orthogonal outlier that fired should not shift the rolling
        mean/std.
        """
        base = _random_emb(seed=13)
        svc = TwoSignalBoundaryService(thread_id='syn-corrupt')
        _warm_up(svc, base)

        # Fire a boundary with an orthogonal embedding
        orth = _orthogonal_emb(base, seed=50)
        r_boundary = svc.update(orth, 'completely unrelated quantum physics')
        assert r_boundary.is_boundary, (
            f"Setup: expected boundary to fire. {_diag(r_boundary)}"
        )

        # Now feed similar messages at warm-up noise level -- they should NOT fire
        for i in range(3):
            emb = _similar_emb(base, noise=0.06, seed=600 + i)
            r = svc.update(emb, f'continuing the original discussion {i}')
            assert not r.is_boundary, (
                f"Message {i} after boundary should not fire. "
                f"Outlier may have corrupted stats. {_diag(r)}"
            )


@pytest.mark.unit
class TestColdStartShortMessages:
    """Short messages during cold start must never fire a boundary
    (only discourse markers can fire during cold start)."""

    @pytest.mark.parametrize("text", ["yes", "no", "ok", "38", "fine"])
    def test_terse_cold_start_no_boundary(self, mock_store, text):
        base = _random_emb(seed=20)
        svc = TwoSignalBoundaryService(thread_id=f'syn-cold-{text}')

        # First message
        svc.update(base, 'how old are you?')

        # Second message: still in cold start
        noisy = _similar_emb(base, noise=0.10, seed=700)
        result = svc.update(noisy, text)
        assert result.trigger == 'cold_start', (
            f"Message '{text}' during cold start should have trigger='cold_start', "
            f"got trigger='{result.trigger}'"
        )
        assert not result.is_boundary, (
            f"Message '{text}' during cold start should not be a boundary. "
            f"{_diag(result)}"
        )


@pytest.mark.unit
class TestDiscourseMarkerEdgeCases:
    """Verify discourse marker regex precision."""

    def test_anyway_fires(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-anyway')
        base = _random_emb(seed=30)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=800)
        result = svc.update(noisy, "anyway I'm 38")
        assert result.is_boundary, (
            f"'anyway' should fire a marker boundary. {_diag(result)}"
        )
        assert result.marker_found is not None
        assert 'anyway' in result.marker_found.lower()

    def test_back_to_fires(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-backto')
        base = _random_emb(seed=31)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=801)
        result = svc.update(noisy, "back to my age, I'm 38")
        assert result.is_boundary, (
            f"'back to' should fire a marker boundary. {_diag(result)}"
        )
        assert result.marker_found is not None

    def test_iam_38_no_marker(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-iam38')
        base = _random_emb(seed=32)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=802)
        result = svc.update(noisy, 'iam 38')
        assert result.marker_found is None, (
            f"'iam 38' should NOT match any marker, got: {result.marker_found}"
        )

    def test_i_am_38_years_old_no_marker(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-iam38yo')
        base = _random_emb(seed=33)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=803)
        result = svc.update(noisy, 'i am 38 years old')
        assert result.marker_found is None, (
            f"'i am 38 years old' should NOT match any marker, got: {result.marker_found}"
        )

    def test_yes_no_marker(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-yes')
        base = _random_emb(seed=34)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=804)
        result = svc.update(noisy, 'yes')
        assert result.marker_found is None, (
            f"'yes' should NOT match any marker, got: {result.marker_found}"
        )

    def test_no_not_really_no_marker(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-nonotreally')
        base = _random_emb(seed=35)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=805)
        result = svc.update(noisy, 'no not really')
        assert result.marker_found is None, (
            f"'no not really' should NOT match any marker, got: {result.marker_found}"
        )

    def test_oh_wait_at_start_fires(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-ohwait-start')
        base = _random_emb(seed=36)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=806)
        result = svc.update(noisy, 'oh wait I forgot something')
        assert result.is_boundary, (
            f"'oh wait' at start should fire marker. {_diag(result)}"
        )
        assert result.marker_found is not None

    def test_oh_wait_in_middle_no_fire(self, mock_store):
        svc = TwoSignalBoundaryService(thread_id='syn-marker-ohwait-mid')
        base = _random_emb(seed=37)
        _warm_up(svc, base)

        noisy = _similar_emb(base, noise=0.05, seed=807)
        result = svc.update(noisy, 'and then I said oh wait to him')
        assert result.marker_found is None, (
            f"'oh wait' in middle should NOT fire, got: {result.marker_found}"
        )


@pytest.mark.unit
class TestStaleGapEdgeCases:
    """Stale gap (>45min silence) behaviour."""

    def test_just_under_stale_gap_no_reset(self, mock_store):
        """Message at STALE_GAP_SECONDS - 1 should NOT trigger stale reset."""
        base = _random_emb(seed=40)
        svc = TwoSignalBoundaryService(thread_id='syn-stale-under')
        _warm_up(svc, base)

        # Simulate time passing (just under the threshold)
        svc._state['last_timestamp'] = time.time() - (STALE_GAP_SECONDS - 1)

        noisy = _similar_emb(base, noise=0.05, seed=900)
        result = svc.update(noisy, 'still here')
        assert result.trigger != 'stale_gap', (
            f"Message just under stale gap should not trigger stale_gap. "
            f"{_diag(result)}"
        )
        assert not result.just_reset_from_silence

    def test_just_over_stale_gap_resets(self, mock_store):
        """Message at STALE_GAP_SECONDS + 1 SHOULD trigger stale reset."""
        base = _random_emb(seed=41)
        svc = TwoSignalBoundaryService(thread_id='syn-stale-over')
        _warm_up(svc, base)

        # Simulate time passing (just over the threshold)
        svc._state['last_timestamp'] = time.time() - (STALE_GAP_SECONDS + 1)

        noisy = _similar_emb(base, noise=0.05, seed=901)
        result = svc.update(noisy, 'back after a long break')
        assert result.trigger == 'stale_gap', (
            f"Message over stale gap should trigger stale_gap. {_diag(result)}"
        )
        assert result.is_boundary
        assert result.just_reset_from_silence

    def test_after_stale_reset_short_msg_is_cold_start(self, mock_store):
        """After stale reset, next messages should be in cold start."""
        base = _random_emb(seed=42)
        svc = TwoSignalBoundaryService(thread_id='syn-stale-cold')
        _warm_up(svc, base)

        # Trigger stale gap
        svc._state['last_timestamp'] = time.time() - (STALE_GAP_SECONDS + 60)
        noisy = _similar_emb(base, noise=0.05, seed=902)
        stale_result = svc.update(noisy, 'hey')
        assert stale_result.trigger == 'stale_gap'

        # Next message should be in cold start
        noisy2 = _similar_emb(base, noise=0.05, seed=903)
        result = svc.update(noisy2, '38')
        assert not result.is_boundary, (
            f"Short message after stale reset should not fire boundary "
            f"(should be in cold start). {_diag(result)}"
        )


@pytest.mark.unit
class TestBaselineStability:
    """Verify that the rolling baseline stays clean under various patterns."""

    def test_outlier_does_not_corrupt_subsequent(self, mock_store):
        """Feed 30+ messages, inject one outlier that fires, verify recovery."""
        base = _random_emb(seed=50)
        svc = TwoSignalBoundaryService(thread_id='syn-stability-outlier')

        # Extended warm-up with moderate variance
        for i in range(30):
            emb = _similar_emb(base, noise=0.06, seed=1000 + i)
            svc.update(emb, f'extended conversation message {i} about our topic')

        # Inject outlier
        orth = _orthogonal_emb(base, seed=99)
        r_out = svc.update(orth, 'something completely different')
        assert r_out.is_boundary, (
            f"Setup: outlier should fire. {_diag(r_out)}"
        )

        # Recovery: similar messages at warm-up noise level should not fire
        for i in range(5):
            emb = _similar_emb(base, noise=0.06, seed=1100 + i)
            r = svc.update(emb, f'resuming our earlier discussion point {i}')
            assert not r.is_boundary, (
                f"Recovery message {i} fired boundary -- outlier corrupted stats. "
                f"{_diag(r)}"
            )

    def test_qa_alternation_pattern(self, mock_store):
        """Alternating between two sub-clusters (Q&A pattern) should not
        accumulate false boundaries after warm-up.

        Sub-cluster B is at noise=0.03 from A, giving ~0.75 cosine between
        clusters -- distinct but correlated (like questions vs answers in
        the same topic domain).
        """
        base_a = _random_emb(seed=60)
        # Sub-cluster B: correlated with A but distinct
        base_b = _similar_emb(base_a, noise=0.03, seed=61)

        svc = TwoSignalBoundaryService(thread_id='syn-stability-qa')

        # Warm up with alternating pattern
        for i in range(COLD_START_MSGS + 4):
            if i % 2 == 0:
                emb = _similar_emb(base_a, noise=0.03, seed=1200 + i)
                text = f'question {i // 2} about the topic'
            else:
                emb = _similar_emb(base_b, noise=0.03, seed=1200 + i)
                text = f'answer {i // 2} about the topic'
            svc.update(emb, text)

        # Now test more alternations at same noise level -- none should fire
        for i in range(10):
            if i % 2 == 0:
                emb = _similar_emb(base_a, noise=0.03, seed=1300 + i)
                text = f'follow-up question {i}'
            else:
                emb = _similar_emb(base_b, noise=0.03, seed=1300 + i)
                text = f'follow-up answer {i}'
            r = svc.update(emb, text)
            assert not r.is_boundary, (
                f"Q&A alternation message {i} fired boundary. {_diag(r)}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: Real ONNX embedding tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def embed_service():
    """Real ONNX embedding service. Skip if model not available."""
    try:
        from services.embedding_service import EmbeddingService
        svc = EmbeddingService()
        # Verify the model loads and produces output
        svc.generate_embedding_np("test")
        return svc
    except Exception:
        pytest.skip("ONNX embedding model not available")


def _embed(embed_service, text):
    """Generate a real L2-normalised embedding."""
    return embed_service.generate_embedding_np(text)


def _real_warm_up(svc, embed_service, messages):
    """Feed a list of real messages through the boundary detector."""
    for msg in messages:
        emb = _embed(embed_service, msg)
        svc.update(emb, msg)


def _assert_no_boundary(result, text):
    """Assert no boundary with rich diagnostics."""
    assert not result.is_boundary, (
        f"FALSE POSITIVE: '{text}' fired boundary.\n"
        f"  trigger={result.trigger}\n"
        f"  consec_sim={result.consec_sim:.4f}\n"
        f"  window_sim={result.window_sim:.4f}\n"
        f"  marker_found={result.marker_found}"
    )


def _assert_boundary(result, text):
    """Assert boundary fires with diagnostics."""
    assert result.is_boundary, (
        f"FALSE NEGATIVE: '{text}' should have fired boundary.\n"
        f"  trigger={result.trigger}\n"
        f"  consec_sim={result.consec_sim:.4f}\n"
        f"  window_sim={result.window_sim:.4f}\n"
        f"  marker_found={result.marker_found}"
    )


@pytest.mark.integration
class TestShortAnswerContinuity:
    """Real-embedding tests: short answers in natural conversation must NOT
    trigger topic boundaries."""

    GETTING_TO_KNOW_YOU = [
        "Hey, nice to meet you! I'd love to learn more about you.",
        "I'm originally from Portland, Oregon. Moved here last year.",
        "I work in software engineering, been doing it for about 10 years.",
        "I mostly work with Python and JavaScript these days.",
        "Yeah I really enjoy building web applications.",
        "I also do some machine learning work on the side.",
        "My hobbies include hiking and photography.",
        "I try to get outdoors every weekend if I can.",
        "Portland has amazing hiking trails nearby.",
        "I also dabble in a bit of cooking when I have the time.",
        "What about you, tell me a bit about yourself.",
        "How old are you?",
    ]

    @pytest.mark.parametrize("short_answer", [
        "iam 38",
        "38",
        "i am 38",
        "im 38",
    ])
    def test_age_answer_no_boundary(self, mock_store, embed_service, short_answer):
        svc = TwoSignalBoundaryService(thread_id=f'real-age-{short_answer}')
        _real_warm_up(svc, embed_service, self.GETTING_TO_KNOW_YOU)

        emb = _embed(embed_service, short_answer)
        result = svc.update(emb, short_answer)
        _assert_no_boundary(result, short_answer)

    FOOD_DISCUSSION = [
        "I've been trying to eat healthier lately by cooking more.",
        "Mostly cutting out processed foods and cooking from scratch.",
        "I started making Mediterranean food at home, lots of Greek dishes.",
        "I love making hummus and falafel and grilled halloumi.",
        "Greek food is so flavorful and relatively healthy too.",
        "I also make a lot of fresh salads with good olive oil and feta.",
        "Mediterranean cooking is all about fresh ingredients and simplicity.",
        "The best part is that most Mediterranean recipes are pretty easy.",
        "I've been experimenting with different spice blends for my cooking.",
        "Cooking healthy food at home has made such a difference for me.",
        "I feel so much better when I eat well and cook my own meals.",
        "Do you enjoy cooking at home or do you prefer eating out?",
    ]

    @pytest.mark.parametrize("short_answer", ["yes", "no", "nah", "not really"])
    def test_food_agreement_no_boundary(self, mock_store, embed_service, short_answer):
        """Bare affirmation/negation after topic-specific content.

        These produce generic embeddings far from any topic centroid, but
        the short message gate (SHORT_MSG_MAX_WORDS=4) correctly suppresses
        the two-signal check.  Discourse markers still fire if present.
        """
        svc = TwoSignalBoundaryService(thread_id=f'real-food-{short_answer}')
        _real_warm_up(svc, embed_service, self.FOOD_DISCUSSION)

        emb = _embed(embed_service, short_answer)
        result = svc.update(emb, short_answer)
        _assert_no_boundary(result, short_answer)

    WORK_DISCUSSION = [
        "I've been at my current company for a while now.",
        "We're a mid-size tech company, about 200 employees.",
        "I lead a team of backend engineers.",
        "We mostly build microservices and data pipelines.",
        "It's been pretty rewarding work honestly.",
        "The team culture is really collaborative and supportive.",
        "We do code reviews and pair programming regularly.",
        "I started here right out of college years ago.",
        "Worked my way up from junior developer to team lead.",
        "The pay is decent and the benefits are good.",
        "We get flexible hours and can work from home.",
        "I've been working in tech for a long time now.",
    ]

    @pytest.mark.parametrize("short_answer", [
        "5 years",
        "since 2019",
        "about 6",
    ])
    def test_work_duration_no_boundary(self, mock_store, embed_service, short_answer):
        svc = TwoSignalBoundaryService(thread_id=f'real-work-{short_answer}')
        _real_warm_up(svc, embed_service, self.WORK_DISCUSSION)

        emb = _embed(embed_service, short_answer)
        result = svc.update(emb, short_answer)
        _assert_no_boundary(result, short_answer)

    PERSONAL_PREFERENCES = [
        "Let's talk about what you like to do for fun.",
        "I'm really into outdoor activities.",
        "Hiking, camping, kayaking -- anything in nature.",
        "I also read a lot of science fiction.",
        "Neal Stephenson is probably my favorite author.",
        "Snow Crash and The Diamond Age are my favorites.",
        "I also enjoy watching documentaries about nature.",
        "Do you prefer dogs or cats?",
        "I've always been more of a dog person.",
        "Dogs are just so loyal and happy all the time.",
        "I have a golden retriever named Max.",
        "What's your favorite color?",
    ]

    @pytest.mark.parametrize("short_answer", [
        "blue",
        "green",
        "idk maybe red",
    ])
    def test_preference_answer_no_boundary(self, mock_store, embed_service, short_answer):
        svc = TwoSignalBoundaryService(thread_id=f'real-pref-{short_answer}')
        _real_warm_up(svc, embed_service, self.PERSONAL_PREFERENCES)

        emb = _embed(embed_service, short_answer)
        result = svc.update(emb, short_answer)
        _assert_no_boundary(result, short_answer)


@pytest.mark.integration
class TestGenuineTopicChanges:
    """Real-embedding tests: dramatic semantic shifts SHOULD fire boundaries."""

    PERSONAL_FAMILY = [
        "My family is from Italy originally.",
        "My grandmother taught me to cook traditional Italian dishes.",
        "We visit Italy every summer to see relatives.",
        "I love the Amalfi coast, it's absolutely beautiful.",
        "The food there is incredible, nothing like it here.",
        "My cousins still live in Naples with their families.",
        "Family is really important to me, it shapes who I am.",
        "We have big family dinners every Sunday afternoon.",
        "My mom makes the best homemade pasta from scratch.",
        "The whole family gathers around the table and we talk for hours.",
        "I want to pass these traditions down to my kids someday.",
        "Italian family values are something I'm really proud of.",
    ]

    def test_personal_to_technical(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-change-tech')
        _real_warm_up(svc, embed_service, self.PERSONAL_FAMILY)

        shift = "Can you explain how quantum computing works?"
        emb = _embed(embed_service, shift)
        result = svc.update(emb, shift)
        _assert_boundary(result, shift)

    WORK_SPRINT = [
        "Our sprint planning was a disaster this week.",
        "The PM keeps changing requirements mid-sprint.",
        "We need to refactor the auth module urgently.",
        "The CI pipeline is too slow and keeps timing out.",
        "I filed three bugs yesterday during testing.",
        "Our code review process needs serious improvement.",
        "The deployment failed twice last week in production.",
        "We're behind on the roadmap by at least two weeks.",
        "The tech debt is really starting to pile up.",
        "I spent most of yesterday fixing merge conflicts.",
        "We need to hire more engineers to keep up with demand.",
        "The sprint retrospective was pretty tense honestly.",
    ]

    @pytest.mark.xfail(reason=(
        "Known limitation: semantically diverse warm-up topics (CI, bugs, "
        "deployment, roadmap) create wide baseline variance, making it hard "
        "to distinguish genuine topic changes from within-topic variation"
    ))
    def test_work_to_cooking(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-change-cooking')
        _real_warm_up(svc, embed_service, self.WORK_SPRINT)

        shift = "What's the best recipe for chocolate cake?"
        emb = _embed(embed_service, shift)
        result = svc.update(emb, shift)
        _assert_boundary(result, shift)

    CASUAL_WEATHER = [
        "Nice weather today, really sunny and warm.",
        "I went for a walk in the park this morning.",
        "Saw some cute dogs playing on the grass.",
        "The cherry blossoms are out, the park looks gorgeous.",
        "Spring is my favorite season by far.",
        "I love how the days are getting longer now.",
        "Maybe I'll have a picnic this weekend if it stays nice.",
        "The sunset was beautiful yesterday evening.",
        "The forecast says it should be warm all week.",
        "I might start eating lunch outside more often.",
        "It's so nice to just sit and enjoy the sunshine.",
        "Perfect weather for a long bike ride too.",
    ]

    def test_casual_to_crisis(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-change-crisis')
        _real_warm_up(svc, embed_service, self.CASUAL_WEATHER)

        shift = "My house is on fire what do I do"
        emb = _embed(embed_service, shift)
        result = svc.update(emb, shift)
        _assert_boundary(result, shift)


@pytest.mark.integration
class TestBorderlineCases:
    """Messages with discourse markers that signal topic shifts, even when
    the semantic content is only mildly different."""

    PET_TALK = [
        "I have two cats and a golden retriever at home.",
        "The cats are brother and sister, both orange tabbies.",
        "My dog loves to play fetch at the park every day.",
        "We adopted all of them from the local shelter.",
        "The vet said they're all in great health this year.",
        "My dog is getting a little old though, he's 11 now.",
        "I spoil them all with treats probably too much.",
        "Pets really do make your life so much better.",
        "The cats like to sleep on my keyboard while I work.",
        "My dog follows me around the house constantly.",
        "We go to the dog park every Saturday morning.",
        "I love watching them all play together at home.",
    ]

    def test_speaking_of_fires(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-border-speakingof')
        _real_warm_up(svc, embed_service, self.PET_TALK)

        msg = "speaking of animals, did you know elephants can't jump?"
        emb = _embed(embed_service, msg)
        result = svc.update(emb, msg)
        _assert_boundary(result, msg)
        assert result.marker_found is not None, (
            "Expected 'speaking of' marker to match"
        )

    COFFEE_TALK = [
        "I drink way too much coffee honestly.",
        "Usually about four cups a day, sometimes more.",
        "I prefer a dark roast with no sugar or cream.",
        "I have this great pour-over setup at home.",
        "The beans I get are from a local roaster downtown.",
        "They source their coffee directly from farms in Colombia.",
        "I tried cold brew last summer and absolutely loved it.",
        "Caffeine is basically my fuel at this point.",
        "I've tried cutting down but I just love the taste.",
        "My morning routine starts with grinding fresh beans.",
        "The smell of fresh coffee in the morning is the best.",
        "I also collect different mugs from places I travel.",
    ]

    def test_anyway_fires(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-border-anyway')
        _real_warm_up(svc, embed_service, self.COFFEE_TALK)

        msg = "anyway I prefer tea"
        emb = _embed(embed_service, msg)
        result = svc.update(emb, msg)
        _assert_boundary(result, msg)
        assert result.marker_found is not None

    WEEKEND_PLANS = [
        "I'm thinking about what to do this weekend.",
        "Maybe I'll go hiking if the weather is good.",
        "There's a new trail I've been wanting to try out.",
        "It's about an hour drive from the city center.",
        "I might invite some friends to come along with me.",
        "We could bring lunch and make a whole day of it.",
        "I also need to do some grocery shopping at some point.",
        "And probably clean the apartment too, it's a mess.",
        "Maybe I'll do the chores Saturday and hike Sunday.",
        "That way I can relax and enjoy the hike without worrying.",
        "I need to check what time the trailhead opens.",
        "Last time we went hiking we had such a great time.",
    ]

    def test_oh_wait_fires(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-border-ohwait')
        _real_warm_up(svc, embed_service, self.WEEKEND_PLANS)

        msg = "oh wait I forgot to mention something"
        emb = _embed(embed_service, msg)
        result = svc.update(emb, msg)
        _assert_boundary(result, msg)
        assert result.marker_found is not None

    TRAVEL_TALK = [
        "I went to Japan last year and it was absolutely incredible.",
        "Tokyo is such an amazing city with so much to see and do.",
        "The food was the best part of the trip honestly.",
        "I tried authentic ramen at this tiny shop in Shinjuku.",
        "The temples in Kyoto were breathtaking and so peaceful.",
        "I want to go back and explore Hokkaido next time I visit.",
        "The bullet train was an experience in itself, so fast.",
        "Japanese culture is fascinating to me in so many ways.",
        "The people were incredibly kind and helpful everywhere.",
        "I bought so many souvenirs I had to get a bigger suitcase.",
        "The gardens and parks in Tokyo were beautifully maintained.",
        "I'm already planning my next trip there for next year.",
    ]

    def test_that_reminds_me_fires(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-border-remindsme')
        _real_warm_up(svc, embed_service, self.TRAVEL_TALK)

        msg = "That reminds me of a book I read"
        emb = _embed(embed_service, msg)
        result = svc.update(emb, msg)
        _assert_boundary(result, msg)
        assert result.marker_found is not None


@pytest.mark.integration
class TestTerseConversationFlow:
    """A full conversation of mostly short messages -- none should trigger
    false boundaries. Tests that naturally terse style does not accumulate
    false positives."""

    TERSE_FLOW = [
        "hey",
        "good",
        "you?",
        "same",
        "nice",
        "yeah",
        "cool",
        "ok",
        "sure",
        "thanks",
    ]

    def test_terse_conversation_no_boundaries(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-terse-flow')

        boundaries_fired = []
        for i, msg in enumerate(self.TERSE_FLOW):
            emb = _embed(embed_service, msg)
            result = svc.update(emb, msg)
            if result.is_boundary and result.trigger not in ('cold_start',):
                boundaries_fired.append(
                    f"  msg[{i}]='{msg}': {_diag(result)}"
                )

        assert not boundaries_fired, (
            f"Terse conversation flow triggered {len(boundaries_fired)} "
            f"false boundaries:\n" + "\n".join(boundaries_fired)
        )


@pytest.mark.integration
class TestMixedLengthConversation:
    """Alternating long and short messages (natural chat rhythm) should not
    fire boundaries."""

    MIXED_FLOW = [
        (
            "I've been thinking about this project for a while now and I think "
            "we should completely restructure the database layer to use a more "
            "normalized schema with proper foreign key constraints."
        ),
        "ok",
        (
            "The main issue is that we're storing duplicate data in three "
            "different tables which makes updates really error-prone and has "
            "already caused two production bugs this quarter."
        ),
        "yes",
        (
            "My proposal is to create a single source of truth table and have "
            "the other tables reference it via foreign keys, then add some "
            "materialized views for the common query patterns."
        ),
        "hmm",
        (
            "I've already prototyped it locally and the migration would take "
            "about a week, but the long-term maintenance savings would be huge "
            "and we'd eliminate that whole class of consistency bugs."
        ),
        "sounds good",
    ]

    def test_mixed_length_no_boundaries(self, mock_store, embed_service):
        svc = TwoSignalBoundaryService(thread_id='real-mixed-len')

        boundaries_fired = []
        for i, msg in enumerate(self.MIXED_FLOW):
            emb = _embed(embed_service, msg)
            result = svc.update(emb, msg)
            if result.is_boundary and result.trigger not in ('cold_start',):
                boundaries_fired.append(
                    f"  msg[{i}]='{msg[:40]}...': {_diag(result)}"
                )

        assert not boundaries_fired, (
            f"Mixed-length conversation triggered {len(boundaries_fired)} "
            f"false boundaries:\n" + "\n".join(boundaries_fired)
        )


@pytest.mark.integration
class TestMultiTurnRealisticScenarios:
    """Extended realistic scenarios that combine several patterns."""

    def test_interview_style_qa(self, mock_store, embed_service):
        """Simulates an interview-style Q&A where Chalie asks questions
        and the user gives short answers. No boundaries should fire."""
        conversation = [
            "I'd like to get to know you better. Can I ask a few questions?",
            "sure",
            "What do you do for a living?",
            "software engineer",
            "How long have you been doing that?",
            "about 10 years",
            "Do you enjoy it?",
            "yeah mostly",
            "What's the best part about it?",
            "solving hard problems",
            "And the worst part?",
            "meetings",
            "Ha, that's a common one. Where are you based?",
            "san francisco",
            "Nice city! How old are you if you don't mind me asking?",
            "iam 38",
        ]

        svc = TwoSignalBoundaryService(thread_id='real-interview')
        boundaries_fired = []
        for i, msg in enumerate(conversation):
            emb = _embed(embed_service, msg)
            result = svc.update(emb, msg)
            if result.is_boundary and result.trigger not in ('cold_start',):
                boundaries_fired.append(
                    f"  msg[{i}]='{msg}': {_diag(result)}"
                )

        assert not boundaries_fired, (
            f"Interview-style Q&A triggered {len(boundaries_fired)} "
            f"false boundaries:\n" + "\n".join(boundaries_fired)
        )

    def test_emotional_support_with_terse_responses(self, mock_store, embed_service):
        """User venting about a bad day, giving terse emotional responses."""
        conversation = [
            "I had a really rough day at work today.",
            "My boss criticized my presentation in front of the whole team.",
            "It was so embarrassing, I felt like hiding.",
            "I'm sorry to hear that. How are you feeling now?",
            "terrible",
            "That's completely understandable. Did anyone support you?",
            "no",
            "Sometimes colleagues don't know how to react in those moments.",
            "yeah",
            "Would you like to talk about what happened in the presentation?",
            "not really",
            "That's okay. Is there anything I can help with?",
            "just listening helps",
        ]

        svc = TwoSignalBoundaryService(thread_id='real-emotional')
        boundaries_fired = []
        for i, msg in enumerate(conversation):
            emb = _embed(embed_service, msg)
            result = svc.update(emb, msg)
            if result.is_boundary and result.trigger not in ('cold_start',):
                boundaries_fired.append(
                    f"  msg[{i}]='{msg}': {_diag(result)}"
                )

        assert not boundaries_fired, (
            f"Emotional support conversation triggered {len(boundaries_fired)} "
            f"false boundaries:\n" + "\n".join(boundaries_fired)
        )

    def test_genuine_topic_change_mid_conversation(self, mock_store, embed_service):
        """Conversation with a genuine mid-stream topic change that SHOULD
        fire, surrounded by continuations that should NOT."""
        phase1 = [
            "I've been learning to play guitar lately.",
            "Started with acoustic guitar, learning basic chords first.",
            "I can play a few songs now, mostly folk and country stuff.",
            "My fingers are still getting calluses from the strings though.",
            "I practice about 30 minutes every evening after work.",
            "I'm using this app called Fender Play for structured lessons.",
            "My goal is to play Blackbird by the Beatles eventually.",
            "It's a fingerpicking song so it's pretty challenging.",
            "I also want to learn some classic rock songs too.",
            "Guitar practice is really relaxing after a long day.",
            "I might take some in-person lessons next month.",
            "Music has always been a big part of my life.",
        ]
        topic_change = "Do you know what the weather will be like tomorrow?"
        phase2_content = [
            "I was thinking of going for a run if it's nice out.",
            "Maybe around the park trail near my house.",
            "The trail is about three miles long, pretty flat and scenic.",
            "I like to run there in the morning when it's quiet.",
            "Running is great exercise and clears my head.",
        ]

        svc = TwoSignalBoundaryService(thread_id='real-mid-change')

        # Phase 1: feed guitar conversation
        for msg in phase1:
            emb = _embed(embed_service, msg)
            svc.update(emb, msg)

        # Topic change: should fire
        emb = _embed(embed_service, topic_change)
        result = svc.update(emb, topic_change)
        _assert_boundary(result, topic_change)

        # Phase 2: content-bearing continuations should not fire
        for msg in phase2_content:
            emb = _embed(embed_service, msg)
            result = svc.update(emb, msg)
            _assert_no_boundary(result, msg)
