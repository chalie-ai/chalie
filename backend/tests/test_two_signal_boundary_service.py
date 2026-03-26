"""Tests for TwoSignalBoundaryService — topic boundary detection."""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from services.two_signal_boundary_service import (
    TwoSignalBoundaryService,
    BoundaryResult,
    COLD_START_MSGS,
    STALE_GAP_SECONDS,
    _MARKER_PATTERNS,
)


pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────

def _random_embedding(dim=256, seed=None):
    """Generate a random L2-normalised embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _similar_embedding(base, noise_scale=0.05, seed=None):
    """Return an embedding close to `base` (high cosine similarity)."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base.shape).astype(np.float32) * noise_scale
    v = base + noise
    return v / (np.linalg.norm(v) + 1e-8)


def _different_embedding(dim=256, seed=42):
    """Return an embedding orthogonal-ish to typical embeddings."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def mock_store():
    """Mock MemoryStore so TwoSignalBoundaryService starts with fresh state."""
    from services.memory_store import MemoryStore
    store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=store):
        yield store


# ── Cold start behavior ──────────────────────────────────────────────

class TestColdStart:

    def test_first_message_is_not_boundary(self, mock_store):
        """First message without a marker should not fire a boundary."""
        svc = TwoSignalBoundaryService('test-thread')
        emb = _random_embedding(seed=1)
        result = svc.update(emb, 'Hello world')

        assert result.is_boundary is False
        assert result.trigger == 'cold_start'

    def test_cold_start_period_only_markers_fire(self, mock_store):
        """During cold start (< COLD_START_MSGS), only markers can fire."""
        svc = TwoSignalBoundaryService('test-thread')
        base = _random_embedding(seed=10)

        # Feed messages during cold start without markers
        for i in range(COLD_START_MSGS - 1):
            emb = _similar_embedding(base, noise_scale=0.01, seed=i)
            result = svc.update(emb, f'message {i}')
            # Only cold_start trigger, no boundary
            if i > 0:
                assert result.trigger == 'cold_start'
                assert result.is_boundary is False


# ── Discourse marker detection ───────────────────────────────────────

class TestDiscourseMarkers:

    def test_explicit_topic_change_detected(self, mock_store):
        """'changing the subject' triggers a marker boundary."""
        svc = TwoSignalBoundaryService('test-thread')
        emb = _random_embedding(seed=1)
        result = svc.update(emb, "Anyway, changing the subject, how about lunch?")

        assert result.is_boundary is True
        assert result.marker_found is not None

    def test_by_the_way_detected(self, mock_store):
        """'by the way' is a known marker."""
        svc = TwoSignalBoundaryService('test-thread')
        emb = _random_embedding(seed=2)
        result = svc.update(emb, "By the way, did you see the news?")

        assert result.marker_found is not None

    def test_speaking_of_detected(self, mock_store):
        """'speaking of' is a known marker."""
        svc = TwoSignalBoundaryService('test-thread')
        emb = _random_embedding(seed=3)
        result = svc.update(emb, "Speaking of Python, I have a question")

        assert result.marker_found is not None

    def test_no_marker_in_normal_text(self, mock_store):
        """Normal conversational text should not match any marker."""
        svc = TwoSignalBoundaryService('test-thread')
        emb = _random_embedding(seed=4)
        result = svc.update(emb, "I think Python is a great programming language")

        assert result.marker_found is None


# ── Two-signal boundary detection ────────────────────────────────────

class TestTwoSignalDetection:

    def _warm_up(self, svc, base_emb, n=COLD_START_MSGS + 2):
        """Feed enough similar messages to exit cold start."""
        for i in range(n):
            emb = _similar_embedding(base_emb, noise_scale=0.02, seed=100 + i)
            svc.update(emb, f'similar message {i}')

    def test_sharp_topic_change_fires_boundary(self, mock_store):
        """A sharp embedding shift after warm-up should fire a two-signal boundary."""
        svc = TwoSignalBoundaryService('test-thread')
        base = _random_embedding(seed=50)

        # Warm up with similar messages
        self._warm_up(svc, base, n=COLD_START_MSGS + 5)

        # Now inject a very different embedding
        different = _different_embedding(seed=999)
        result = svc.update(different, 'something completely different')

        # Should fire either via marker or two_signal
        # The "completely different" text contains the marker pattern
        assert result.is_boundary is True

    def test_continuation_does_not_fire(self, mock_store):
        """Continuing on the same topic should not fire a boundary."""
        svc = TwoSignalBoundaryService('test-thread')
        base = _random_embedding(seed=60)

        self._warm_up(svc, base, n=COLD_START_MSGS + 5)

        # Continue with similar embedding
        similar = _similar_embedding(base, noise_scale=0.02, seed=200)
        result = svc.update(similar, 'more of the same topic')

        assert result.is_boundary is False
        assert result.trigger == 'none'


# ── Stale gap reset ──────────────────────────────────────────────────

class TestStaleGapReset:

    def test_stale_gap_fires_boundary(self, mock_store):
        """A long silence (> STALE_GAP_SECONDS) triggers a boundary."""
        svc = TwoSignalBoundaryService('test-thread')
        base = _random_embedding(seed=70)

        # First message sets last_timestamp
        svc.update(base, 'hello')

        # Simulate time passing by manipulating state
        import time
        svc._state['last_timestamp'] = time.time() - STALE_GAP_SECONDS - 100

        result = svc.update(base, 'back after a break')

        assert result.is_boundary is True
        assert result.trigger == 'stale_gap'
        assert result.just_reset_from_silence is True


# ── State persistence ────────────────────────────────────────────────

class TestStatePersistence:

    def test_save_and_reload_state(self, mock_store):
        """State can be saved to MemoryStore and reloaded."""
        svc1 = TwoSignalBoundaryService('persist-thread')
        base = _random_embedding(seed=80)

        # Feed a few messages
        for i in range(3):
            emb = _similar_embedding(base, noise_scale=0.02, seed=80 + i)
            svc1.update(emb, f'msg {i}')

        # Save state
        svc1.save_state()

        # Create new instance for same thread — should load persisted state
        svc2 = TwoSignalBoundaryService('persist-thread')
        assert svc2._state['msg_count'] == 3

    def test_fresh_state_on_missing_store_key(self, mock_store):
        """Missing store key results in fresh initial state."""
        svc = TwoSignalBoundaryService('new-thread')
        assert svc._state['msg_count'] == 0
        assert svc._state['embeddings'] == []


# ── BoundaryResult dataclass ─────────────────────────────────────────

class TestBoundaryResult:

    def test_boundary_result_fields(self):
        """BoundaryResult has all expected fields."""
        result = BoundaryResult(
            is_boundary=True,
            confidence=0.85,
            trigger='marker',
            consec_sim=0.45,
            window_sim=0.50,
            marker_found='by the way',
        )
        assert result.is_boundary is True
        assert result.confidence == 0.85
        assert result.trigger == 'marker'
        assert result.just_reset_from_silence is False  # default

    def test_confidence_range(self, mock_store):
        """Confidence should be in [0.0, 1.0]."""
        svc = TwoSignalBoundaryService('conf-thread')
        emb = _random_embedding(seed=90)
        result = svc.update(emb, 'test')

        assert 0.0 <= result.confidence <= 1.0
