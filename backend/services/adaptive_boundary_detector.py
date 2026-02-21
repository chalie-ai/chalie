"""
Adaptive Topic Boundary Detector — 3-layer self-calibrating detector.

Brain analogue: Event Segmentation Theory (EST)
- Layer 1: NEWMA  — fast/slow EWMA divergence (anterior temporal early warning)
- Layer 2: Transient Surprise — z-score of similarity drop (prefrontal prediction error)
- Layer 3: Leaky Accumulator — hysteresis preventing single-message false positives

All thresholds are derived from running statistics of the conversation itself.
A noisy conversation raises the bar; a focused conversation lowers it. No tuning needed.
"""

import json
import logging
import time
from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np

from services.redis_client import RedisClientService

logger = logging.getLogger(__name__)

# Cold start: use conservative static threshold until enough stats accumulate
COLD_START_MSGS = 5
COLD_START_THRESHOLD = 0.55

# Stale baseline reset: dishabituation after long silence (brain analogue)
STALE_GAP_SECONDS = 2700  # 45 minutes


@dataclass
class BoundaryResult:
    is_boundary: bool
    accumulator: float
    boundary: float
    newma_signal: float
    surprise_signal: float
    confidence: float  # 0-1 accumulator fraction toward boundary
    just_reset_from_silence: bool = False  # True when stale-gap dishabituation fired


class AdaptiveBoundaryDetector:
    """
    Self-calibrating topic boundary detector.

    State is persisted in Redis at ``adaptive_boundary:{thread_id}`` with a 24h TTL.
    If Redis is unavailable the detector degrades gracefully to cold-start mode
    (conservative 0.55 threshold) for the current message.
    """

    # NEWMA window defaults (EWMAs on raw embeddings)
    DEFAULT_FAST_WINDOW = 4    # ~4-message recent center
    DEFAULT_SLOW_WINDOW = 18   # ~18-message long-term center

    # Leaky accumulator defaults
    DEFAULT_LEAK_RATE = 0.4         # 40% leaked per message
    DEFAULT_BOUNDARY_BASE = 2.5     # base firing threshold
    BOUNDARY_MIN = 1.5
    BOUNDARY_MAX = 5.0

    # Input blending weights for accumulator
    NEWMA_WEIGHT = 0.6
    SURPRISE_WEIGHT = 0.4

    # Redis TTL (match thread lifetime)
    _REDIS_TTL = 86400  # 24 hours

    def __init__(
        self,
        thread_id: str,
        regulator_params: Optional[dict] = None,
        focus_modifier: float = 0.0,
    ):
        self.thread_id = thread_id
        self._redis_key = f"adaptive_boundary:{thread_id}"

        params = regulator_params or {}
        self._fast_alpha = self._window_to_alpha(
            params.get('newma_window_fast', self.DEFAULT_FAST_WINDOW)
        )
        self._slow_alpha = self._window_to_alpha(
            params.get('newma_window_slow', self.DEFAULT_SLOW_WINDOW)
        )
        self._leak_rate = float(params.get('accumulator_leak_rate', self.DEFAULT_LEAK_RATE))
        self._boundary_base = float(
            params.get('accumulator_boundary_base', self.DEFAULT_BOUNDARY_BASE)
        )
        self._focus_modifier = float(focus_modifier)

        self._state = self._load_state()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def update(self, embedding: np.ndarray, best_similarity: float) -> BoundaryResult:
        """
        Process one message and determine whether a topic boundary occurred.

        Args:
            embedding: Embedding of the current message (already L2-normalised by caller).
            best_similarity: Cosine similarity to the best-matching existing topic.

        Returns:
            BoundaryResult with ``is_boundary`` and diagnostic fields.
        """
        s = self._state
        now = time.time()

        # Safety: re-normalise embedding (caller should already do this)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Stale baseline reset — dishabituation after long silence
        _just_reset = False
        if s['last_timestamp'] is not None:
            if now - s['last_timestamp'] > STALE_GAP_SECONDS:
                self._reset_emas(embedding, best_similarity)
                _just_reset = True
                logger.debug(
                    f"[BOUNDARY DETECTOR] Dishabituation reset for thread {self.thread_id}"
                )
        s['last_timestamp'] = now
        s['msg_count'] += 1

        # Cold start: gather stats but use conservative threshold
        if s['msg_count'] < COLD_START_MSGS:
            self._warm_emas(embedding, best_similarity)
            is_boundary = best_similarity < COLD_START_THRESHOLD
            return BoundaryResult(
                is_boundary=is_boundary,
                accumulator=0.0,
                boundary=self.BOUNDARY_MAX,
                newma_signal=0.0,
                surprise_signal=0.0,
                confidence=float(best_similarity) if is_boundary else 1.0 - float(best_similarity),
                just_reset_from_silence=_just_reset,
            )

        # ── Layer 1: NEWMA ────────────────────────────────────────
        fast = np.array(s['ewma_fast'])
        slow = np.array(s['ewma_slow'])
        fast = (1.0 - self._fast_alpha) * fast + self._fast_alpha * embedding
        slow = (1.0 - self._slow_alpha) * slow + self._slow_alpha * embedding
        s['ewma_fast'] = fast.tolist()
        s['ewma_slow'] = slow.tolist()

        drift = float(np.sum((fast - slow) ** 2))

        drift_diff = drift - s['drift_ema']
        s['drift_ema'] += self._slow_alpha * drift_diff
        s['drift_var_ema'] = (
            (1.0 - self._slow_alpha) * s['drift_var_ema']
            + self._slow_alpha * drift_diff ** 2
        )
        drift_std = max(sqrt(s['drift_var_ema']), 1e-6)
        newma_signal = (drift - s['drift_ema']) / drift_std

        # ── Layer 2: Transient Surprise ───────────────────────────
        drop = max(0.0, s['sim_ema'] - best_similarity)

        drop_diff = drop - s['drop_ema']
        s['drop_ema'] += self._slow_alpha * drop_diff
        s['drop_var_ema'] = (
            (1.0 - self._slow_alpha) * s['drop_var_ema']
            + self._slow_alpha * drop_diff ** 2
        )
        drop_std = max(sqrt(s['drop_var_ema']), 1e-6)
        surprise_signal = (drop - s['drop_ema']) / drop_std

        # Update similarity EMA *after* computing drop
        s['sim_ema'] = (
            (1.0 - self._slow_alpha) * s['sim_ema']
            + self._slow_alpha * best_similarity
        )

        # ── Layer 3: Leaky Accumulator ────────────────────────────
        input_signal = (
            self.NEWMA_WEIGHT * max(0.0, newma_signal)
            + self.SURPRISE_WEIGHT * max(0.0, surprise_signal)
        )
        accumulator = max(
            0.0,
            (1.0 - self._leak_rate) * s['accumulator'] + input_signal
        )

        # Dynamic boundary: harder to cross when drift is noisy or focus is active
        drift_ratio = s['drift_ema'] / drift_std if drift_std > 1e-6 else 1.0
        boundary = float(
            np.clip(
                self._boundary_base + 0.5 * drift_ratio + self._focus_modifier,
                self.BOUNDARY_MIN,
                self.BOUNDARY_MAX + self._focus_modifier,  # allow exceeding normal max during focus
            )
        )

        # Accumulator runaway cap (prevent slow drift from unbounded growth)
        accumulator = min(accumulator, boundary * 1.2)

        is_boundary = accumulator >= boundary

        if is_boundary:
            s['accumulator'] = 0.0  # reset on fire
        else:
            s['accumulator'] = accumulator

        confidence = min(1.0, accumulator / boundary)

        logger.debug(
            f"[BOUNDARY DETECTOR] thread={self.thread_id} "
            f"msgs={s['msg_count']} "
            f"acc={accumulator:.3f} bound={boundary:.3f} "
            f"newma={newma_signal:.3f} surprise={surprise_signal:.3f} "
            f"fired={is_boundary}"
        )

        return BoundaryResult(
            is_boundary=is_boundary,
            accumulator=accumulator,
            boundary=boundary,
            newma_signal=newma_signal,
            surprise_signal=surprise_signal,
            confidence=confidence,
            just_reset_from_silence=_just_reset,
        )

    def save_state(self):
        """Persist current state to Redis with 24h TTL."""
        try:
            r = RedisClientService.create_connection()
            r.setex(self._redis_key, self._REDIS_TTL, json.dumps(self._state))
        except Exception as e:
            logger.warning(
                f"[BOUNDARY DETECTOR] Failed to save state for thread "
                f"{self.thread_id}: {e}"
            )

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _window_to_alpha(window: int) -> float:
        """Convert EWMA window size N to smoothing factor α = 2/(N+1)."""
        return 2.0 / (max(1, int(window)) + 1)

    def _load_state(self) -> dict:
        """Load persisted detector state from Redis; fall back to cold-start."""
        try:
            r = RedisClientService.create_connection()
            raw = r.get(self._redis_key)
            if raw:
                state = json.loads(raw)
                logger.debug(
                    f"[BOUNDARY DETECTOR] Loaded state for thread {self.thread_id} "
                    f"(msgs={state.get('msg_count', 0)})"
                )
                return state
        except Exception as e:
            logger.warning(
                f"[BOUNDARY DETECTOR] Redis unavailable, cold-start mode: {e}"
            )
        return self._initial_state()

    def _initial_state(self) -> dict:
        """Return a fresh cold-start state dictionary."""
        return {
            'msg_count': 0,
            'ewma_fast': None,
            'ewma_slow': None,
            'drift_ema': 0.0,
            'drift_var_ema': 1e-4,
            'sim_ema': 0.5,
            'drop_ema': 0.0,
            'drop_var_ema': 1e-4,
            'accumulator': 0.0,
            'last_timestamp': None,
        }

    def _warm_emas(self, embedding: np.ndarray, similarity: float):
        """Update EMAs during cold-start phase (no boundary decisions yet)."""
        s = self._state
        if s['ewma_fast'] is None:
            s['ewma_fast'] = embedding.tolist()
            s['ewma_slow'] = embedding.tolist()
            s['sim_ema'] = similarity
        else:
            fast = np.array(s['ewma_fast'])
            slow = np.array(s['ewma_slow'])
            fast = (1.0 - self._fast_alpha) * fast + self._fast_alpha * embedding
            slow = (1.0 - self._slow_alpha) * slow + self._slow_alpha * embedding
            s['ewma_fast'] = fast.tolist()
            s['ewma_slow'] = slow.tolist()
            s['sim_ema'] = (
                (1.0 - self._slow_alpha) * s['sim_ema']
                + self._slow_alpha * similarity
            )

    def _reset_emas(self, embedding: np.ndarray, similarity: float):
        """Reset all EMAs to current values (dishabituation after long silence)."""
        s = self._state
        s['ewma_fast'] = embedding.tolist()
        s['ewma_slow'] = embedding.tolist()
        s['drift_ema'] = 0.0
        s['drift_var_ema'] = 1e-4
        s['sim_ema'] = similarity
        s['drop_ema'] = 0.0
        s['drop_var_ema'] = 1e-4
        s['accumulator'] = 0.0
