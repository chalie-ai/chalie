"""
Two-Signal Topic Boundary Detector.

Fires when EITHER:
  1. A discourse marker is detected in the message text, OR
  2. Both consecutive similarity AND window similarity drop below
     self-calibrating thresholds (mean - K*std)

State persisted in MemoryStore per thread_id with 24h TTL.
Degrades gracefully to cold-start mode if MemoryStore is unavailable.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from services.memory_store import get_shared_store

logger = logging.getLogger(__name__)

# ── Cold start ───────────────────────────────────────────────────────────────
COLD_START_MSGS = 6

# ── Short message gate ───────────────────────────────────────────────────────
# Messages with <= this many words skip the two-signal check.  Short terse
# answers ("yes", "38", "iam 38") produce semantically generic embeddings that
# are statistically far from any topic centroid.  The two-signal detector
# correctly flags them as outliers, but they're contextual continuations, not
# topic changes.  Discourse markers still fire for short messages.
SHORT_MSG_MAX_WORDS = 4

# ── Stale baseline reset (dishabituation after long silence) ─────────────────
STALE_GAP_SECONDS = 2700  # 45 minutes

# ── Discourse marker patterns ────────────────────────────────────────────────
# Ordered by specificity (longer/more specific patterns first).
# Only markers with high precision for topic switching — no "instead of",
# "actually", or other patterns that commonly appear in continuations.
_MARKER_PATTERNS = [
    # Explicit switch declarations
    re.compile(r'\bcompletely\s+different\b', re.I),
    re.compile(r'\btotally\s+(?:different|unrelated)\b', re.I),
    re.compile(r'\bunrelated\b', re.I),
    re.compile(r'\boff\s+topic\b', re.I),
    re.compile(r'\bswitch(?:ing)?\s+gears\b', re.I),
    re.compile(r'\bchang(?:e|ing)\s+(?:the\s+)?(?:subject|topic)\b', re.I),

    # Bridging / association markers
    re.compile(r'\bspeaking\s+of\b', re.I),
    re.compile(r'\bthat\s+reminds\s+me\b', re.I),
    re.compile(r'\bgot\s+me\s+thinking\s+about\b', re.I),
    re.compile(r'\bon\s+(?:a\s+)?(?:different|personal|separate|side)\s+note\b', re.I),

    # Topic return
    re.compile(r'\bback\s+to\s+(?:the\s+|my\s+)?\w+', re.I),
    re.compile(r'\banyway(?:s)?\b', re.I),

    # Attention redirects
    re.compile(r'\bby\s+the\s+way\b', re.I),
    re.compile(r'\bon\s+that\s+note\b', re.I),

    # Exclamatory interruptions (start of message only)
    re.compile(r'^oh\s+(?:wait|my\s+god|hey)\b', re.I),
]


@dataclass
class BoundaryResult:
    """Result of a single boundary-detection step."""

    is_boundary: bool
    confidence: float       # 0-1, how confident the detection is
    trigger: str            # 'none' | 'marker' | 'two_signal' | 'both' | 'stale_gap' | 'cold_start'
    consec_sim: float       # cosine similarity to previous message
    window_sim: float       # cosine similarity to window centroid
    marker_found: Optional[str]  # matched marker text, if any
    just_reset_from_silence: bool = False


class TwoSignalBoundaryService:
    """
    Self-calibrating topic boundary detector.

    Two independent signals (consecutive similarity and window centroid
    similarity) must both drop for a boundary to fire.  Discourse markers
    provide a high-precision fast path that bypasses the two-signal gate.

    State is persisted in MemoryStore at ``two_signal_boundary:{thread_id}``
    with a 24h TTL.  If MemoryStore is unavailable the detector degrades
    gracefully to cold-start mode.
    """

    # Defaults
    DEFAULT_WINDOW_SIZE = 5
    DEFAULT_THRESHOLD_K = 1.0

    _STORE_TTL = 86400  # 24h

    def __init__(
        self,
        thread_id: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold_k: float = DEFAULT_THRESHOLD_K,
    ):
        self.thread_id = thread_id
        self.window_size = window_size
        self.threshold_k = threshold_k
        self._store_key = f"two_signal_boundary:{thread_id}"
        self._state = self._load_state()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def update(
        self,
        embedding: np.ndarray,
        message_text: str = '',
    ) -> BoundaryResult:
        """
        Process one message and determine whether a topic boundary occurred.

        Args:
            embedding: L2-normalised embedding of the current message.
            message_text: Raw message text for discourse marker detection.

        Returns:
            BoundaryResult with detection outcome and diagnostics.
        """
        s = self._state
        now = time.time()

        # Safety: re-normalise
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # ── Stale gap reset ─────────────────────────────────────
        just_reset = False
        if s['last_timestamp'] is not None:
            if now - s['last_timestamp'] > STALE_GAP_SECONDS:
                self._reset(embedding)
                just_reset = True
                logger.debug(
                    f"[TWO-SIGNAL] Dishabituation reset for thread {self.thread_id}"
                )
                s['last_timestamp'] = now
                return BoundaryResult(
                    is_boundary=True,
                    confidence=1.0,
                    trigger='stale_gap',
                    consec_sim=0.0,
                    window_sim=0.0,
                    marker_found=None,
                    just_reset_from_silence=True,
                )

        s['last_timestamp'] = now

        # ── Discourse marker check ──────────────────────────────
        marker_match = self._detect_marker(message_text) if message_text else None

        # ── Short message gate ────────────────────────────────────
        # Terse answers (≤ SHORT_MSG_MAX_WORDS) skip two-signal.
        # Their embeddings are too generic to produce meaningful
        # similarity scores.  Still add to history for centroid.
        word_count = len(message_text.split()) if message_text else 0
        is_short = word_count <= SHORT_MSG_MAX_WORDS

        # ── First message: seed state, no detection ─────────────
        embeddings = s['embeddings']
        n = len(embeddings)

        if n == 0:
            embeddings.append(embedding.tolist())
            s['msg_count'] = 1
            # First message with a marker is likely a new topic introduction
            return BoundaryResult(
                is_boundary=bool(marker_match),
                confidence=0.8 if marker_match else 0.0,
                trigger='marker' if marker_match else 'cold_start',
                consec_sim=0.0,
                window_sim=0.0,
                marker_found=marker_match,
                just_reset_from_silence=just_reset,
            )

        # ── Compute signals ─────────────────────────────────────
        prev_emb = np.array(embeddings[-1])
        consec_sim = float(np.dot(embedding, prev_emb))

        window = [np.array(e) for e in embeddings[-self.window_size:]]
        centroid = np.mean(window, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        window_sim = float(np.dot(embedding, centroid))

        # Add embedding to history
        embeddings.append(embedding.tolist())
        # Keep only what we need (window + some margin for stats)
        max_keep = max(self.window_size * 4, 30)
        if len(embeddings) > max_keep:
            embeddings[:] = embeddings[-max_keep:]

        s['msg_count'] += 1

        # ── Cold start: gather stats ────────────────────────────
        consec_hist = s['consec_history']
        window_hist = s['window_history']

        if len(consec_hist) < COLD_START_MSGS:
            consec_hist.append(consec_sim)
            window_hist.append(window_sim)
            # During cold start, only markers can fire
            return BoundaryResult(
                is_boundary=bool(marker_match),
                confidence=0.8 if marker_match else 0.0,
                trigger='marker' if marker_match else 'cold_start',
                consec_sim=round(consec_sim, 4),
                window_sim=round(window_sim, 4),
                marker_found=marker_match,
                just_reset_from_silence=just_reset,
            )

        # ── Self-calibrating thresholds ─────────────────────────
        consec_mean = float(np.mean(consec_hist))
        consec_std = max(float(np.std(consec_hist)), 0.01)
        window_mean = float(np.mean(window_hist))
        window_std = max(float(np.std(window_hist)), 0.01)

        consec_thresh = consec_mean - self.threshold_k * consec_std
        window_thresh = window_mean - self.threshold_k * window_std

        consec_drop = consec_sim < consec_thresh
        window_drop = window_sim < window_thresh

        two_signal_fire = consec_drop and window_drop

        # Short messages: suppress two-signal, markers only
        if is_short and two_signal_fire:
            logger.debug(
                f"[TWO-SIGNAL] Short message gate suppressed boundary "
                f"({word_count} words): '{message_text[:40]}'"
            )
            two_signal_fire = False

        # ── Combine signals ─────────────────────────────────────
        is_boundary = two_signal_fire or bool(marker_match)

        if two_signal_fire and marker_match:
            trigger = 'both'
            confidence = 0.95
        elif marker_match:
            trigger = 'marker'
            confidence = 0.80
        elif two_signal_fire:
            trigger = 'two_signal'
            # Confidence based on how far below thresholds
            consec_z = (consec_mean - consec_sim) / consec_std if consec_std > 0.01 else 0.0
            window_z = (window_mean - window_sim) / window_std if window_std > 0.01 else 0.0
            confidence = min(1.0, (consec_z + window_z) / 4.0)
        else:
            trigger = 'none'
            confidence = 0.0

        # Only update stats on non-boundary messages (keep baseline clean)
        if not is_boundary:
            consec_hist.append(consec_sim)
            window_hist.append(window_sim)
            # Keep history bounded
            max_hist = 50
            if len(consec_hist) > max_hist:
                consec_hist[:] = consec_hist[-max_hist:]
                window_hist[:] = window_hist[-max_hist:]

        logger.debug(
            f"[TWO-SIGNAL] thread={self.thread_id} "
            f"msgs={s['msg_count']} "
            f"consec={consec_sim:.3f}(t={consec_thresh:.3f}) "
            f"window={window_sim:.3f}(t={window_thresh:.3f}) "
            f"marker={marker_match} "
            f"fired={is_boundary}({trigger})"
        )

        return BoundaryResult(
            is_boundary=is_boundary,
            confidence=round(confidence, 3),
            trigger=trigger,
            consec_sim=round(consec_sim, 4),
            window_sim=round(window_sim, 4),
            marker_found=marker_match,
            just_reset_from_silence=just_reset,
        )

    def save_state(self):
        """Persist current state to MemoryStore with 24h TTL."""
        try:
            store = get_shared_store()
            store.setex(
                self._store_key,
                self._STORE_TTL,
                json.dumps(self._state),
            )
        except Exception as e:
            logger.warning(
                f"[TWO-SIGNAL] Failed to save state for thread "
                f"{self.thread_id}: {e}"
            )

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_marker(text: str) -> Optional[str]:
        """Check message text for discourse markers. Returns matched text or None."""
        for pattern in _MARKER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group()
        return None

    def _load_state(self) -> dict:
        """Load persisted state from MemoryStore; fall back to fresh state."""
        try:
            store = get_shared_store()
            raw = store.get(self._store_key)
            if raw:
                state = json.loads(raw)
                logger.debug(
                    f"[TWO-SIGNAL] Loaded state for thread {self.thread_id} "
                    f"(msgs={state.get('msg_count', 0)})"
                )
                return state
        except Exception as e:
            logger.warning(
                f"[TWO-SIGNAL] MemoryStore unavailable, cold-start mode: {e}"
            )
        return self._initial_state()

    def _initial_state(self) -> dict:
        return {
            'msg_count': 0,
            'embeddings': [],
            'consec_history': [],
            'window_history': [],
            'last_timestamp': None,
        }

    def _reset(self, embedding: np.ndarray):
        """Reset all state (dishabituation after long silence)."""
        s = self._state
        s['embeddings'] = [embedding.tolist()]
        s['consec_history'] = []
        s['window_history'] = []
        s['msg_count'] = 1
