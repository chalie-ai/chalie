"""
Conversation Phase Service — deterministic per-thread conversation state machine.

Tracks what phase a conversation is in based purely on observable message signals.
No LLM — pure math on rolling windows maintained in MemoryStore.

States:
    opening   → Early exchanges or restart after a long gap
    exploring → Active topic changes or high question density
    deepening → Sustained topic focus with growing message depth
    resolving → Question density dropping, affirmative signals appearing
    closing   → Shortening messages, declining rate, or explicit goodbye words

Additional outputs:
    momentum  (0-1): exchange rate normalised against a 2 msg/min baseline (EMA-smoothed)
    direction : "deepening" | "sustaining" | "winding_down"

State is stored in MemoryStore at ``conversation_phase:{thread_id}`` (TTL 3600s).
"""

import json
import logging
import re
import threading
from typing import Optional

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_KEY_PREFIX = "conversation_phase"
_TTL = 3600  # 1 hour — matches thread expiry

# Momentum baseline: 2 messages per minute expressed as seconds between messages
_MOMENTUM_BASELINE_SECONDS = 30.0  # 2 msg/min = 1 msg every 30s
_MOMENTUM_EMA_ALPHA = 0.3

# How many items to keep in each rolling window
_WINDOW = 10

# Minimum signals required before a phase transition is committed
_TRANSITION_SIGNALS_REQUIRED = 2

# Gap threshold for resetting to "opening" (seconds)
_OPENING_GAP_SECONDS = 30 * 60  # 30 minutes

# Same-topic run required before "deepening" can fire
_DEEPENING_TOPIC_RUN = 5

# Question density thresholds
_EXPLORING_QUESTION_DENSITY = 0.5   # > this  → exploring candidate
_RESOLVING_QUESTION_DENSITY = 0.2   # < this  → resolving candidate

# Affirmative words (resolving signal)
_AFFIRMATIVES = frozenset([
    "ok", "okay", "got it", "makes sense", "thanks", "thank you", "understood",
    "right", "sure", "perfect", "great", "alright", "sounds good", "cool",
    "yes", "yep", "yup", "roger", "absolutely",
])

# Closing words (immediate transition, no stickiness required)
_CLOSING_WORDS = frozenset([
    "bye", "goodbye", "good bye", "talk later", "ttyl", "night", "goodnight",
    "good night", "see you", "see ya", "gotta go", "have to go", "cya",
    "take care", "later", "catch you later", "until next time",
])

# Singleton guard
_instance: Optional["ConversationPhaseService"] = None
_instance_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_questions(text: str) -> int:
    """Count question marks in a message."""
    return text.count("?")


def _has_affirmative(text: str) -> bool:
    """Return True if any affirmative phrase is present (case-insensitive)."""
    lower = text.lower()
    for phrase in _AFFIRMATIVES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", lower):
            return True
    return False


def _has_closing(text: str) -> bool:
    """Return True if any closing phrase is present (case-insensitive)."""
    lower = text.lower()
    for phrase in _CLOSING_WORDS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", lower):
            return True
    return False


def _window(lst: list, n: int = _WINDOW) -> list:
    """Keep only the last n items from a list."""
    return lst[-n:]


def _mean(values: list) -> float:
    """Mean of a non-empty list; 0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _is_increasing(values: list) -> bool:
    """True if the values list shows a general upward trend (last half > first half)."""
    if len(values) < 4:
        return False
    mid = len(values) // 2
    return _mean(values[mid:]) > _mean(values[:mid])


def _is_decreasing(values: list) -> bool:
    """True if the values list shows a general downward trend (last half < first half)."""
    if len(values) < 4:
        return False
    mid = len(values) // 2
    return _mean(values[mid:]) < _mean(values[:mid])


def _same_topic_run(topic_history: list) -> int:
    """Return the length of the current consecutive same-topic run at the tail."""
    if not topic_history:
        return 0
    current = topic_history[-1]
    count = 0
    for t in reversed(topic_history):
        if t == current:
            count += 1
        else:
            break
    return count


def _topic_changes_in_last(topic_history: list, n: int = 5) -> int:
    """Count topic changes in the last n entries."""
    tail = topic_history[-n:]
    if len(tail) < 2:
        return 0
    return sum(1 for i in range(1, len(tail)) if tail[i] != tail[i - 1])


def _question_density(question_counts: list, n: int = 5) -> float:
    """Fraction of messages in the last n that contained at least one question."""
    tail = question_counts[-n:]
    if not tail:
        return 0.0
    return sum(1 for q in tail if q > 0) / len(tail)


# ── Default state ─────────────────────────────────────────────────────────────

def _default_state() -> dict:
    now_str = utc_now().isoformat()
    return {
        "current": "opening",
        "momentum": 0.5,
        "direction": "sustaining",
        "exchange_count": 0,
        "topic_history": [],
        "message_lengths": [],
        "message_times": [],
        "question_counts": [],
        "last_exchange_at": now_str,
        "updated_at": now_str,
        # Pending-transition tracking (stickiness)
        "_pending_phase": None,
        "_pending_signals": 0,
    }


# ── Service ───────────────────────────────────────────────────────────────────

class ConversationPhaseService:
    """
    Deterministic conversation phase state machine.

    All state is stored in MemoryStore; no SQLite, no LLM.
    """

    # ── MemoryStore I/O ────────────────────────────────────────────────────

    def _store(self):
        from services.memory_client import MemoryClientService
        return MemoryClientService.create_connection()

    def _key(self, thread_id: str) -> str:
        return f"{_KEY_PREFIX}:{thread_id}"

    def _load(self, thread_id: str) -> dict:
        try:
            raw = self._store().get(self._key(thread_id))
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.debug(f"[PHASE] Load failed for {thread_id}: {exc}")
        return _default_state()

    def _save(self, thread_id: str, state: dict) -> None:
        try:
            self._store().set(self._key(thread_id), json.dumps(state), ex=_TTL)
        except Exception as exc:
            logger.debug(f"[PHASE] Save failed for {thread_id}: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        thread_id: str,
        message_text: str,
        is_user: bool,
        topic: str = None,
    ) -> dict:
        """
        Update phase state with a new message.

        Args:
            thread_id:    Conversation thread identifier.
            message_text: Raw message text.
            is_user:      True for user messages, False for assistant messages.
            topic:        Current topic label (optional — use last known if absent).

        Returns:
            Dict with keys: current, momentum, direction, updated_at.
        """
        state = self._load(thread_id)
        now = utc_now()
        now_str = now.isoformat()

        # ── Update rolling windows ─────────────────────────────────────────

        msg_len = len(message_text.strip())
        q_count = _count_questions(message_text)

        # Topic: fall back to last known if not supplied
        effective_topic = topic or (state["topic_history"][-1] if state["topic_history"] else "unknown")

        state["topic_history"] = _window(state["topic_history"] + [effective_topic])
        state["message_lengths"] = _window(state["message_lengths"] + [msg_len])
        state["message_times"] = _window(state["message_times"] + [now_str])

        # Only count questions from user messages
        if is_user:
            state["question_counts"] = _window(state["question_counts"] + [q_count])
            state["exchange_count"] = state.get("exchange_count", 0) + 1

        # ── Momentum ───────────────────────────────────────────────────────

        # Capture last_exchange_at BEFORE updating it so that gap detection
        # in _detect_phase sees the previous exchange timestamp, not the current one.
        prev_last_exchange_at = state.get("last_exchange_at")

        if prev_last_exchange_at and is_user:
            last_at = parse_utc(prev_last_exchange_at)
            elapsed = (now - last_at).total_seconds()
            if elapsed > 0:
                # Instantaneous rate: how many baselines fit in this gap
                # gap shorter than baseline → rate > 1 → capped at 1
                instant_rate = _MOMENTUM_BASELINE_SECONDS / elapsed
                instant_rate = min(instant_rate, 1.0)
            else:
                instant_rate = 1.0

            prev_momentum = state.get("momentum", 0.5)
            state["momentum"] = round(
                _MOMENTUM_EMA_ALPHA * instant_rate + (1.0 - _MOMENTUM_EMA_ALPHA) * prev_momentum,
                4,
            )

        if is_user:
            state["last_exchange_at"] = now_str

        # ── Direction ──────────────────────────────────────────────────────

        lengths = state["message_lengths"]
        times = state["message_times"]
        momentum_trend = self._momentum_trend(times)

        len_increasing = _is_increasing(lengths)
        len_decreasing = _is_decreasing(lengths)

        if (momentum_trend >= 0 or state["momentum"] >= 0.5) and len_increasing:
            state["direction"] = "deepening"
        elif momentum_trend < 0 or len_decreasing:
            state["direction"] = "winding_down"
        else:
            state["direction"] = "sustaining"

        # ── Phase detection ────────────────────────────────────────────────

        new_phase = self._detect_phase(state, message_text, is_user, now, prev_last_exchange_at)

        # Normalise sticky-closing sentinel before comparison
        effective_phase = new_phase.replace("~sticky", "") if new_phase else new_phase
        is_sticky = new_phase.endswith("~sticky") if new_phase else False

        if effective_phase != state["current"]:
            state = self._apply_transition(state, effective_phase, message_text, sticky=is_sticky)

        state["updated_at"] = now_str
        # Remove internal pending fields before persisting
        state_to_save = {k: v for k, v in state.items()}
        self._save(thread_id, state_to_save)

        logger.debug(
            f"[PHASE] thread={thread_id} phase={state['current']} "
            f"momentum={state['momentum']:.2f} direction={state['direction']}"
        )

        return {
            "current": state["current"],
            "momentum": state["momentum"],
            "direction": state["direction"],
            "updated_at": state["updated_at"],
        }

    def get_phase(self, thread_id: str) -> dict:
        """
        Return current phase for a thread.

        If no state exists, returns opening defaults.

        Returns:
            Dict with keys: current, momentum, direction, updated_at.
        """
        state = self._load(thread_id)
        return {
            "current": state.get("current", "opening"),
            "momentum": state.get("momentum", 0.5),
            "direction": state.get("direction", "sustaining"),
            "updated_at": state.get("updated_at", utc_now().isoformat()),
        }

    def reset(self, thread_id: str) -> None:
        """Clear phase state for a thread (called on thread expiry)."""
        try:
            self._store().delete(self._key(thread_id))
            logger.debug(f"[PHASE] Reset thread {thread_id}")
        except Exception as exc:
            logger.debug(f"[PHASE] Reset failed for {thread_id}: {exc}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _momentum_trend(self, message_times: list) -> float:
        """
        Compute momentum trend over the last 5 exchange timestamps.

        Returns a positive float for accelerating, negative for decelerating,
        0 for stable / insufficient data.
        """
        recent = message_times[-5:]
        if len(recent) < 3:
            return 0.0
        try:
            dts = [parse_utc(t) for t in recent]
            gaps = [(dts[i + 1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
            if len(gaps) < 2:
                return 0.0
            mid = len(gaps) // 2
            early_mean = _mean(gaps[:mid + 1])
            late_mean = _mean(gaps[mid:])
            if early_mean == 0:
                return 0.0
            # Negative = gaps shrinking = accelerating = positive trend
            return (early_mean - late_mean) / early_mean
        except Exception:
            return 0.0

    def _detect_phase(
        self,
        state: dict,
        message_text: str,
        is_user: bool,
        now,
        prev_last_exchange_at: str = None,
    ) -> str:
        """
        Evaluate transition conditions and return the best candidate phase.

        Args:
            prev_last_exchange_at: The last_exchange_at value captured BEFORE this
                message was processed — used for gap detection to avoid comparing
                now against itself.

        Phase priority when multiple conditions fire:
          closing > opening > resolving > deepening > exploring > (stay current)
        """
        current = state["current"]
        exchange_count = state.get("exchange_count", 0)
        topic_history = state["topic_history"]
        message_lengths = state["message_lengths"]
        question_counts = state["question_counts"]

        # ── Closing (immediate — keyword triggered) ───────────────────────
        if is_user and _has_closing(message_text):
            return "closing"

        # Three-message shortening trend → closing candidate (goes through stickiness)
        # Prefixed with "~" to signal to _apply_transition that stickiness applies.
        if len(message_lengths) >= 6:
            last3 = message_lengths[-3:]
            prev3 = message_lengths[-6:-3]
            if _mean(last3) < _mean(prev3) * 0.6:  # 40%+ drop
                return "closing~sticky"

        # ── Anti-false-close: substantive message after closing ─────────────
        if current == "closing" and is_user:
            # A substantive message is longer than 20 chars AND has no closing word
            if len(message_text.strip()) > 20 and not _has_closing(message_text):
                return "exploring"

        # ── Opening ────────────────────────────────────────────────────────
        # Only apply cold-start guard when we aren't already in a further-along phase.
        # Prevents a brief two-message thread that ends in "closing" from reverting to
        # "opening" on the very next message.
        if exchange_count <= 2 and current not in ("deepening", "resolving", "closing"):
            return "opening"

        # Use the pre-update timestamp so the gap reflects the actual inter-message
        # interval, not a zero-second gap against the current call's timestamp.
        last_at_str = prev_last_exchange_at or state.get("last_exchange_at")
        if is_user and last_at_str:
            last_at = parse_utc(last_at_str)
            gap_seconds = (now - last_at).total_seconds()
            if gap_seconds > _OPENING_GAP_SECONDS:
                return "opening"

        # ── Resolving ─────────────────────────────────────────────────────
        q_density = _question_density(question_counts)
        has_affirmative = is_user and _has_affirmative(message_text)

        if q_density < _RESOLVING_QUESTION_DENSITY and has_affirmative:
            return "resolving"

        # ── Deepening ─────────────────────────────────────────────────────
        topic_run = _same_topic_run(topic_history)
        len_increasing = _is_increasing(message_lengths)

        if topic_run >= _DEEPENING_TOPIC_RUN and len_increasing:
            return "deepening"

        # Also deepen when topic run is long enough even without length trend
        # (sustained focus is the primary signal; length is secondary)
        if topic_run >= _DEEPENING_TOPIC_RUN + 2:
            return "deepening"

        # ── Exploring ─────────────────────────────────────────────────────
        topic_changes = _topic_changes_in_last(topic_history, 5)
        if topic_changes > 1:
            return "exploring"

        if q_density > _EXPLORING_QUESTION_DENSITY:
            return "exploring"

        # No condition fired — stay in current phase
        return current

    def _apply_transition(
        self,
        state: dict,
        candidate: str,
        message_text: str,
        sticky: bool = False,
    ) -> dict:
        """
        Apply stickiness rules and commit or defer a phase transition.

        Keyword-triggered closing is immediate. Length-trend closing (sticky=True)
        and all other transitions require _TRANSITION_SIGNALS_REQUIRED consecutive
        signals pointing to the same new phase before the transition commits.
        """
        current = state["current"]

        # Closing via explicit keyword is always immediate; via length-trend is sticky
        if candidate == "closing" and not sticky:
            state["current"] = "closing"
            state["_pending_phase"] = None
            state["_pending_signals"] = 0
            logger.debug(f"[PHASE] Immediate transition → closing")
            return state

        # Anti-false-close: immediate transition back out of closing
        if current == "closing" and candidate == "exploring":
            state["current"] = "exploring"
            state["_pending_phase"] = None
            state["_pending_signals"] = 0
            logger.debug(f"[PHASE] Anti-false-close: closing → exploring")
            return state

        # Opening resets are always immediate (gap or cold start)
        if candidate == "opening":
            state["current"] = "opening"
            state["_pending_phase"] = None
            state["_pending_signals"] = 0
            logger.debug(f"[PHASE] Immediate transition → opening")
            return state

        # Sticky transitions for all other phases
        pending = state.get("_pending_phase")
        signals = state.get("_pending_signals", 0)

        if pending == candidate:
            signals += 1
        else:
            # New candidate — reset counter
            pending = candidate
            signals = 1

        if signals >= _TRANSITION_SIGNALS_REQUIRED:
            logger.debug(f"[PHASE] Committed transition: {current} → {candidate}")
            state["current"] = candidate
            state["_pending_phase"] = None
            state["_pending_signals"] = 0
        else:
            logger.debug(
                f"[PHASE] Pending transition: {current} → {candidate} "
                f"({signals}/{_TRANSITION_SIGNALS_REQUIRED} signals)"
            )
            state["_pending_phase"] = pending
            state["_pending_signals"] = signals

        return state


# ── Singleton ─────────────────────────────────────────────────────────────────

def get_conversation_phase_service() -> ConversationPhaseService:
    """Return the shared ConversationPhaseService singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ConversationPhaseService()
    return _instance
