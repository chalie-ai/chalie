# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Adaptive Layer Service — Rule-based communication style directives for LLM prompt injection.

Pure-Python, no LLM call, sub-1ms.  Measures the current message with
StyleMetricsService, blends with a per-thread EMA baseline (MemoryStore),
micro-preferences, and challenge tolerance, then outputs natural-language
directives ready to be appended to any frontal-cortex prompt.

Design principle: directives are behavioural framing, not imperative commands.
They tilt the LLM's default posture toward the user's observed patterns without
overriding its identity voice.
"""

import json
import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Directive rules
# Each entry: (low_threshold, high_threshold, low_directive, high_directive)
# A dimension fires its directive only when its value is outside the mid-range.
# ─────────────────────────────────────────────────────────────────────────────
DIRECTIVE_RULES: Dict[str, tuple] = {
    'verbosity': (
        4, 7,
        "Lead with the answer, then stop. One paragraph ceiling.",
        "Unpack reasoning and context. Layered responses land well here.",
    ),
    'directness': (
        4, 7,
        "Frame suggestions gently — open doors rather than point.",
        "Lead with a clear position, then expand if needed.",
    ),
    'formality': (
        4, 7,
        "Keep it casual and conversational. Avoid stiff phrasing.",
        "Maintain a composed, measured tone throughout.",
    ),
    'certainty': (
        4, 7,
        "Validate before expanding. Build confidence through clarity.",
        "Speak as equals. No hedging — conviction lands well here.",
    ),
    'pacing': (
        4, 7,
        "Match their pace — crisp, no filler.",
        "Give ideas room to breathe. Paragraphs and reflection.",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Challenge style tiers — used when an explicit challenge_tolerance is stored
# ─────────────────────────────────────────────────────────────────────────────
CHALLENGE_STYLE_TIERS: Dict[str, str] = {
    'low':    "Add nuance gently. Ask reflective questions. Present alternative perspectives without asserting them.",
    'medium': "Offer counterpoints when relevant. Explore tradeoffs openly.",
    'high':   "Pressure-test assumptions. Explore edge cases. Introduce contradictions when productive.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Cognitive load directives (keyed by load tier name)
# ─────────────────────────────────────────────────────────────────────────────
LOAD_DIRECTIVES: Dict[str, str] = {
    'LOW':      "",
    'NORMAL':   "",
    'HIGH':     "Lead with a summary. Use bullet points. Reduce abstraction. One thread at a time.",
    'OVERLOAD': "Compress to essentials. One core idea per response. Postpone tangents.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fork triggers — dimensions that benefit from a choice framing when ambiguous
# ─────────────────────────────────────────────────────────────────────────────
FORK_TRIGGERS: Dict[str, str] = {
    'verbosity': "I can give you the quick take — or go deeper if you'd like more.",
    'directness': "I can give you a straight answer — or frame it more gently if you prefer.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Growth reflection templates — one dimension -> list of variant phrasings
# ─────────────────────────────────────────────────────────────────────────────
GROWTH_REFLECTIONS: Dict[str, List[str]] = {
    'certainty': [
        "You're approaching decisions more decisively lately.",
        "There's a sharper clarity in how you're framing things.",
        "Your positions are landing with more conviction.",
    ],
    'verbosity': [
        "You're expressing more fully — taking space to think out loud.",
        "You're giving your thoughts more room to develop.",
        "There's more texture in how you're communicating.",
    ],
    'directness': [
        "You're getting more direct — leading with the point.",
        "There's less hedging in how you're presenting ideas.",
        "You're owning your positions more clearly.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Micro-preference labels — maps trait_key to natural language
# ─────────────────────────────────────────────────────────────────────────────
PREF_LABELS: Dict[str, str] = {
    'prefers_bullet_format': "User prefers bullet-point / list formatting.",
    'prefers_concise':       "User prefers concise, to-the-point responses.",
    'prefers_depth':         "User enjoys detailed, in-depth explanations.",
    'enjoys_challenge':      "User enjoys being challenged and hearing counterpoints.",
}

# Minimum observed messages before EMA baseline is trusted
_MIN_OBSERVATION_COUNT = 2

# Midpoint of the 1–10 scale — used for salience calculations
_MIDPOINT = 5.5

# Max directives in the final block (excluding micro-preferences)
_MAX_DIRECTIVES = 4

# Max micro-preference lines appended after directives
_MAX_MICRO_PREFS = 2

# MemoryStore TTLs (seconds)
_FORK_COOLDOWN_TTL = 300
_FORK_PENDING_TTL = 600
_GROWTH_COOLDOWN_TTL = 86400
_BASELINE_TTL = 86400 * 30  # 30 days

# EMA blend weight for the measured message vs stored baseline
# 0.3 new + 0.7 baseline — smooths single-message noise
_EMA_WEIGHT = 0.3


class AdaptiveLayerService:
    """
    Generates natural-language style directives for LLM prompt injection.

    Measures the current message with StyleMetricsService, blends with a
    per-thread EMA baseline stored in MemoryStore, applies slot-selection
    logic to avoid over-biasing, and returns a ready-to-inject directive block.

    No LLM call is made.  Designed to run in < 1ms on warm paths.
    """

    def __init__(self, db_service=None):
        """Accept optional db_service for API compatibility (not used internally)."""
        pass

    def generate_directives(
        self,
        thread_id: Optional[str] = None,
        current_signals: Optional[Dict] = None,
        working_memory_turns: Optional[List[Dict]] = None,
        current_message: Optional[str] = None,
    ) -> str:
        """
        Build the full adaptive directive block for injection into an LLM prompt.

        Args:
            thread_id:            Active conversation thread ID (used for EMA baseline
                                  and fork cooldown tracking).
            current_signals:      Dict with optional keys:
                                    'prompt_token_count' (int),
                                    'explicit_feedback' (str|None).
            working_memory_turns: List of turn dicts [{'role': ..., 'content': ...}].
            current_message:      Raw current user message text for style measurement.

        Returns:
            str: Formatted directive block, or empty string if cold-start gate fails.
        """
        try:
            current_signals = current_signals or {}
            working_memory_turns = working_memory_turns or []

            # -- 1. Measure current message style ---------------------------------
            from services.style_metrics_service import StyleMetricsService
            measured = StyleMetricsService().measure(current_message or "")

            # -- 2. Blend with EMA baseline (smooths noise) -----------------------
            style = self._blend_with_baseline(measured, thread_id)

            # -- 3. Cold-start gate -----------------------------------------------
            obs_count = self._get_observation_count(thread_id)
            if obs_count < _MIN_OBSERVATION_COUNT:
                self._increment_observation_count(thread_id)
                return ""
            self._increment_observation_count(thread_id)

            # -- 4. Fetch supporting data -----------------------------------------
            micro_prefs   = self._get_micro_preferences()
            challenge_tol = self._get_challenge_tolerance()

            # -- 5. Cognitive load ------------------------------------------------
            load_tier = self._estimate_cognitive_load(working_memory_turns, micro_prefs)
            load_directive = LOAD_DIRECTIVES.get(load_tier, "")

            # -- 6. Energy mirror -------------------------------------------------
            mirror_directive = self._get_energy_mirror_directive(style, current_signals)

            # -- 7. Build core directive slots ------------------------------------
            directives: List[str] = []

            if load_directive:
                directives.append(load_directive)

            # Pacing slot — always include if eligible
            pacing_directive = self._resolve_directive('pacing', style)
            if pacing_directive:
                directives.append(pacing_directive)

            # Cognitive slot dimensions: verbosity, directness, formality, certainty
            cognitive_dims = ['verbosity', 'directness', 'formality', 'certainty']

            # challenge_tolerance supersedes the certainty slot when explicitly stored
            challenge_handled = False
            if challenge_tol is not None:
                tier = self._challenge_tier(challenge_tol)
                challenge_directive = CHALLENGE_STYLE_TIERS.get(tier, "")
                if challenge_directive:
                    directives.append(challenge_directive)
                    challenge_handled = True

            # Rank remaining cognitive dims by salience
            scored: List[tuple] = []
            for dim in cognitive_dims:
                val = style.get(dim)
                if val is None:
                    continue
                salience = abs(float(val) - _MIDPOINT)
                directive_text = self._resolve_directive(dim, style)
                if directive_text:
                    scored.append((salience, dim, directive_text))

            scored.sort(key=lambda x: x[0], reverse=True)

            cognitive_added = 0
            for salience, dim, text in scored:
                if len(directives) >= _MAX_DIRECTIVES:
                    break
                if cognitive_added >= 2:
                    break
                directives.append(text)
                cognitive_added += 1

            # Energy mirror appended after core slots (if room and not already at cap)
            if mirror_directive and len(directives) < _MAX_DIRECTIVES:
                directives.append(mirror_directive)

            # -- 8. Fork directive (at most one) ----------------------------------
            fork_directive = self._get_fork_directive(style, thread_id)

            # -- 9. Growth reflection --------------------------------------------
            growth_reflection = self._get_growth_reflection()

            # -- 10. Assemble output ---------------------------------------------
            if not directives and not micro_prefs and not fork_directive and not growth_reflection:
                return ""

            lines: List[str] = ["## Adaptive Response Style"]

            for d in directives:
                lines.append(f"- {d}")

            if fork_directive:
                lines.append(f"- {fork_directive}")

            pref_lines = micro_prefs[:_MAX_MICRO_PREFS]
            for pref in pref_lines:
                lines.append(f"- {pref}")

            if growth_reflection:
                lines.append(
                    f'- If natural, weave in this observation: "{growth_reflection}". '
                    "One sentence, no fanfare."
                )

            lines.append(
                "When these directives conflict with your identity voice, "
                "your voice takes priority."
            )

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"[adaptive_layer] generate_directives failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # EMA baseline helpers (MemoryStore)
    # ─────────────────────────────────────────────────────────────────────────

    def _blend_with_baseline(self, measured: dict, thread_id: Optional[str]) -> dict:
        """Blend measured style with stored EMA baseline.

        If no baseline exists, measured values are used as-is.

        Args:
            measured: Dict of 5 dimension scores from StyleMetricsService.
            thread_id: Thread identifier for per-thread baseline key.

        Returns:
            Blended style dict (same 5 keys).
        """
        try:
            if not thread_id:
                return measured

            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            key = f"style_baseline:{thread_id}"
            raw = store.get(key)
            if not raw:
                store.set(key, json.dumps(measured), ex=_BASELINE_TTL)
                return measured

            baseline = json.loads(raw)
            blended = {}
            for dim in ('verbosity', 'directness', 'formality', 'certainty', 'pacing'):
                m = float(measured.get(dim, 5))
                b = float(baseline.get(dim, 5))
                blended[dim] = round(_EMA_WEIGHT * m + (1 - _EMA_WEIGHT) * b, 2)

            store.set(key, json.dumps(blended), ex=_BASELINE_TTL)
            return blended
        except Exception as e:
            logger.debug(f"[adaptive_layer] baseline blend failed: {e}")
            return measured

    def _get_observation_count(self, thread_id: Optional[str]) -> int:
        """Read per-thread observation counter from MemoryStore."""
        try:
            if not thread_id:
                return _MIN_OBSERVATION_COUNT  # no gating when thread unknown
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            val = store.get(f"style_obs_count:{thread_id}")
            return int(val) if val else 0
        except Exception:
            return _MIN_OBSERVATION_COUNT

    def _increment_observation_count(self, thread_id: Optional[str]) -> None:
        """Increment per-thread observation counter."""
        try:
            if not thread_id:
                return
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            key = f"style_obs_count:{thread_id}"
            store.incr(key)
            store.expire(key, _BASELINE_TTL)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Data retrieval helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_micro_preferences(self) -> List[str]:
        """
        Query user_traits for micro-preference rows and return natural-language labels.

        Selects WHERE category='micro_preference' AND confidence > 0.4,
        ordered by confidence DESC, limited to 3.

        Returns:
            List of human-readable preference strings (may be empty).
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_key, confidence
                    FROM user_traits
                    WHERE category = 'micro_preference'
                      AND confidence > 0.4
                    ORDER BY confidence DESC
                    LIMIT 3
                """)
                rows = cursor.fetchall()
                cursor.close()

            labels: List[str] = []
            for trait_key, _confidence in rows:
                label = PREF_LABELS.get(trait_key)
                if label:
                    labels.append(label)
            return labels
        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_micro_preferences failed: {e}")
            return []

    def _get_challenge_tolerance(self) -> Optional[float]:
        """
        Retrieve explicit challenge tolerance float (1-10) from user_traits.

        Stored as trait_key='challenge_tolerance', category='micro_preference'.

        Returns:
            float if found, None otherwise.
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_value
                    FROM user_traits
                    WHERE trait_key = 'challenge_tolerance'
                      AND category = 'micro_preference'
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                cursor.close()

            if not row:
                return None
            return float(row[0])
        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_challenge_tolerance failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Cognitive load estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_cognitive_load(
        self,
        working_memory_turns: List[Dict],
        micro_prefs: Optional[List[str]] = None,
    ) -> str:
        """
        Estimate cognitive load from working-memory user turns.

        Signals:
          +2  Reply length trending down (last 3 user turns getting shorter)
          +2  More than 1 question mark per 20 words (confusion signal)
          +1  'prefers_concise' micro-preference active
          +1  2 of the last 3 user turns are < 10 words

        Thresholds:
          score < 2  -> LOW
          score < 4  -> NORMAL
          score < 6  -> HIGH
          score >= 6 -> OVERLOAD

        Returns:
            str: One of LOW | NORMAL | HIGH | OVERLOAD
        """
        try:
            micro_prefs = micro_prefs or []

            user_turns: List[str] = [
                t.get('content', '')
                for t in working_memory_turns
                if t.get('role', '').lower() in ('user', 'human')
            ]

            score = 0

            recent = user_turns[-3:] if len(user_turns) >= 3 else user_turns
            if len(recent) >= 3:
                lengths = [len(t.split()) for t in recent]
                if lengths[0] > lengths[1] > lengths[2]:
                    score += 2

            for turn in recent:
                words = turn.split()
                if not words:
                    continue
                qmarks = turn.count('?')
                ratio = qmarks / (len(words) / 20) if len(words) > 0 else 0
                if ratio > 1:
                    score += 2
                    break

            if any('concise' in p.lower() for p in micro_prefs):
                score += 1

            last_three = user_turns[-3:] if len(user_turns) >= 3 else user_turns
            short_count = sum(1 for t in last_three if len(t.split()) < 10)
            if short_count >= 2:
                score += 1

            if score < 2:
                return 'LOW'
            elif score < 4:
                return 'NORMAL'
            elif score < 6:
                return 'HIGH'
            else:
                return 'OVERLOAD'

        except Exception as e:
            logger.warning(f"[adaptive_layer] _estimate_cognitive_load failed: {e}")
            return 'NORMAL'

    # ─────────────────────────────────────────────────────────────────────────
    # Energy mirror directive
    # ─────────────────────────────────────────────────────────────────────────

    def _get_energy_mirror_directive(
        self,
        style: dict,
        current_signals: dict,
    ) -> str:
        """
        Detect energy mismatches between baseline verbosity and the current message.

        Args:
            style:           Communication style dict.
            current_signals: Dict with optional 'prompt_token_count' (int) and
                             'explicit_feedback' (str|None).

        Returns:
            str: Mirror directive, or "" if no notable deviation.
        """
        try:
            if not style:
                return ""

            explicit_feedback = current_signals.get('explicit_feedback')
            if explicit_feedback and str(explicit_feedback).lower() in (
                'negative', 'friction', 'dislike', 'bad', 'wrong'
            ):
                return "User shows friction — acknowledge before addressing content."

            verbosity = style.get('verbosity')
            if verbosity is None:
                return ""

            token_count = current_signals.get('prompt_token_count')
            if token_count is None:
                return ""

            approx_words = int(token_count * 0.75)

            if float(verbosity) >= 7 and approx_words < 10:
                return "User is being unusually brief — match that energy first."

            if float(verbosity) <= 3 and approx_words > 80:
                return "User is going deeper than usual — give ideas room."

            return ""

        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_energy_mirror_directive failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Fork directive
    # ─────────────────────────────────────────────────────────────────────────

    def _get_fork_directive(
        self,
        style: dict,
        thread_id: Optional[str],
    ) -> str:
        """
        Suggest a response-path fork when a dimension is in the ambiguous mid-range.

        Only fires when:
          - thread_id is provided
          - a dimension in FORK_TRIGGERS sits in the 4-7 mid-range (ambiguous)
          - the MemoryStore cooldown key is not set

        Sets adaptive_fork_pending:{thread_id} (TTL 600s) when a fork is chosen.

        Returns:
            str: Fork suggestion sentence, or "" if conditions are not met.
        """
        try:
            if not thread_id or not style:
                return ""

            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()

            cooldown_key = f"adaptive_fork_cooldown:{thread_id}"
            if store.exists(cooldown_key):
                return ""

            candidates: List[tuple] = []
            for dim, fork_text in FORK_TRIGGERS.items():
                val = style.get(dim)
                if val is None:
                    continue
                val_f = float(val)
                if 4 <= val_f <= 7:
                    distance_from_mid = abs(val_f - _MIDPOINT)
                    candidates.append((distance_from_mid, dim, fork_text))

            if not candidates:
                return ""

            candidates.sort(key=lambda x: x[0])
            _, chosen_dim, fork_text = candidates[0]

            pending_key = f"adaptive_fork_pending:{thread_id}"
            store.set(pending_key, chosen_dim, ex=_FORK_PENDING_TTL)

            return fork_text

        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_fork_directive failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Growth reflection
    # ─────────────────────────────────────────────────────────────────────────

    def _get_growth_reflection(self) -> str:
        """
        Return a single-sentence growth observation if a pattern qualifies.

        Queries user_traits for growth_signal:* rows (category='core'), parses
        the JSON value, and checks for consecutive_cycles >= 6.

        Respects a 24-hour MemoryStore cooldown.

        Returns:
            str: Randomly selected reflection sentence, or "".
        """
        try:
            from services.memory_client import MemoryClientService
            from services.database_service import get_shared_db_service

            store = MemoryClientService.create_connection()
            cooldown_key = "adaptive_growth_reflection_cooldown"
            if store.exists(cooldown_key):
                return ""

            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_key, trait_value
                    FROM user_traits
                    WHERE trait_key LIKE 'growth_signal:%'
                      AND category = 'core'
                """)
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return ""

            eligible: List[tuple] = []
            for trait_key, trait_value in rows:
                try:
                    data = json.loads(trait_value)
                    cycles = int(data.get('consecutive_cycles', 0))
                    if cycles >= 6:
                        dim = trait_key.split(':', 1)[1] if ':' in trait_key else trait_key
                        eligible.append((cycles, dim))
                except Exception:
                    continue

            if not eligible:
                return ""

            eligible.sort(key=lambda x: x[0], reverse=True)
            _, strongest_dim = eligible[0]

            variants = GROWTH_REFLECTIONS.get(strongest_dim)
            if not variants:
                return ""

            reflection = random.choice(variants)

            store.set(cooldown_key, '1', ex=_GROWTH_COOLDOWN_TTL)

            return reflection

        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_growth_reflection failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_directive(self, dim: str, style: dict) -> str:
        """
        Return the appropriate directive string for a given dimension.

        Returns "" if the value is within the neutral mid-range (4 < val < 7)
        or the dimension is not present in style.
        """
        rule = DIRECTIVE_RULES.get(dim)
        if not rule:
            return ""
        val = style.get(dim)
        if val is None:
            return ""
        val_f = float(val)
        low_thresh, high_thresh, low_text, high_text = rule
        if val_f <= low_thresh:
            return low_text
        if val_f >= high_thresh:
            return high_text
        return ""

    @staticmethod
    def _challenge_tier(tolerance: float) -> str:
        """
        Map a 1-10 challenge tolerance value to a tier name.

          1-3  -> low
          4-7  -> medium
          8-10 -> high
        """
        if tolerance <= 3:
            return 'low'
        if tolerance <= 7:
            return 'medium'
        return 'high'
