# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Adaptive Layer Service — Rule-based communication style directives for LLM prompt injection.

Pure-Python, no LLM call, sub-1ms.  Reads stored communication style,
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
    'emotional_valence': (
        4, 7,
        "Stay structured and evidence-focused. Logic over feeling.",
        "Acknowledge the emotional context before moving into solutions.",
    ),
    'certainty_level': (
        4, 7,
        "Validate before expanding. Build confidence through clarity.",
        "Speak as equals. No hedging — conviction lands well here.",
    ),
    'challenge_appetite': (
        4, 7,
        "Build on their perspective. Challenge only when invited.",
        "Introduce counterpoints and edge cases. Push the thinking.",
    ),
    'depth_preference': (
        4, 7,
        "Surface practical, actionable next steps.",
        "Explore layers — perspectives, connections, reflective prompts.",
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
    'depth_preference': "I can give you the quick take — or we can dig deeper.",
    'challenge_appetite': "I can build on this with you — or poke some holes if you'd rather stress-test it.",
    'abstraction_level': "I can map out practical steps — or zoom out to the bigger picture.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Growth reflection templates — one dimension → list of variant phrasings
# ─────────────────────────────────────────────────────────────────────────────
GROWTH_REFLECTIONS: Dict[str, List[str]] = {
    'certainty_level': [
        "You're approaching decisions more decisively lately.",
        "There's a sharper clarity in how you're framing things.",
        "Your positions are landing with more conviction.",
    ],
    'depth_preference': [
        "Your thinking is going deeper — exploring more layers.",
        "You're pulling threads further than before.",
        "There's more depth in how you're engaging with ideas.",
    ],
    'challenge_appetite': [
        "You're leaning into challenge more than before.",
        "You seem more comfortable with friction in ideas.",
        "There's a shift — you're seeking the harder questions.",
    ],
    'verbosity': [
        "You're expressing more fully — taking space to think out loud.",
        "You're giving your thoughts more room to develop.",
        "There's more texture in how you're communicating.",
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

# Minimum observed interactions before directives are emitted (cold-start gate)
_MIN_OBSERVATION_COUNT = 2

# Midpoint of the 1–10 scale — used for salience calculations
_MIDPOINT = 5.5

# Max directives in the final block (excluding micro-preferences)
_MAX_DIRECTIVES = 4

# Max micro-preference lines appended after directives
_MAX_MICRO_PREFS = 2

# Redis cooldown TTLs (seconds)
_FORK_COOLDOWN_TTL = 300
_FORK_PENDING_TTL = 600
_GROWTH_COOLDOWN_TTL = 86400


class AdaptiveLayerService:
    """
    Generates natural-language style directives for LLM prompt injection.

    Reads persisted communication style, micro-preferences, and challenge
    tolerance from user_traits, applies slot-selection logic to avoid
    over-biasing, and returns a ready-to-inject directive block.

    No LLM call is made.  Designed to run in < 1ms on warm paths (DB results
    will be the dominant cost on first call).
    """

    def generate_directives(
        self,
        user_id: str = 'primary',
        thread_id: Optional[str] = None,
        current_signals: Optional[Dict] = None,
        working_memory_turns: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build the full adaptive directive block for injection into an LLM prompt.

        Args:
            user_id:              User identifier (default 'primary').
            thread_id:            Active conversation thread ID (used for fork).
            current_signals:      Dict with optional keys:
                                    'prompt_token_count' (int),
                                    'explicit_feedback' (str|None).
            working_memory_turns: List of turn dicts [{'role': ..., 'content': ...}].

        Returns:
            str: Formatted directive block, or empty string if cold-start gate fails.
        """
        try:
            current_signals = current_signals or {}
            working_memory_turns = working_memory_turns or []

            # ── 1. Fetch style ────────────────────────────────────────────────
            style = self._get_communication_style(user_id)

            # ── 2. Cold-start gate ────────────────────────────────────────────
            if not style:
                return ""
            observation_count = style.get('_observation_count', 0)
            if observation_count < _MIN_OBSERVATION_COUNT:
                return ""

            # ── 3. Fetch supporting data ──────────────────────────────────────
            micro_prefs    = self._get_micro_preferences(user_id)
            challenge_tol  = self._get_challenge_tolerance(user_id)

            # ── 4. Cognitive load ─────────────────────────────────────────────
            load_tier = self._estimate_cognitive_load(working_memory_turns, micro_prefs)
            load_directive = LOAD_DIRECTIVES.get(load_tier, "")

            # ── 5. Energy mirror ──────────────────────────────────────────────
            mirror_directive = self._get_energy_mirror_directive(style, current_signals)

            # ── 6. Build core directive slots ─────────────────────────────────
            directives: List[str] = []

            # Load directive takes the first slot when load is HIGH/OVERLOAD
            if load_directive:
                directives.append(load_directive)

            # Pacing slot — always include if eligible
            pacing_directive = self._resolve_directive('pacing', style)
            if pacing_directive:
                directives.append(pacing_directive)

            # Cognitive slot dimensions: verbosity, directness, depth_preference,
            # challenge_appetite — ranked by salience (abs distance from midpoint)
            cognitive_dims = ['verbosity', 'directness', 'depth_preference', 'challenge_appetite']

            # challenge_appetite is superseded by challenge_style_tiers when
            # an explicit tolerance value exists — exclude it from the scored
            # cognitive slot so we don't double-count.
            challenge_handled = False
            if challenge_tol is not None:
                tier = self._challenge_tier(challenge_tol)
                challenge_directive = CHALLENGE_STYLE_TIERS.get(tier, "")
                if challenge_directive:
                    directives.append(challenge_directive)
                    challenge_handled = True
                cognitive_dims = [d for d in cognitive_dims if d != 'challenge_appetite']

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

            # Fill remaining slots up to cap (2 from cognitive dims)
            cognitive_added = 0
            for salience, dim, text in scored:
                if len(directives) >= _MAX_DIRECTIVES:
                    break
                if cognitive_added >= 2:
                    break
                directives.append(text)
                cognitive_added += 1

            # Emotional slot — include only if salience > 1.5 on either dimension
            if len(directives) < _MAX_DIRECTIVES:
                emotional_directive = self._resolve_emotional_slot(style)
                if emotional_directive:
                    directives.append(emotional_directive)

            # Energy mirror appended after core slots (if room and not already at cap)
            if mirror_directive and len(directives) < _MAX_DIRECTIVES:
                directives.append(mirror_directive)

            # ── 7. Fork directive (at most one) ───────────────────────────────
            fork_directive = self._get_fork_directive(style, thread_id)

            # ── 8. Growth reflection ──────────────────────────────────────────
            growth_reflection = self._get_growth_reflection(user_id)

            # ── 9. Assemble output ────────────────────────────────────────────
            if not directives and not micro_prefs and not fork_directive and not growth_reflection:
                return ""

            lines: List[str] = ["## Adaptive Response Style"]

            for d in directives:
                lines.append(f"- {d}")

            if fork_directive:
                lines.append(f"- {fork_directive}")

            # Micro-preferences (up to 2)
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
    # Data retrieval helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_communication_style(self, user_id: str) -> dict:
        """
        Retrieve stored communication style dict from UserTraitService.

        Returns:
            dict with dimension scores (1–10 scale) and '_observation_count', or {}.
        """
        try:
            from services.user_trait_service import UserTraitService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            trait_svc = UserTraitService(db)
            return trait_svc.get_communication_style(user_id=user_id)
        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_communication_style failed: {e}")
            return {}

    def _get_micro_preferences(self, user_id: str) -> List[str]:
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
                    WHERE user_id = %s
                      AND category = 'micro_preference'
                      AND confidence > 0.4
                    ORDER BY confidence DESC
                    LIMIT 3
                """, (user_id,))
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

    def _get_challenge_tolerance(self, user_id: str) -> Optional[float]:
        """
        Retrieve explicit challenge tolerance float (1–10) from user_traits.

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
                    WHERE user_id = %s
                      AND trait_key = 'challenge_tolerance'
                      AND category = 'micro_preference'
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (user_id,))
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
          score < 2  → LOW
          score < 4  → NORMAL
          score < 6  → HIGH
          score >= 6 → OVERLOAD

        Returns:
            str: One of LOW | NORMAL | HIGH | OVERLOAD
        """
        try:
            micro_prefs = micro_prefs or []

            # Extract user turns only
            user_turns: List[str] = [
                t.get('content', '')
                for t in working_memory_turns
                if t.get('role', '').lower() in ('user', 'human')
            ]

            score = 0

            # ── Length trend: last 3 user turns decreasing ────────────────────
            recent = user_turns[-3:] if len(user_turns) >= 3 else user_turns
            if len(recent) >= 3:
                lengths = [len(t.split()) for t in recent]
                if lengths[0] > lengths[1] > lengths[2]:
                    score += 2

            # ── Question-mark density (confusion signal) ──────────────────────
            for turn in recent:
                words = turn.split()
                if not words:
                    continue
                qmarks = turn.count('?')
                ratio = qmarks / (len(words) / 20) if len(words) > 0 else 0
                if ratio > 1:
                    score += 2
                    break  # count once even if multiple turns trigger it

            # ── prefers_concise micro-preference ─────────────────────────────
            if any('concise' in p.lower() for p in micro_prefs):
                score += 1

            # ── Short turn density in last 3 ─────────────────────────────────
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

            # Approximate word count from token count (rough: ~0.75 tokens/word)
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
          - a dimension in FORK_TRIGGERS sits in the 4–7 mid-range (ambiguous)
          - the Redis cooldown key is not set

        Sets adaptive_fork_pending:{thread_id} (TTL 600s) when a fork is chosen.

        Returns:
            str: Fork suggestion sentence, or "" if conditions are not met.
        """
        try:
            if not thread_id or not style:
                return ""

            from services.redis_client import RedisClientService
            r = RedisClientService.create_connection()

            cooldown_key = f"adaptive_fork_cooldown:{thread_id}"
            if r.exists(cooldown_key):
                return ""

            # Find eligible dims: in fork triggers AND in ambiguous zone (4–7)
            candidates: List[tuple] = []
            for dim, fork_text in FORK_TRIGGERS.items():
                val = style.get(dim)
                if val is None:
                    continue
                val_f = float(val)
                if 4 <= val_f <= 7:
                    # Closeness to midpoint = ambiguity (lower distance = more ambiguous)
                    distance_from_mid = abs(val_f - _MIDPOINT)
                    candidates.append((distance_from_mid, dim, fork_text))

            if not candidates:
                return ""

            # Pick the most ambiguous (closest to midpoint)
            candidates.sort(key=lambda x: x[0])
            _, chosen_dim, fork_text = candidates[0]

            # Set pending key so downstream can observe which fork was offered
            pending_key = f"adaptive_fork_pending:{thread_id}"
            r.set(pending_key, chosen_dim, ex=_FORK_PENDING_TTL)

            return fork_text

        except Exception as e:
            logger.warning(f"[adaptive_layer] _get_fork_directive failed: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Growth reflection
    # ─────────────────────────────────────────────────────────────────────────

    def _get_growth_reflection(self, user_id: str) -> str:
        """
        Return a single-sentence growth observation if a pattern qualifies.

        Queries user_traits for growth_signal:* rows (category='core'), parses
        the JSON value, and checks for consecutive_cycles >= 6.

        Respects a 24-hour per-user Redis cooldown.

        Returns:
            str: Randomly selected reflection sentence, or "".
        """
        try:
            from services.redis_client import RedisClientService
            from services.database_service import get_shared_db_service

            r = RedisClientService.create_connection()
            cooldown_key = f"adaptive_growth_reflection_cooldown:{user_id}"
            if r.exists(cooldown_key):
                return ""

            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_key, trait_value
                    FROM user_traits
                    WHERE user_id = %s
                      AND trait_key LIKE 'growth_signal:%%'
                      AND category = 'core'
                """, (user_id,))
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return ""

            # Parse and filter for sufficient consecutive cycles
            eligible: List[tuple] = []
            for trait_key, trait_value in rows:
                try:
                    data = json.loads(trait_value)
                    cycles = int(data.get('consecutive_cycles', 0))
                    if cycles >= 6:
                        # Extract dimension name after 'growth_signal:'
                        dim = trait_key.split(':', 1)[1] if ':' in trait_key else trait_key
                        eligible.append((cycles, dim))
                except Exception:
                    continue

            if not eligible:
                return ""

            # Pick the strongest (highest consecutive_cycles)
            eligible.sort(key=lambda x: x[0], reverse=True)
            _, strongest_dim = eligible[0]

            variants = GROWTH_REFLECTIONS.get(strongest_dim)
            if not variants:
                return ""

            reflection = random.choice(variants)

            # Set 24-hour cooldown
            r.set(cooldown_key, '1', ex=_GROWTH_COOLDOWN_TTL)

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

    def _resolve_emotional_slot(self, style: dict) -> str:
        """
        Return an emotional/certainty directive only when salience > 1.5
        on either emotional_valence or certainty_level.

        Picks the dimension with the higher salience.
        """
        candidates: List[tuple] = []
        for dim in ('emotional_valence', 'certainty_level'):
            val = style.get(dim)
            if val is None:
                continue
            val_f = float(val)
            salience = abs(val_f - _MIDPOINT)
            if salience > 1.5:
                text = self._resolve_directive(dim, style)
                if text:
                    candidates.append((salience, text))
        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _challenge_tier(tolerance: float) -> str:
        """
        Map a 1–10 challenge tolerance value to a tier name.

          1–3  → low
          4–7  → medium
          8–10 → high
        """
        if tolerance <= 3:
            return 'low'
        if tolerance <= 7:
            return 'medium'
        return 'high'
