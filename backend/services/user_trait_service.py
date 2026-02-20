"""
User Trait Service - Per-user trait memory with confidence decay and reinforcement.

Stores learned facts about the user (name, preferences, relationships) with
confidence-based decay, reinforcement on re-observation, and contextual retrieval.

This is not a "user profiles" system. It's a trait memory with a familiarity signal.
Traits decay unless reinforced, and uncertain knowledge fades naturally.
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence levels for natural language injection
CONFIDENCE_LABELS = {
    'high': '(well established)',
    'medium': '(likely)',
    'low': '(uncertain)',
}

# Category-specific decay configuration
# base_decay: per decay cycle (30min), floor: minimum confidence before deletion eligibility
CATEGORY_DECAY = {
    'core':         {'base_decay': 0.01, 'floor': 0.3},
    'relationship': {'base_decay': 0.01, 'floor': 0.25},
    'physical':     {'base_decay': 0.015, 'floor': 0.2},
    'preference':   {'base_decay': 0.02, 'floor': 0.1},
    'general':      {'base_decay': 0.02, 'floor': 0.1},
}

MAX_TRAITS_IN_PROMPT = 6
INJECTION_THRESHOLD = 0.3


class UserTraitService:
    """Manages user trait storage, retrieval, decay, and prompt injection."""

    def __init__(self, database_service):
        self.db = database_service

    def store_trait(
        self,
        trait_key: str,
        trait_value: str,
        confidence: float,
        category: str = 'general',
        source: str = 'inferred',
        is_literal: bool = True,
        user_id: str = 'primary',
        speaker_confidence: float = 1.0,
    ) -> bool:
        """
        Store or update a user trait with conflict resolution.

        On extraction, check for existing trait with same key:
        - Same value observed again → reinforce (increment count, refresh timestamp, boost confidence)
        - Different value, new confidence significantly higher (>2x) → overwrite
        - Different value, not significantly higher → record conflict timestamp only

        Args:
            trait_key: Trait identifier (e.g., 'name', 'favourite_food')
            trait_value: Trait value (e.g., 'Dylan', 'ramen')
            confidence: Confidence score (0.0-1.0)
            category: One of: core, preference, physical, relationship, general
            source: 'explicit' or 'inferred'
            is_literal: False for humor/figurative statements
            user_id: User identifier (default 'primary')
            speaker_confidence: How confident we are this is the known user (0.0-1.0)

        Returns:
            bool: True if stored/updated, False if rejected
        """
        # Reject traits from untrusted sources
        if speaker_confidence < 0.3:
            logger.debug(f"[USER_TRAITS] Rejecting trait '{trait_key}' from untrusted speaker")
            return False

        # Apply speaker confidence penalty
        if speaker_confidence < 0.5:
            confidence *= 0.5

        # Apply inferred source penalty
        if source == 'inferred':
            confidence *= 0.7

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Check for existing trait
                cursor.execute("""
                    SELECT trait_value, confidence, reinforcement_count
                    FROM user_traits
                    WHERE user_id = %s AND trait_key = %s
                """, (user_id, trait_key))
                existing = cursor.fetchone()

                if existing:
                    old_value, old_confidence, old_count = existing

                    if old_value.lower().strip() == trait_value.lower().strip():
                        # Reinforcement: same value observed again
                        new_count = old_count + 1
                        # Boost confidence slightly (diminishing returns)
                        boost = 0.05 / math.log2(new_count + 1)
                        new_confidence = min(1.0, max(old_confidence, confidence) + boost)

                        cursor.execute("""
                            UPDATE user_traits
                            SET confidence = %s,
                                reinforcement_count = %s,
                                last_reinforced_at = NOW(),
                                updated_at = NOW()
                            WHERE user_id = %s AND trait_key = %s
                        """, (new_confidence, new_count, user_id, trait_key))

                        logger.info(
                            f"[USER_TRAITS] Reinforced '{trait_key}': "
                            f"confidence {old_confidence:.2f} → {new_confidence:.2f} "
                            f"(count: {new_count})"
                        )
                    elif confidence > old_confidence * 2:
                        # Conflict with significantly higher confidence → overwrite
                        embedding = self._generate_embedding(trait_key, trait_value)
                        cursor.execute("""
                            UPDATE user_traits
                            SET trait_value = %s, confidence = %s,
                                category = %s, source = %s, is_literal = %s,
                                reinforcement_count = 1,
                                last_reinforced_at = NOW(),
                                last_conflict_at = NOW(),
                                embedding = %s,
                                updated_at = NOW()
                            WHERE user_id = %s AND trait_key = %s
                        """, (trait_value, confidence, category, source,
                              is_literal, embedding, user_id, trait_key))

                        logger.info(
                            f"[USER_TRAITS] Overwritten '{trait_key}': "
                            f"'{old_value}' → '{trait_value}' "
                            f"(confidence {old_confidence:.2f} → {confidence:.2f})"
                        )
                    else:
                        # Conflict but not strong enough to overwrite — record conflict
                        cursor.execute("""
                            UPDATE user_traits
                            SET last_conflict_at = NOW(), updated_at = NOW()
                            WHERE user_id = %s AND trait_key = %s
                        """, (user_id, trait_key))

                        logger.debug(
                            f"[USER_TRAITS] Conflict on '{trait_key}': "
                            f"'{trait_value}' vs existing '{old_value}' "
                            f"(new conf {confidence:.2f} <= 2x old {old_confidence:.2f})"
                        )
                else:
                    # New trait
                    embedding = self._generate_embedding(trait_key, trait_value)
                    cursor.execute("""
                        INSERT INTO user_traits
                            (user_id, trait_key, trait_value, category, confidence,
                             source, is_literal, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, trait_key, trait_value, category, confidence,
                          source, is_literal, embedding))

                    logger.info(
                        f"[USER_TRAITS] Stored new trait '{trait_key}': "
                        f"'{trait_value}' (confidence: {confidence:.2f}, "
                        f"category: {category}, source: {source})"
                    )

                cursor.close()
                return True

        except Exception as e:
            logger.error(f"[USER_TRAITS] Failed to store trait '{trait_key}': {e}")
            return False

    def get_traits_for_prompt(
        self,
        prompt: str = "",
        user_id: str = 'primary',
        speaker_confidence: float = 1.0,
    ) -> str:
        """
        Get user traits formatted for prompt injection.

        Always injects core traits (confidence > threshold, is_literal=true).
        Contextually injects relevant traits by embedding similarity.
        Hard cap: MAX_TRAITS_IN_PROMPT (6) total.

        Confidence rendered as natural language:
        - > 0.7 → (well established)
        - 0.4 - 0.7 → (likely)
        - 0.3 - 0.4 → (uncertain)

        Args:
            prompt: Current user message for contextual retrieval
            user_id: User identifier
            speaker_confidence: Confidence this is the known user

        Returns:
            str: Formatted traits section or empty string
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Always get core traits above injection threshold
                cursor.execute("""
                    SELECT trait_key, trait_value, confidence, category
                    FROM user_traits
                    WHERE user_id = %s
                      AND category = 'core'
                      AND confidence > %s
                      AND is_literal = true
                    ORDER BY confidence DESC
                """, (user_id, INJECTION_THRESHOLD))
                core_traits = cursor.fetchall()

                remaining_slots = MAX_TRAITS_IN_PROMPT - len(core_traits)
                contextual_traits = []

                if remaining_slots > 0 and prompt:
                    # Contextual retrieval by embedding similarity
                    prompt_embedding = self._generate_embedding_raw(prompt)
                    if prompt_embedding:
                        cursor.execute("""
                            SELECT trait_key, trait_value, confidence, category
                            FROM user_traits
                            WHERE user_id = %s
                              AND category != 'core'
                              AND confidence > %s
                              AND is_literal = true
                              AND embedding IS NOT NULL
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (user_id, INJECTION_THRESHOLD, prompt_embedding, remaining_slots))
                        contextual_traits = cursor.fetchall()

                elif remaining_slots > 0:
                    # No prompt for context — get highest confidence non-core traits
                    cursor.execute("""
                        SELECT trait_key, trait_value, confidence, category
                        FROM user_traits
                        WHERE user_id = %s
                          AND category != 'core'
                          AND confidence > %s
                          AND is_literal = true
                        ORDER BY confidence DESC
                        LIMIT %s
                    """, (user_id, INJECTION_THRESHOLD, remaining_slots))
                    contextual_traits = cursor.fetchall()

                cursor.close()

                all_traits = list(core_traits) + list(contextual_traits)
                if not all_traits:
                    return ""

                # Scale confidence by speaker_confidence for unknown speakers
                lines = ["## Known About User"]
                for trait_key, trait_value, confidence, category in all_traits:
                    effective_confidence = confidence * speaker_confidence
                    if effective_confidence <= INJECTION_THRESHOLD:
                        continue
                    label = self._confidence_label(effective_confidence)
                    # Title-case the key for readability
                    display_key = trait_key.replace('_', ' ').title()
                    lines.append(f"- {display_key}: {trait_value} {label}")

                if len(lines) <= 1:
                    return ""

                return "\n".join(lines)

        except Exception as e:
            logger.debug(f"[USER_TRAITS] Failed to get traits for prompt: {e}")
            return ""

    def apply_decay(self) -> dict:
        """
        Apply confidence decay to all user traits.

        Decay formula:
        effective_decay = base_decay * (1 / log2(reinforcement_count + 1))
        - 1 observation: full decay
        - 3 observations: ~63% decay
        - 8 observations: ~33% decay

        Inferred traits decay 1.5x faster than explicit traits.

        Traits below floor for 7+ days get deleted.

        Returns:
            dict: {decayed: int, deleted: int}
        """
        stats = {'decayed': 0, 'deleted': 0}

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Load all traits
                cursor.execute("""
                    SELECT id, trait_key, confidence, category, source,
                           reinforcement_count, updated_at
                    FROM user_traits
                """)
                rows = cursor.fetchall()

                for row in rows:
                    trait_id, trait_key, confidence, category, source, reinf_count, updated_at = row

                    decay_cfg = CATEGORY_DECAY.get(category, CATEGORY_DECAY['general'])
                    base_decay = decay_cfg['base_decay']
                    floor = decay_cfg['floor']

                    # Reinforcement resistance
                    resistance = 1.0 / math.log2(max(reinf_count, 1) + 1)
                    effective_decay = base_decay * resistance

                    # Inferred traits decay 1.5x faster
                    if source == 'inferred':
                        effective_decay *= 1.5

                    new_confidence = max(floor, confidence - effective_decay)

                    if abs(new_confidence - confidence) > 0.001:
                        cursor.execute("""
                            UPDATE user_traits
                            SET confidence = %s, updated_at = NOW()
                            WHERE id = %s
                        """, (new_confidence, trait_id))
                        stats['decayed'] += 1

                    # Delete traits below floor for 7+ days
                    if new_confidence <= floor:
                        cursor.execute("""
                            DELETE FROM user_traits
                            WHERE id = %s
                              AND confidence <= %s
                              AND updated_at < NOW() - INTERVAL '7 days'
                        """, (trait_id, floor + 0.01))
                        if cursor.rowcount > 0:
                            stats['deleted'] += 1
                            logger.info(
                                f"[USER_TRAITS] Deleted stale trait '{trait_key}' "
                                f"(below floor for 7+ days)"
                            )

                cursor.close()

                if stats['decayed'] > 0 or stats['deleted'] > 0:
                    logger.info(
                        f"[USER_TRAITS] Decay cycle: "
                        f"{stats['decayed']} decayed, {stats['deleted']} deleted"
                    )

        except Exception as e:
            logger.error(f"[USER_TRAITS] Decay failed: {e}")

        return stats

    def get_speaker_confidence(self, metadata: dict) -> float:
        """
        Single-user app — all authenticated requests are the primary user.

        Returns:
            float: Always 1.0
        """
        return 1.0

    def _confidence_label(self, confidence: float) -> str:
        """Convert numeric confidence to natural language label."""
        if confidence > 0.7:
            return CONFIDENCE_LABELS['high']
        elif confidence >= 0.4:
            return CONFIDENCE_LABELS['medium']
        else:
            return CONFIDENCE_LABELS['low']

    def _generate_embedding(self, trait_key: str, trait_value: str) -> Optional[str]:
        """Generate embedding for a trait (key + value combined)."""
        raw = self._generate_embedding_raw(f"{trait_key}: {trait_value}")
        return raw

    def _generate_embedding_raw(self, text: str) -> Optional[str]:
        """Generate embedding vector as pgvector-compatible string."""
        try:
            from services.embedding_service import EmbeddingService
            emb_service = EmbeddingService()
            embedding = emb_service.generate_embedding(text)
            return '[' + ','.join(str(float(x)) for x in embedding) + ']'
        except Exception as e:
            logger.debug(f"[USER_TRAITS] Embedding generation failed: {e}")
            return None
