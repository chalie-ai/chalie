"""
User Trait Service - Per-user trait memory with confidence decay and reinforcement.

Stores learned facts about the user (name, preferences, relationships) with
confidence-based decay, reinforcement on re-observation, and contextual retrieval.

This is not a "user profiles" system. It's a trait memory with a familiarity signal.
Traits decay unless reinforced, and uncertain knowledge fades naturally.
"""

import logging
import math
import re
import struct
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ── Trait validation (deterministic, zero LLM) ────────────────────
# Catches garbage traits from weak models and noisy temporal patterns.

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'like',
    'through', 'after', 'over', 'between', 'out', 'up', 'down', 'off',
    'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'because', 'if', 'when', 'while', 'this',
    'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'how', 'where', 'there', 'here',
    'often', 'discusses', 'yes', 'no', 'ok', 'okay', 'true', 'false',
})

# Placeholder values that indicate no real trait was extracted
_PLACEHOLDER_RE = re.compile(
    r'^(?:unknown|n/?a|none|null|undefined|not specified|unspecified|empty|default)$',
    re.IGNORECASE,
)

# Maximum segments in a topic_time key slug before it's likely a sentence fragment.
# Real topics: "machine_learning" (2), "natural_language_processing" (3).
# Garbage: "bloody_tired_trying_finalise" (4), "yeah_look_skill_itself" (4).
_MAX_TOPIC_SLUG_SEGMENTS = 3


def _extract_content_words(text: str) -> list:
    """Extract meaningful content words from text (not stop words, len > 1)."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _information_density(text: str) -> float:
    """Ratio of content words to total words. Higher = more meaningful."""
    all_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    if not all_words:
        return 0.0
    content = [w for w in all_words if w not in _STOP_WORDS and len(w) > 1]
    return len(content) / len(all_words)


def _validate_trait(key: str, value: str, category: str) -> Optional[str]:
    """Validate a trait before storage. Returns rejection reason or None if valid.

    Uses information density and structural heuristics rather than hardcoded
    template matching. Catches garbage from weak LLMs and noisy temporal patterns.
    """
    # Key validation
    if not key or len(key) < 2:
        return 'key_too_short'

    # Reject keys that are pure stop words
    key_segments = [w for w in re.split(r'[_\-\s]+', key.lower()) if w]
    content_key = [w for w in key_segments if w not in _STOP_WORDS and len(w) > 1]
    if not content_key:
        return 'key_is_stop_words'

    # Value validation
    if not value or len(value.strip()) < 3:
        return 'value_too_short'
    if len(value) > 500:
        return 'value_too_long'

    # Placeholder detection
    if _PLACEHOLDER_RE.match(value.strip()):
        return 'placeholder_value'

    # Information density: value must carry enough meaning
    # Low density = mostly stop/template words, little actual content
    density = _information_density(value)
    content_words_value = _extract_content_words(value)
    if len(content_words_value) < 1:
        return 'no_content_words'

    # Combined content: key + value must have ≥ 2 unique content words
    content_all = set(content_key + content_words_value)
    if len(content_all) < 2:
        return 'insufficient_content'

    # Topic-time trait validation: the topic slug must be a real concept, not a
    # slugified sentence fragment. Real topics have ≤ 3-4 slug segments
    # ("machine_learning", "system_design"). Sentence fragments have more
    # ("bloody_tired_trying_finalise", "yeah_look_skill_itself").
    if key.startswith('topic_time_'):
        topic_slug = key[len('topic_time_'):]
        slug_segments = [s for s in topic_slug.split('_') if s]

        if len(slug_segments) > _MAX_TOPIC_SLUG_SEGMENTS:
            return 'topic_slug_too_long'

        # Topic slug must have at least one content word (not a stop word)
        topic_content = [s for s in slug_segments if s not in _STOP_WORDS and len(s) > 1]
        if not topic_content:
            return 'topic_slug_no_content'

    return None

# Confidence levels for natural language injection
CONFIDENCE_LABELS = {
    'high': '(well established)',
    'medium': '(likely)',
    'low': '(uncertain)',
}

# Category-specific decay configuration
# base_decay: per decay cycle (30min), floor: minimum confidence before deletion eligibility
CATEGORY_DECAY = {
    'core':       {'base_decay': 0.01, 'floor': 0.25},
    'preference': {'base_decay': 0.02, 'floor': 0.10},
    'behavioral': {'base_decay': 0.005, 'floor': 0.20},
}

MAX_TRAITS_IN_PROMPT = 8
WILDCARD_SLOTS = 2              # High-confidence identity traits regardless of semantic match
WILDCARD_CONFIDENCE = 0.7       # Minimum confidence to qualify as a wildcard
SEMANTIC_RETRIEVAL_K = 25       # Wide retrieval window for fuzzy cross-domain connections
INJECTION_THRESHOLD = 0.3


def _nudge_uncertainty_tolerance(db_service, direction: float) -> None:
    """
    Phase 4: Nudge Chalie's uncertainty_tolerance identity dimension.

    direction > 0 → higher tolerance (surface less, resolve more silently)
    direction < 0 → lower tolerance (surface sooner, clarify more)
    """
    try:
        from services.identity_service import IdentityService
        identity = IdentityService(db_service)
        # Use emotion_signal=direction (identity reads this as a positive/negative nudge)
        identity.update_activation(
            vector_name='uncertainty_tolerance',
            emotion_signal=direction,
            reward_signal=direction * 0.5,
            topic='self_calibration',
        )
    except Exception:
        pass  # Identity vector may not exist yet — safe to skip


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list/tuple of floats into a binary blob for sqlite-vec."""
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, (list, tuple)):
        return struct.pack(f'{len(embedding)}f', *embedding)
    # numpy arrays
    if hasattr(embedding, 'tolist'):
        flat = embedding.tolist()
        return struct.pack(f'{len(flat)}f', *flat)
    return embedding


class UserTraitService:
    """Manages user trait storage, retrieval, decay, and prompt injection."""

    def __init__(self, database_service):
        self.db = database_service

    def _store_embedding(self, conn, trait_rowid: int, embedding_blob: Optional[bytes]):
        """Store embedding in the user_traits_vec companion virtual table."""
        if embedding_blob is None:
            return
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO user_traits_vec(rowid, embedding) VALUES (?, ?)",
                (trait_rowid, embedding_blob)
            )
            cursor.close()
        except Exception as e:
            logger.warning(f"[USER_TRAITS] Failed to store trait embedding: {e}")

    def _get_rowid(self, conn, trait_key: str) -> Optional[int]:
        """Get the rowid for a trait by trait_key."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rowid FROM user_traits WHERE trait_key = ?",
            (trait_key,)
        )
        row = cursor.fetchone()
        cursor.close()
        return row[0] if row else None

    def store_trait(
        self,
        trait_key: str,
        trait_value: str,
        confidence: float,
        category: str = 'preference',
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
            category: One of: core, preference, behavioral

        Returns:
            bool: True if stored/updated, False if rejected
        """
        # Validate trait quality before any DB operations
        rejection = _validate_trait(trait_key, trait_value, category)
        if rejection:
            logger.info(
                f"[USER_TRAITS] Rejected '{trait_key}': {rejection} "
                f"(value='{trait_value[:60]}', category={category})"
            )
            return False

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Check for existing trait
                cursor.execute("""
                    SELECT trait_value, confidence, reinforcement_count
                    FROM user_traits
                    WHERE trait_key = ?
                """, (trait_key,))
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
                            SET confidence = ?,
                                reinforcement_count = ?,
                                last_reinforced_at = datetime('now'),
                                updated_at = datetime('now')
                            WHERE trait_key = ?
                        """, (new_confidence, new_count, trait_key))

                        logger.info(
                            f"[USER_TRAITS] Reinforced '{trait_key}': "
                            f"confidence {old_confidence:.2f} → {new_confidence:.2f} "
                            f"(count: {new_count})"
                        )

                        # Point A — Reinforcement signal
                        try:
                            from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                            emit_reasoning_signal(ReasoningSignal(
                                signal_type='trait_changed',
                                source='user_trait_service',
                                topic=category or 'general',
                                content=f"Reinforced '{trait_key}' = '{trait_value}' (confidence={new_confidence:.2f})",
                                activation_energy=0.3,
                            ))
                        except Exception:
                            pass

                        # Phase 4 — evidence-based resolution: reinforcement resolves uncertainties
                        # on this trait if the other side is significantly weaker
                        cursor.execute("SELECT id FROM user_traits WHERE trait_key = ?", (trait_key,))
                        rein_id_row = cursor.fetchone()
                        if rein_id_row:
                            try:
                                from services.uncertainty_service import UncertaintyService
                                UncertaintyService(self.db).resolve_by_reinforcement(
                                    memory_type='trait',
                                    memory_id=rein_id_row[0],
                                    reinforced_confidence=new_confidence,
                                )
                            except Exception as ue:
                                logger.debug(f"[USER_TRAITS] Evidence resolution skipped: {ue}")
                    elif confidence > old_confidence * 2:
                        # Conflict with significantly higher confidence → overwrite
                        # Fetch the old trait's ID before overwriting for uncertainty tracking
                        cursor.execute("SELECT id FROM user_traits WHERE trait_key = ?", (trait_key,))
                        old_id_row = cursor.fetchone()
                        old_trait_id = old_id_row[0] if old_id_row else None

                        embedding_blob = self._generate_embedding_blob(trait_key, trait_value)
                        cursor.execute("""
                            UPDATE user_traits
                            SET trait_value = ?, confidence = ?,
                                category = ?,
                                reinforcement_count = 1,
                                last_reinforced_at = datetime('now'),
                                last_conflict_at = datetime('now'),
                                updated_at = datetime('now')
                            WHERE trait_key = ?
                        """, (trait_value, confidence, category, trait_key))

                        # Update embedding in companion vec table
                        trait_rowid = self._get_rowid(conn, trait_key)
                        if trait_rowid is not None:
                            self._store_embedding(conn, trait_rowid, embedding_blob)

                        logger.info(
                            f"[USER_TRAITS] Overwritten '{trait_key}': "
                            f"'{old_value}' → '{trait_value}' "
                            f"(confidence {old_confidence:.2f} → {confidence:.2f})"
                        )

                        # Point B — Overwrite signal
                        try:
                            from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                            emit_reasoning_signal(ReasoningSignal(
                                signal_type='trait_changed',
                                source='user_trait_service',
                                topic=category or 'general',
                                content=f"Changed '{trait_key}': '{old_value}' → '{trait_value}' (confidence={confidence:.2f})",
                                activation_energy=0.6,
                            ))
                        except Exception:
                            pass

                        # Create uncertainty for the conflict (confidence dominance path)
                        if old_trait_id:
                            try:
                                from services.uncertainty_service import UncertaintyService
                                unc_svc = UncertaintyService(self.db)
                                unc_svc.create_uncertainty(
                                    memory_a_type='trait',
                                    memory_a_id=old_trait_id,
                                    memory_b_type=None,
                                    memory_b_id=None,
                                    uncertainty_type='contradiction',
                                    detection_context='ingestion',
                                    reasoning=(
                                        f"Trait '{trait_key}' overwritten: "
                                        f"'{old_value}' → '{trait_value}' "
                                        f"(confidence dominance: {confidence:.2f} > 2x {old_confidence:.2f})"
                                    ),
                                )
                            except Exception as ue:
                                logger.debug(f"[USER_TRAITS] Uncertainty creation skipped: {ue}")
                    else:
                        # Conflict but not strong enough to overwrite — create uncertainty record
                        cursor.execute("SELECT id FROM user_traits WHERE trait_key = ?", (trait_key,))
                        conflict_id_row = cursor.fetchone()
                        conflict_trait_id = conflict_id_row[0] if conflict_id_row else None

                        cursor.execute("""
                            UPDATE user_traits
                            SET last_conflict_at = datetime('now'), updated_at = datetime('now')
                            WHERE trait_key = ?
                        """, (trait_key,))

                        logger.debug(
                            f"[USER_TRAITS] Conflict on '{trait_key}': "
                            f"'{trait_value}' vs existing '{old_value}' "
                            f"(new conf {confidence:.2f} <= 2x old {old_confidence:.2f})"
                        )

                        # Create uncertainty record (replaces silent timestamp update)
                        if conflict_trait_id:
                            try:
                                from services.uncertainty_service import UncertaintyService
                                unc_svc = UncertaintyService(self.db)
                                # Only create if not already tracked
                                existing = unc_svc.get_uncertainties_for_memory('trait', conflict_trait_id)
                                open_contradictions = [
                                    u for u in existing
                                    if u.get('uncertainty_type') == 'contradiction'
                                    and u.get('state') in ('open', 'surfaced')
                                ]
                                if not open_contradictions:
                                    unc_svc.create_uncertainty(
                                        memory_a_type='trait',
                                        memory_a_id=conflict_trait_id,
                                        memory_b_type=None,
                                        memory_b_id=None,
                                        uncertainty_type='contradiction',
                                        detection_context='ingestion',
                                        reasoning=(
                                            f"Trait '{trait_key}': new value '{trait_value}' "
                                            f"conflicts with existing '{old_value}' "
                                            f"(insufficient confidence to overwrite)"
                                        ),
                                    )
                            except Exception as ue:
                                logger.debug(f"[USER_TRAITS] Uncertainty creation skipped: {ue}")
                else:
                    # New trait
                    trait_id = str(uuid.uuid4())
                    embedding_blob = self._generate_embedding_blob(trait_key, trait_value)
                    cursor.execute("""
                        INSERT INTO user_traits
                            (id, trait_key, trait_value, category, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (trait_id, trait_key, trait_value, category, confidence))

                    # Store embedding in companion vec table
                    trait_rowid = self._get_rowid(conn, trait_key)
                    if trait_rowid is not None:
                        self._store_embedding(conn, trait_rowid, embedding_blob)

                    logger.info(
                        f"[USER_TRAITS] Stored new trait '{trait_key}': "
                        f"'{trait_value}' (confidence: {confidence:.2f}, "
                        f"category: {category})"
                    )

                    # Point C — New trait signal
                    try:
                        from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                        emit_reasoning_signal(ReasoningSignal(
                            signal_type='trait_changed',
                            source='user_trait_service',
                            topic=category or 'general',
                            content=f"New trait '{trait_key}' = '{trait_value}' (confidence={confidence:.2f})",
                            activation_energy=0.5,
                        ))
                    except Exception:
                        pass

                cursor.close()
                return True

        except Exception as e:
            logger.error(f"[USER_TRAITS] Failed to store trait '{trait_key}': {e}")
            return False

    def get_traits_for_prompt(
        self,
        prompt: str = "",
        injection_threshold: float = INJECTION_THRESHOLD,
    ) -> str:
        """
        Get user traits formatted for prompt injection.

        Three-tier retrieval:
        1. Core traits (name, etc.) — always injected
        2. Semantic matches — KNN against current message embedding (wide k for
           cross-domain fuzziness: "Docker" can surface "boxer" for metaphor)
        3. Identity wildcards — highest-confidence non-core traits regardless of
           semantic match, giving the LLM creative personality material

        Hard cap: MAX_TRAITS_IN_PROMPT total.

        Confidence rendered as natural language:
        - > 0.7 → (well established)
        - 0.4 - 0.7 → (likely)
        - 0.3 - 0.4 → (uncertain)

        Args:
            prompt: Current user message for contextual retrieval
            injection_threshold: Minimum confidence for inclusion

        Returns:
            str: Formatted traits section or empty string
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # ── Tier 1: Core traits (always present) ──────────────────
                cursor.execute("""
                    SELECT trait_key, trait_value, confidence, category
                    FROM user_traits
                    WHERE category = 'core'
                      AND confidence > ?
                    ORDER BY confidence DESC
                """, (injection_threshold,))
                core_traits = cursor.fetchall()

                remaining_slots = MAX_TRAITS_IN_PROMPT - len(core_traits)
                core_keys = {r[0] for r in core_traits}
                semantic_traits = []
                wildcard_traits = []

                if remaining_slots > 0 and prompt:
                    # ── Tier 2: Semantic matches (wide retrieval) ─────────
                    prompt_embedding = self._generate_embedding_raw(prompt)
                    if prompt_embedding:
                        packed = _pack_embedding(prompt_embedding)
                        if packed:
                            cursor.execute("""
                                SELECT t.trait_key, t.trait_value, t.confidence, t.category
                                FROM user_traits_vec v
                                JOIN user_traits t ON t.rowid = v.rowid
                                WHERE v.embedding MATCH ? AND k = ?
                                ORDER BY v.distance
                            """, (packed, SEMANTIC_RETRIEVAL_K))
                            all_vec_results = cursor.fetchall()

                            semantic_traits = [
                                r for r in all_vec_results
                                if r[3] != 'core'
                                   and r[0] not in core_keys
                                   and r[2] > injection_threshold
                            ]

                    # ── Tier 3: Identity wildcards ────────────────────────
                    # Highest-confidence non-core traits — the user's most
                    # defining characteristics, always accessible for creative
                    # analogies and personalisation even when not semantically
                    # close to the current topic.
                    semantic_keys = {r[0] for r in semantic_traits}
                    cursor.execute("""
                        SELECT trait_key, trait_value, confidence, category
                        FROM user_traits
                        WHERE category != 'core'
                          AND confidence >= ?
                        ORDER BY confidence DESC, reinforcement_count DESC
                        LIMIT ?
                    """, (WILDCARD_CONFIDENCE, WILDCARD_SLOTS + len(semantic_keys) + len(core_keys)))
                    wildcard_candidates = cursor.fetchall()

                    wildcard_traits = [
                        r for r in wildcard_candidates
                        if r[0] not in core_keys and r[0] not in semantic_keys
                    ][:WILDCARD_SLOTS]

                    # Allocate: semantic fills remaining after wildcards
                    semantic_cap = remaining_slots - len(wildcard_traits)
                    semantic_traits = semantic_traits[:max(0, semantic_cap)]

                elif remaining_slots > 0:
                    # No prompt for context — get highest confidence non-core traits
                    cursor.execute("""
                        SELECT trait_key, trait_value, confidence, category
                        FROM user_traits
                        WHERE category != 'core'
                          AND confidence > ?
                        ORDER BY confidence DESC
                        LIMIT ?
                    """, (injection_threshold, remaining_slots))
                    semantic_traits = cursor.fetchall()

                cursor.close()

                all_traits = list(core_traits) + list(semantic_traits) + list(wildcard_traits)
                if not all_traits:
                    return ""

                lines = ["## Known About User"]
                for trait_key, trait_value, confidence, category in all_traits:
                    if confidence <= injection_threshold:
                        continue
                    label = self._confidence_label(confidence)
                    display_key = trait_key.replace('_', ' ').title()
                    lines.append(f"- {display_key}: {trait_value} {label}")

                if len(lines) <= 1:
                    return ""

                return "\n".join(lines)

        except Exception as e:
            logger.debug(f"[USER_TRAITS] Failed to get traits for prompt: {e}")
            return ""

    def get_communication_style(self, threshold: float = 0.3) -> dict:
        """
        Get the user's detected communication style dimensions.

        Queries the latest communication_style trait (stored as JSON value),
        returning a dict of dimension → score.

        Args:
            threshold: Minimum confidence to include

        Returns:
            dict with keys: verbosity, directness, formality, abstraction_level (1-10 scale)
            Empty dict if no style detected.
        """
        import json as _json
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trait_value, confidence
                    FROM user_traits
                    WHERE trait_key = 'communication_style'
                      AND category = 'communication_style'
                      AND confidence > ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (threshold,))
                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return {}

                value_str, confidence = row
                try:
                    return _json.loads(value_str)
                except Exception:
                    return {}
        except Exception as e:
            logger.debug(f"[USER_TRAITS] get_communication_style failed: {e}")
            return {}

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
                    SELECT id, trait_key, confidence, category,
                           reinforcement_count, updated_at,
                           COALESCE(reliability, 'reliable') AS reliability
                    FROM user_traits
                """)
                rows = cursor.fetchall()

                for row in rows:
                    trait_id, trait_key, confidence, category, reinf_count, updated_at, reliability = row

                    decay_cfg = CATEGORY_DECAY.get(category, CATEGORY_DECAY['preference'])
                    base_decay = decay_cfg['base_decay']
                    floor = decay_cfg['floor']

                    # Reinforcement resistance
                    resistance = 1.0 / math.log2(max(reinf_count, 1) + 1)
                    effective_decay = base_decay * resistance

                    # Reliability multiplier: contradicted/uncertain traits decay faster
                    from services.decay_engine_service import RELIABILITY_DECAY_MULTIPLIER
                    effective_decay *= RELIABILITY_DECAY_MULTIPLIER.get(reliability, 1.0)

                    new_confidence = max(floor, confidence - effective_decay)

                    if abs(new_confidence - confidence) > 0.001:
                        cursor.execute("""
                            UPDATE user_traits
                            SET confidence = ?, updated_at = datetime('now')
                            WHERE id = ?
                        """, (new_confidence, trait_id))
                        stats['decayed'] += 1

                    # Delete traits below floor for 7+ days
                    if new_confidence <= floor:
                        cursor.execute("""
                            DELETE FROM user_traits
                            WHERE id = ?
                              AND confidence <= ?
                              AND updated_at < datetime('now', '-7 days')
                        """, (trait_id, floor + 0.01))
                        if cursor.rowcount > 0:
                            stats['deleted'] += 1
                            logger.info(
                                f"[USER_TRAITS] Deleted stale trait '{trait_key}' "
                                f"(below floor for 7+ days)"
                            )
                            # Resolve any open uncertainties tied to this deleted trait
                            try:
                                from services.uncertainty_service import UncertaintyService
                                UncertaintyService(self.db).resolve_decayed('trait', trait_id)
                            except Exception as ue:
                                logger.debug(f"[USER_TRAITS] resolve_decayed skipped: {ue}")

                cursor.close()

                if stats['decayed'] > 0 or stats['deleted'] > 0:
                    logger.info(
                        f"[USER_TRAITS] Decay cycle: "
                        f"{stats['decayed']} decayed, {stats['deleted']} deleted"
                    )

        except Exception as e:
            logger.error(f"[USER_TRAITS] Decay failed: {e}")

        return stats

    def correct_trait(self, trait_key: str, new_value: str, category: str = None) -> bool:
        """
        Explicit user correction — overwrites regardless of confidence threshold.
        This is the conversational path for "that's wrong about me."

        Bypasses the >2x threshold by design — user's explicit word always wins.
        Sets confidence to 0.95 and source to 'explicit_correction' for audit trail.
        """
        try:
            embedding_blob = self._generate_embedding_blob(trait_key, new_value)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, category FROM user_traits WHERE trait_key = ?",
                    (trait_key,)
                )
                existing = cursor.fetchone()

                if existing:
                    existing_trait_id = existing[0]
                    # Overwrite with high confidence, reset reinforcement, preserve category
                    update_category = category or existing[1]
                    cursor.execute(
                        "UPDATE user_traits SET trait_value = ?, confidence = ?, "
                        "reinforcement_count = 1, "
                        "category = ?, "
                        "last_conflict_at = datetime('now'), updated_at = datetime('now') "
                        "WHERE trait_key = ?",
                        (new_value, 0.95, update_category, trait_key)
                    )

                    # Update embedding in companion vec table
                    trait_rowid = self._get_rowid(conn, trait_key)
                    if trait_rowid is not None:
                        self._store_embedding(conn, trait_rowid, embedding_blob)

                    logger.info(f"[USER_TRAITS] Corrected trait '{trait_key}' → '{new_value}' (explicit_correction)")

                    # Point D — Existing trait corrected signal
                    try:
                        from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                        emit_reasoning_signal(ReasoningSignal(
                            signal_type='trait_changed',
                            source='user_trait_service',
                            topic=update_category or 'general',
                            content=f"User corrected '{trait_key}' → '{new_value}'",
                            activation_energy=0.7,
                        ))
                    except Exception:
                        pass

                    # Phase 3 — Resolution feedback loop: resolve linked uncertainties
                    # and Phase 4 — lower uncertainty tolerance (user correcting us)
                    try:
                        from services.uncertainty_service import UncertaintyService
                        unc_svc = UncertaintyService(self.db)
                        open_uncs = unc_svc.get_uncertainties_for_memory('trait', existing_trait_id)
                        for unc in open_uncs:
                            unc_svc.resolve_uncertainty(
                                uncertainty_id=unc['id'],
                                strategy='user_clarified',
                                detail=f"User explicitly corrected '{trait_key}' → '{new_value}'",
                                winner_type='trait',
                                winner_id=existing_trait_id,
                                loser_type=unc.get('memory_b_type'),
                                loser_id=unc.get('memory_b_id'),
                            )
                        # Lower uncertainty tolerance in identity (user corrects = wants accuracy)
                        _nudge_uncertainty_tolerance(self.db, direction=-0.03)
                    except Exception as ue:
                        logger.debug(f"[USER_TRAITS] Post-correction resolution skipped: {ue}")
                else:
                    # New trait from explicit correction
                    trait_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO user_traits (id, trait_key, trait_value, confidence, "
                        "category, reinforcement_count) "
                        "VALUES (?, ?, ?, ?, ?, 1)",
                        (trait_id, trait_key, new_value, 0.95, category or 'preference')
                    )

                    # Store embedding in companion vec table
                    trait_rowid = self._get_rowid(conn, trait_key)
                    if trait_rowid is not None:
                        self._store_embedding(conn, trait_rowid, embedding_blob)

                    logger.info(f"[USER_TRAITS] Inserted corrected trait '{trait_key}' = '{new_value}'")

                    # Point E — New trait from correction signal
                    try:
                        from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                        emit_reasoning_signal(ReasoningSignal(
                            signal_type='trait_changed',
                            source='user_trait_service',
                            topic=category if category else 'general',
                            content=f"User set new trait '{trait_key}' = '{new_value}'",
                            activation_energy=0.7,
                        ))
                    except Exception:
                        pass
            return True
        except Exception as e:
            logger.error(f"[TRAITS] Correction failed for {trait_key}: {e}")
            return False

    def delete_trait(self, trait_key: str) -> bool:
        """Delete a trait by key. Used for explicit negations ('I don't like X')."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                # Fetch id before deleting (needed for uncertainty resolution)
                cursor.execute(
                    "SELECT id, category FROM user_traits WHERE trait_key = ?",
                    (trait_key,)
                )
                existing = cursor.fetchone()
                if not existing:
                    return False

                trait_id, category = existing[0], existing[1]

                cursor.execute(
                    "DELETE FROM user_traits WHERE trait_key = ?",
                    (trait_key,)
                )

            logger.info(f"[USER_TRAITS] Deleted trait '{trait_key}' (user negation)")

            # Emit reasoning signal — deletion is a correction event
            try:
                from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
                emit_reasoning_signal(ReasoningSignal(
                    signal_type='trait_changed',
                    source='user_trait_service',
                    topic=category or 'general',
                    content=f"User deleted trait '{trait_key}'",
                    activation_energy=0.7,
                ))
            except Exception:
                pass

            # Phase 4 — lower uncertainty tolerance (user deleting = wants accuracy)
            try:
                _nudge_uncertainty_tolerance(self.db, direction=-0.03)
            except Exception as ue:
                logger.debug(f"[USER_TRAITS] Post-delete tolerance nudge skipped: {ue}")

            return True
        except Exception as e:
            logger.error(f"[TRAITS] Delete failed for {trait_key}: {e}")
            return False

    def get_all_traits(self) -> list:
        """Return all stored traits."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trait_key, trait_value, confidence, category "
                    "FROM user_traits"
                )
                rows = cursor.fetchall()
                return [
                    {'trait_key': r[0], 'trait_value': r[1], 'confidence': r[2], 'category': r[3]}
                    for r in rows
                ]
        except Exception as e:
            logger.debug(f"[USER_TRAITS] get_all_traits failed: {e}")
            return []

    def get_traits_by_category(self, category: str) -> list:
        """Return all traits matching a specific category."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trait_key, trait_value, confidence, category "
                    "FROM user_traits WHERE category = ?",
                    (category,)
                )
                rows = cursor.fetchall()
                return [
                    {'trait_key': r[0], 'trait_value': r[1], 'confidence': r[2], 'category': r[3]}
                    for r in rows
                ]
        except Exception as e:
            logger.debug(f"[USER_TRAITS] get_traits_by_category failed: {e}")
            return []

    def _confidence_label(self, confidence: float) -> str:
        """Convert numeric confidence to natural language label."""
        if confidence > 0.7:
            return CONFIDENCE_LABELS['high']
        elif confidence >= 0.4:
            return CONFIDENCE_LABELS['medium']
        else:
            return CONFIDENCE_LABELS['low']

    def _generate_embedding(self, trait_key: str, trait_value: str) -> Optional[str]:
        """Generate embedding for a trait (key + value combined). Returns JSON-serialized vector string."""
        raw = self._generate_embedding_raw(f"{trait_key}: {trait_value}")
        return raw

    def _generate_embedding_blob(self, trait_key: str, trait_value: str) -> Optional[bytes]:
        """Generate embedding as packed binary blob for sqlite-vec storage."""
        raw = self._generate_embedding_raw(f"{trait_key}: {trait_value}")
        if raw is None:
            return None
        return _pack_embedding(raw)

    def _generate_embedding_raw(self, text: str) -> Optional[list]:
        """Generate embedding vector as a list of floats."""
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            embedding = emb_service.generate_embedding(text)
            # Return as list of floats (not string)
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            return list(embedding)
        except Exception as e:
            logger.debug(f"[USER_TRAITS] Embedding generation failed: {e}")
            return None
