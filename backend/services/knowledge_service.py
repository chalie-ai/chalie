# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Knowledge Service — Unified knowledge store.

Replaces UserTraitService, SemanticService concept CRUD, and ProceduralMemoryService
with a single service operating on the ``knowledge`` table.  All knowledge entries
share a common schema: (kind, entity, key, value, data, decay_class, confidence,
reliability, source, evidence_count).

Retrieval uses Reciprocal Rank Fusion (RRF) across three signals: exact key match,
FTS5 full-text search, and sqlite-vec KNN.
"""

import json
import logging
import math
import re
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.topic_context import TopicContext

from services.database_service import get_shared_db_service
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

# ── Trait validation (deterministic, zero LLM) ────────────────────────
# Ported from UserTraitService — catches garbage traits from weak models.

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

_PLACEHOLDER_RE = re.compile(
    r'^(?:unknown|n/?a|none|null|undefined|not specified|unspecified|empty|default)$',
    re.IGNORECASE,
)

_MAX_TOPIC_SLUG_SEGMENTS = 3


def _extract_content_words(text: str) -> list:
    """Extract meaningful content words (not stop words, len > 1)."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _validate_trait(key: str, value: str) -> Optional[str]:
    """Validate a trait before storage. Returns rejection reason or None if valid."""
    if not key or len(key) < 2:
        return 'key_too_short'

    key_segments = [w for w in re.split(r'[_\-\s]+', key.lower()) if w]
    content_key = [w for w in key_segments if w not in _STOP_WORDS and len(w) > 1]
    if not content_key:
        return 'key_is_stop_words'

    if not value or len(value.strip()) < 3:
        return 'value_too_short'
    if len(value) > 500:
        return 'value_too_long'

    if _PLACEHOLDER_RE.match(value.strip()):
        return 'placeholder_value'

    content_words_value = _extract_content_words(value)
    if len(content_words_value) < 1:
        return 'no_content_words'

    content_all = set(content_key + content_words_value)
    if len(content_all) < 2:
        return 'insufficient_content'

    if key.startswith('topic_time_'):
        topic_slug = key[len('topic_time_'):]
        slug_segments = [s for s in topic_slug.split('_') if s]
        if len(slug_segments) > _MAX_TOPIC_SLUG_SEGMENTS:
            return 'topic_slug_too_long'
        topic_content = [s for s in slug_segments if s not in _STOP_WORDS and len(s) > 1]
        if not topic_content:
            return 'topic_slug_no_content'

    return None


# ── Confidence labels for prompt injection ─────────────────────────────

_CONFIDENCE_LABELS = {
    'high': '(well established)',
    'medium': '(likely)',
    'low': '(uncertain)',
}

_MAX_TRAITS_IN_PROMPT = 8
_WILDCARD_SLOTS = 2
_WILDCARD_CONFIDENCE = 0.7
_SEMANTIC_RETRIEVAL_K = 25
_INJECTION_THRESHOLD = 0.3


def _confidence_label(confidence: float) -> str:
    """Convert numeric confidence to natural language label."""
    if confidence > 0.7:
        return _CONFIDENCE_LABELS['high']
    elif confidence >= 0.4:
        return _CONFIDENCE_LABELS['medium']
    return _CONFIDENCE_LABELS['low']


class KnowledgeService:
    """Unified knowledge store — replaces UserTraitService, SemanticService concept CRUD, ProceduralMemoryService."""

    DECAY_RATES = {
        'permanent':  0.000,
        'slow':       0.002,   # ~17 days from 0.85 → floor
        'standard':   0.005,   # ~7 days from 0.85 → floor
        'fast':       0.015,   # ~2.5 days from 0.85 → floor
        'ephemeral':  0.040,   # ~20 hours from 0.85 → floor
    }

    RELIABILITY_MULTIPLIER = {
        'reliable':     1.0,
        'uncertain':    1.5,
        'contradicted': 2.0,
        'superseded':   3.0,
    }

    VALID_KINDS = {'trait', 'concept', 'fact', 'procedure', 'preference', 'relationship', 'rule', 'metric'}
    VALID_DECAY_CLASSES = {'permanent', 'slow', 'standard', 'fast', 'ephemeral'}

    # Procedural learning constants (ported from ProceduralMemoryService)
    _LEARNING_RATE = 0.1
    _DEFAULT_ACTION_WEIGHT = 1.0
    _REWARD_HISTORY_MAX = 100

    def __init__(self, db_service=None):
        self.db = db_service or get_shared_db_service()

    # ── Helpers ────────────────────────────────────────────────────────

    def _row_to_dict(self, row) -> dict:
        """Convert a sqlite3.Row to a plain dict, parsing data JSON."""
        if row is None:
            return None
        d = dict(row)
        raw_data = d.get('data')
        if raw_data and isinstance(raw_data, str):
            try:
                d['data'] = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def _generate_embedding(self, text: str):
        """Generate embedding as list of floats. Returns None on failure."""
        try:
            from services.embedding_service import get_embedding_service
            emb = get_embedding_service().generate_embedding(text)
            if hasattr(emb, 'tolist'):
                return emb.tolist()
            return list(emb)
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] Embedding generation failed: {e}")
            return None

    def _store_vec(self, conn, rowid: int, embedding):
        """Store embedding in the knowledge_vec companion table."""
        if embedding is None or rowid is None:
            return
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO knowledge_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, blob)
            )
            cursor.close()
        except Exception as e:
            logger.warning(f"[KNOWLEDGE] Failed to store vec embedding: {e}")

    def _sync_fts(self, conn, rowid: int, key: str, value: str, kind: str, entity: str):
        """Sync the FTS index for a knowledge row."""
        try:
            cursor = conn.cursor()
            # Delete old entry (if exists)
            cursor.execute(
                "DELETE FROM knowledge_fts WHERE rowid = ?",
                (rowid,)
            )
            # Insert new entry
            cursor.execute(
                "INSERT INTO knowledge_fts(rowid, key, value, kind, entity) VALUES (?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, entity)
            )
            cursor.close()
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] FTS sync failed for rowid={rowid}: {e}")

    def _remove_fts(self, conn, rowid: int):
        """Remove a row from the FTS index."""
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_fts WHERE rowid = ?", (rowid,))
            cursor.close()
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] FTS removal failed for rowid={rowid}: {e}")

    def _emit_signal(self, signal_type: str, content: str, topic: str = 'general', energy: float = 0.5):
        """Fire-and-forget reasoning signal emission."""
        try:
            from services.cognitive_drift_engine import emit_reasoning_signal, ReasoningSignal
            emit_reasoning_signal(ReasoningSignal(
                signal_type=signal_type,
                source='knowledge_service',
                topic=topic,
                content=content,
                activation_energy=energy,
            ))
        except Exception:
            pass

    # ── Core: store ────────────────────────────────────────────────────

    def store(
        self,
        kind: str,
        entity: str,
        key: str,
        value: str = None,
        data: dict = None,
        decay_class: str = 'standard',
        confidence: float = 0.5,
        reliability: str = 'reliable',
        source: str = None,
        embedding=None,
    ) -> Optional[dict]:
        """
        Store or reinforce a knowledge entry. UPSERT on (entity, key).

        On conflict (same entity+key exists):
        - Same value re-observed: reinforce (boost confidence with diminishing returns)
        - Different value AND new confidence > old * 2: overwrite
        - Otherwise: just reinforce evidence_count and timestamp

        For kind=trait: applies trait validation before storage.

        Returns the stored/updated row as dict, or None on failure.
        """
        if kind not in self.VALID_KINDS:
            logger.warning(f"[KNOWLEDGE] Invalid kind '{kind}', must be one of {self.VALID_KINDS}")
            return None
        if decay_class not in self.VALID_DECAY_CLASSES:
            decay_class = 'standard'

        # Trait validation
        if kind == 'trait' and value:
            rejection = _validate_trait(key, value)
            if rejection:
                logger.info(f"[KNOWLEDGE] Rejected trait '{key}': {rejection} (value='{(value or '')[:60]}')")
                return None

        confidence = max(0.0, min(1.0, confidence))
        data_json = json.dumps(data) if data is not None else None

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Check for existing entry
                cursor.execute("""
                    SELECT rowid, id, kind, value, data, confidence, evidence_count, reliability
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                existing = cursor.fetchone()

                if existing:
                    row_id = existing[0]
                    old_value = existing[3]
                    old_confidence = existing[5]
                    old_evidence = existing[6]

                    new_evidence = old_evidence + 1
                    # Diminishing confidence boost
                    boost = 0.05 / math.log2(new_evidence + 1)
                    new_confidence = min(1.0, max(old_confidence, confidence) + boost)

                    # Value change: only overwrite if confidence dominance
                    value_changed = False
                    if value and old_value and value.lower().strip() != old_value.lower().strip():
                        if confidence > old_confidence * 2:
                            value_changed = True
                            new_confidence = confidence
                            logger.info(
                                f"[KNOWLEDGE] Overwriting '{key}': '{old_value}' -> '{value}' "
                                f"(confidence {old_confidence:.2f} -> {confidence:.2f})"
                            )
                        else:
                            # Conflict but not strong enough — just reinforce count
                            logger.debug(
                                f"[KNOWLEDGE] Conflict on '{key}': new='{(value or '')[:40]}' "
                                f"vs existing='{(old_value or '')[:40]}' (insufficient confidence)"
                            )

                    update_value = value if value_changed else old_value
                    update_data = data_json if (data_json is not None and value_changed) else existing[4]

                    cursor.execute("""
                        UPDATE knowledge
                        SET value = ?,
                            data = COALESCE(?, data),
                            confidence = ?,
                            evidence_count = ?,
                            reliability = COALESCE(?, reliability),
                            source = COALESCE(?, source),
                            updated_at = datetime('now')
                        WHERE rowid = ?
                    """, (update_value, update_data, new_confidence, new_evidence,
                          reliability, source, row_id))

                    # Update embedding if value changed
                    if value_changed and embedding is None and value:
                        embedding = self._generate_embedding(f"{key}: {value}")
                    if value_changed or embedding:
                        self._store_vec(conn, row_id, embedding)

                    # Sync FTS if value changed
                    if value_changed:
                        self._sync_fts(conn, row_id, key, update_value, kind, entity)

                    cursor.close()

                    # Emit signal
                    if kind == 'trait':
                        self._emit_signal(
                            'trait_changed',
                            f"Reinforced '{key}' = '{update_value}' (confidence={new_confidence:.2f})",
                            energy=0.3 if not value_changed else 0.6,
                        )
                    elif kind == 'concept':
                        self._emit_signal(
                            'new_knowledge',
                            f"Reinforced concept '{key}' (confidence={new_confidence:.2f})",
                            energy=0.3,
                        )

                    return self.get(entity, key)

                else:
                    # New entry
                    if embedding is None and value:
                        embedding = self._generate_embedding(f"{key}: {value}")

                    cursor.execute("""
                        INSERT INTO knowledge (kind, entity, key, value, data, decay_class,
                                               confidence, reliability, source, evidence_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (kind, entity, key, value, data_json, decay_class,
                          confidence, reliability, source))

                    row_id = cursor.lastrowid

                    # Store embedding
                    self._store_vec(conn, row_id, embedding)

                    # Sync FTS
                    self._sync_fts(conn, row_id, key, value or '', kind, entity)

                    cursor.close()

                    logger.info(
                        f"[KNOWLEDGE] Stored new {kind} '{key}' = '{(value or '')[:60]}' "
                        f"(confidence={confidence:.2f}, decay={decay_class}, entity={entity})"
                    )

                    # Emit signal
                    if kind == 'trait':
                        self._emit_signal(
                            'trait_changed',
                            f"New trait '{key}' = '{value}' (confidence={confidence:.2f})",
                            energy=0.5,
                        )
                    elif kind == 'concept':
                        self._emit_signal(
                            'new_knowledge',
                            f"New concept '{key}': {(value or '')[:80]}",
                            energy=0.5,
                        )

                    return self.get(entity, key)

        except Exception as e:
            logger.error(f"[KNOWLEDGE] Failed to store '{key}': {e}")
            return None

    # ── Core: recall (hybrid RRF retrieval) ────────────────────────────

    def recall(
        self,
        query: str,
        kinds: List[str] = None,
        entity: str = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        _context: 'TopicContext' = None,
    ) -> List[dict]:
        """
        Hybrid retrieval with 3 signals fused via Reciprocal Rank Fusion (RRF, k=60).

        Signals:
          1. Exact key match
          2. FTS5 full-text match
          3. Vector KNN (requires embedding the query)

        Returns rows sorted by fused RRF score descending.
        """
        rrf_k = 60
        scores = {}  # rowid -> float
        row_cache = {}  # rowid -> dict

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Build filter clause
                filters = ["k.deleted_at IS NULL"]
                params_base = []
                if kinds:
                    placeholders = ','.join('?' for _ in kinds)
                    filters.append(f"k.kind IN ({placeholders})")
                    params_base.extend(kinds)
                if entity:
                    filters.append("k.entity = ?")
                    params_base.append(entity)
                if min_confidence > 0.0:
                    filters.append("k.confidence >= ?")
                    params_base.append(min_confidence)
                where_clause = " AND ".join(filters)

                # ── Signal 1: Exact key match ──────────────────────────
                exact_params = [query] + params_base
                cursor.execute(f"""
                    SELECT k.rowid, k.*
                    FROM knowledge k
                    WHERE k.key = ? AND {where_clause}
                    ORDER BY k.confidence DESC
                    LIMIT 5
                """, exact_params)
                for rank, row in enumerate(cursor.fetchall()):
                    rid = row[0]
                    row_cache[rid] = self._row_to_dict(row)
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)

                # ── Signal 2: FTS5 match ───────────────────────────────
                try:
                    # Sanitize query for FTS: remove special characters
                    fts_query = re.sub(r'[^\w\s]', '', query).strip()
                    if fts_query:
                        # Use prefix search for partial matches
                        fts_terms = ' OR '.join(f'"{w}"*' for w in fts_query.split() if w)
                        if fts_terms:
                            cursor.execute("""
                                SELECT f.rowid, f.rank
                                FROM knowledge_fts f
                                WHERE knowledge_fts MATCH ?
                                ORDER BY f.rank
                                LIMIT 30
                            """, (fts_terms,))
                            fts_rowids = [(r[0], r[1]) for r in cursor.fetchall()]

                            # Filter by kind/entity/confidence via join
                            for rank, (rid, _fts_rank) in enumerate(fts_rowids):
                                if rid in row_cache:
                                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
                                    continue
                                cursor.execute(f"""
                                    SELECT k.rowid, k.*
                                    FROM knowledge k
                                    WHERE k.rowid = ? AND {where_clause}
                                """, [rid] + params_base)
                                row = cursor.fetchone()
                                if row:
                                    row_cache[rid] = self._row_to_dict(row)
                                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
                except Exception as e:
                    logger.debug(f"[KNOWLEDGE] FTS search failed (non-fatal): {e}")

                # ── Signal 3: Vector KNN ───────────────────────────────
                try:
                    query_embedding = self._generate_embedding(query)
                    if query_embedding:
                        blob = pack_embedding(query_embedding)
                        if blob:
                            vec_k = min(limit * 3, 50)
                            cursor.execute("""
                                SELECT rowid, distance
                                FROM knowledge_vec
                                WHERE embedding MATCH ? AND k = ?
                                ORDER BY distance
                            """, (blob, vec_k))
                            vec_results = cursor.fetchall()

                            for rank, (rid, _dist) in enumerate(vec_results):
                                if rid in row_cache:
                                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
                                    continue
                                cursor.execute(f"""
                                    SELECT k.rowid, k.*
                                    FROM knowledge k
                                    WHERE k.rowid = ? AND {where_clause}
                                """, [rid] + params_base)
                                row = cursor.fetchone()
                                if row:
                                    row_cache[rid] = self._row_to_dict(row)
                                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
                except Exception as e:
                    logger.debug(f"[KNOWLEDGE] Vector search failed (non-fatal): {e}")

                cursor.close()

                # Sort by RRF score and return top results
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
                results = []
                for rid, score in ranked:
                    entry = row_cache.get(rid)
                    if entry:
                        entry['rrf_score'] = score
                        results.append(entry)

                # Update last_accessed_at for returned rows
                if results:
                    self._touch_accessed(conn, [r.get('rowid', r.get('id')) for r in results if r])

                return results

        except Exception as e:
            logger.error(f"[KNOWLEDGE] recall failed: {e}")
            if _context is not None:
                _context.record_failure('knowledge_recall', e)
            return []

    def _touch_accessed(self, conn, rowids: list):
        """Update last_accessed_at for a set of rowids."""
        try:
            cursor = conn.cursor()
            for rid in rowids:
                if rid is not None:
                    cursor.execute(
                        "UPDATE knowledge SET last_accessed_at = datetime('now') WHERE rowid = ?",
                        (rid,)
                    )
            cursor.close()
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] Failed to update last_accessed_at: {e}")

    # ── Core: get ──────────────────────────────────────────────────────

    def get(self, entity: str, key: str) -> Optional[dict]:
        """Exact lookup by (entity, key). Updates last_accessed_at on hit."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rowid, *
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                # Update access time
                conn.execute(
                    "UPDATE knowledge SET last_accessed_at = datetime('now') WHERE rowid = ?",
                    (row[0],)
                )

                return self._row_to_dict(row)

        except Exception as e:
            logger.error(f"[KNOWLEDGE] get failed for ({entity}, {key}): {e}")
            return None

    # ── Core: update ───────────────────────────────────────────────────

    def update(self, entity: str, key: str, **changes) -> Optional[dict]:
        """
        Update specific fields on an existing knowledge entry.

        Allowed fields: value, data, confidence, reliability, decay_class.
        Automatically syncs FTS and embedding if value changes.
        """
        allowed = {'value', 'data', 'confidence', 'reliability', 'decay_class'}
        updates = {k: v for k, v in changes.items() if k in allowed}
        if not updates:
            return self.get(entity, key)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT rowid, kind, value
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                existing = cursor.fetchone()
                if not existing:
                    cursor.close()
                    return None

                row_id = existing[0]
                kind = existing[1]
                old_value = existing[2]

                # Serialize data if present
                if 'data' in updates and isinstance(updates['data'], dict):
                    updates['data'] = json.dumps(updates['data'])

                # Build SET clause
                set_parts = [f"{col} = ?" for col in updates]
                set_parts.append("updated_at = datetime('now')")
                set_clause = ", ".join(set_parts)
                params = list(updates.values()) + [row_id]

                cursor.execute(f"UPDATE knowledge SET {set_clause} WHERE rowid = ?", params)

                # Sync FTS and embedding if value changed
                new_value = updates.get('value')
                if new_value and new_value != old_value:
                    self._sync_fts(conn, row_id, key, new_value, kind, entity)
                    emb = self._generate_embedding(f"{key}: {new_value}")
                    self._store_vec(conn, row_id, emb)

                cursor.close()
                return self.get(entity, key)

        except Exception as e:
            logger.error(f"[KNOWLEDGE] update failed for ({entity}, {key}): {e}")
            return None

    # ── Core: forget (soft delete) ─────────────────────────────────────

    def forget(self, entity: str, key: str) -> bool:
        """Soft delete a knowledge entry. Removes from FTS index."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rowid FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                row_id = row[0]
                cursor.execute(
                    "UPDATE knowledge SET deleted_at = datetime('now') WHERE rowid = ?",
                    (row_id,)
                )
                self._remove_fts(conn, row_id)
                cursor.close()

                logger.info(f"[KNOWLEDGE] Soft-deleted ({entity}, {key})")
                return True

        except Exception as e:
            logger.error(f"[KNOWLEDGE] forget failed for ({entity}, {key}): {e}")
            return False

    # ── Core: strengthen ───────────────────────────────────────────────

    def strengthen(self, entity: str, key: str, episode_id: str = None) -> bool:
        """Increment evidence_count, boost confidence with diminishing returns."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rowid, confidence, evidence_count, data, kind
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                row_id, old_conf, old_evidence, raw_data, kind = row
                new_evidence = old_evidence + 1
                boost = 0.05 / math.log2(new_evidence + 1)
                new_conf = min(1.0, old_conf + boost)

                # If episode_id provided and kind=concept, append to source_episodes
                data_update = None
                if episode_id and kind == 'concept':
                    data_obj = {}
                    if raw_data and isinstance(raw_data, str):
                        try:
                            data_obj = json.loads(raw_data)
                        except (json.JSONDecodeError, TypeError):
                            data_obj = {}
                    episodes = data_obj.get('source_episodes', [])
                    if episode_id not in episodes:
                        episodes.append(episode_id)
                        data_obj['source_episodes'] = episodes
                        data_update = json.dumps(data_obj)

                if data_update:
                    cursor.execute("""
                        UPDATE knowledge
                        SET confidence = ?, evidence_count = ?, data = ?,
                            last_accessed_at = datetime('now'), updated_at = datetime('now')
                        WHERE rowid = ?
                    """, (new_conf, new_evidence, data_update, row_id))
                else:
                    cursor.execute("""
                        UPDATE knowledge
                        SET confidence = ?, evidence_count = ?,
                            last_accessed_at = datetime('now'), updated_at = datetime('now')
                        WHERE rowid = ?
                    """, (new_conf, new_evidence, row_id))

                cursor.close()

                logger.debug(
                    f"[KNOWLEDGE] Strengthened ({entity}, {key}): "
                    f"{old_conf:.2f} -> {new_conf:.2f} (evidence={new_evidence})"
                )
                return True

        except Exception as e:
            logger.error(f"[KNOWLEDGE] strengthen failed for ({entity}, {key}): {e}")
            return False

    # ── Core: decay_cycle ──────────────────────────────────────────────

    def decay_cycle(self) -> int:
        """
        Single-pass decay across all decay classes.

        Applies per-class decay rate, factored by reliability multiplier.
        Soft-deletes entries that hit the confidence floor (0.05).
        Emits memory_pressure signals for entries approaching threshold.
        Returns total count of updated rows.
        """
        total_updated = 0

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                for decay_class, rate in self.DECAY_RATES.items():
                    if rate <= 0.0:
                        continue

                    cursor.execute("""
                        UPDATE knowledge SET
                            confidence = MAX(0.05, confidence - (? * CASE COALESCE(reliability, 'reliable')
                                WHEN 'uncertain' THEN 1.5
                                WHEN 'contradicted' THEN 2.0
                                WHEN 'superseded' THEN 3.0
                                ELSE 1.0 END)),
                            updated_at = datetime('now')
                        WHERE deleted_at IS NULL
                          AND decay_class = ?
                          AND confidence > 0.05
                          AND COALESCE(last_accessed_at, created_at) < datetime('now', '-1 hour')
                    """, (rate, decay_class))

                    total_updated += cursor.rowcount

                # Soft-delete entries at confidence floor
                cursor.execute("""
                    UPDATE knowledge
                    SET deleted_at = datetime('now')
                    WHERE deleted_at IS NULL
                      AND confidence <= 0.05
                      AND decay_class != 'permanent'
                """)
                deleted_count = cursor.rowcount

                # Emit memory_pressure for entries approaching threshold
                cursor.execute("""
                    SELECT key, kind, entity, confidence
                    FROM knowledge
                    WHERE deleted_at IS NULL
                      AND confidence < 0.15
                      AND confidence > 0.05
                      AND decay_class != 'permanent'
                    LIMIT 20
                """)
                pressure_rows = cursor.fetchall()

                cursor.close()

            # Emit signals outside the connection context
            for row in pressure_rows:
                self._emit_signal(
                    'memory_pressure',
                    f"Knowledge '{row[0]}' ({row[1]}) approaching decay threshold (confidence={row[3]:.2f})",
                    energy=0.2,
                )

            if total_updated > 0 or deleted_count > 0:
                logger.info(
                    f"[KNOWLEDGE] Decay cycle: {total_updated} decayed, {deleted_count} soft-deleted"
                )

            return total_updated

        except Exception as e:
            logger.error(f"[KNOWLEDGE] decay_cycle failed: {e}")
            return 0

    # ── Procedural: record outcome ─────────────────────────────────────

    def record_procedure_outcome(
        self,
        action_name: str,
        success: bool,
        reward: float = 0.0,
        topic: str = None,
        failure_class: str = None,
    ) -> Optional[dict]:
        """
        Record action outcome for procedural learning.

        Gets or creates a kind=procedure entry for the action, then updates
        stats (attempts, successes, success_rate, avg_reward, reward_history,
        context_stats) and recalculates weight via learning rate.
        """
        entity = 'chalie'
        key = action_name

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Get or create procedure entry
                cursor.execute("""
                    SELECT rowid, data, confidence
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                existing = cursor.fetchone()

                if existing:
                    row_id = existing[0]
                    raw_data = existing[1]
                    data_obj = {}
                    if raw_data and isinstance(raw_data, str):
                        try:
                            data_obj = json.loads(raw_data)
                        except (json.JSONDecodeError, TypeError):
                            data_obj = {}
                else:
                    # Create new procedure entry (ON CONFLICT handles race conditions
                    # where multiple threads try to create the same procedure)
                    cursor.execute("""
                        INSERT INTO knowledge (kind, entity, key, value, data, decay_class,
                                               confidence, reliability, source, evidence_count)
                        VALUES ('procedure', ?, ?, ?, '{}', 'slow', 0.5, 'reliable', 'act_loop', 1)
                        ON CONFLICT(entity, key) DO UPDATE SET
                            updated_at = datetime('now'),
                            deleted_at = NULL
                    """, (entity, key, action_name))
                    # Re-fetch to get rowid and current data (may have been created by another thread)
                    cursor.execute("""
                        SELECT rowid, data FROM knowledge
                        WHERE entity = ? AND key = ?
                    """, (entity, key))
                    row = cursor.fetchone()
                    if not row:
                        logger.warning(f"[KNOWLEDGE] Failed to fetch procedure '{action_name}' after upsert")
                        return None
                    row_id = row[0]
                    raw_data = row[1]
                    data_obj = {}
                    if raw_data and isinstance(raw_data, str):
                        try:
                            data_obj = json.loads(raw_data)
                        except (json.JSONDecodeError, TypeError):
                            data_obj = {}

                    # Sync FTS
                    self._sync_fts(conn, row_id, key, action_name, 'procedure', entity)

                # Update stats
                total_attempts = data_obj.get('total_attempts', 0) + 1
                total_successes = data_obj.get('total_successes', 0) + (1 if success else 0)
                success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

                # EMA for avg_reward
                old_avg_reward = data_obj.get('avg_reward', 0.0)
                if total_attempts <= 1:
                    avg_reward = reward
                else:
                    avg_reward = (old_avg_reward * (total_attempts - 1) + reward) / total_attempts

                # Reward history
                reward_history = data_obj.get('reward_history', [])
                reward_history.append({
                    'reward': reward,
                    'success': success,
                    'failure_class': failure_class,
                    'timestamp': time.time(),
                })
                reward_history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                reward_history = reward_history[:self._REWARD_HISTORY_MAX]

                # Context stats
                context_stats = data_obj.get('context_stats', {})
                if topic:
                    topic_data = context_stats.get(topic, {'attempts': 0, 'successes': 0})
                    topic_data['attempts'] = topic_data.get('attempts', 0) + 1
                    if success:
                        topic_data['successes'] = topic_data.get('successes', 0) + 1
                    context_stats[topic] = topic_data

                # Recalculate weight
                old_weight = data_obj.get('weight', self._DEFAULT_ACTION_WEIGHT)
                clamped_reward = max(-0.5, min(0.5, avg_reward))
                target_weight = success_rate * (1.0 + clamped_reward)
                new_weight = old_weight + self._LEARNING_RATE * (target_weight - old_weight)
                new_weight = max(0.1, min(5.0, new_weight))

                data_obj.update({
                    'total_attempts': total_attempts,
                    'total_successes': total_successes,
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'weight': new_weight,
                    'reward_history': reward_history,
                    'context_stats': context_stats,
                })

                summary = f"{success_rate * 100:.0f}% success over {total_attempts} attempts"
                cursor.execute("""
                    UPDATE knowledge
                    SET data = ?, value = ?, evidence_count = evidence_count + 1,
                        updated_at = datetime('now')
                    WHERE rowid = ?
                """, (json.dumps(data_obj), summary, row_id))

                cursor.close()

                fc_tag = f", failure_class={failure_class}" if failure_class else ""
                logger.info(
                    f"[KNOWLEDGE] Recorded procedure outcome for '{action_name}': "
                    f"success={success}, reward={reward:.2f}{fc_tag}"
                )

                return self.get(entity, key)

        except Exception as e:
            logger.error(f"[KNOWLEDGE] record_procedure_outcome failed for '{action_name}': {e}")
            return None

    # ── Procedural: get ranked ─────────────────────────────────────────

    def get_ranked_procedures(self, topic: str = None, limit: int = 10) -> List[dict]:
        """
        Return procedures ranked by expected value.

        expected_value = success_rate * (1 + clamp(avg_reward)) * topic_affinity
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT key, value, data, confidence
                    FROM knowledge
                    WHERE kind = 'procedure' AND entity = 'chalie' AND deleted_at IS NULL
                    ORDER BY confidence DESC
                """)
                rows = cursor.fetchall()
                cursor.close()

                ranked = []
                for row in rows:
                    data_obj = {}
                    raw_data = row[2]
                    if raw_data and isinstance(raw_data, str):
                        try:
                            data_obj = json.loads(raw_data)
                        except (json.JSONDecodeError, TypeError):
                            data_obj = {}

                    success_rate = data_obj.get('success_rate', 0.0)
                    avg_reward = data_obj.get('avg_reward', 0.0)
                    weight = data_obj.get('weight', self._DEFAULT_ACTION_WEIGHT)
                    total_attempts = data_obj.get('total_attempts', 0)
                    context_stats = data_obj.get('context_stats', {})

                    # Topic affinity
                    topic_affinity = 1.0
                    if topic and topic in context_stats:
                        td = context_stats[topic]
                        t_attempts = td.get('attempts', 0)
                        t_successes = td.get('successes', 0)
                        if t_attempts > 0:
                            topic_affinity = t_successes / t_attempts

                    clamped_reward = max(-0.5, min(0.5, avg_reward))
                    expected_value = success_rate * (1.0 + clamped_reward) * topic_affinity

                    ranked.append({
                        'name': row[1] or row[0],
                        'weight': weight,
                        'success_rate': success_rate,
                        'avg_reward': avg_reward,
                        'attempts': total_attempts,
                        'topic_affinity': topic_affinity,
                        'expected_value': expected_value,
                    })

                ranked.sort(key=lambda x: x['expected_value'], reverse=True)
                return ranked[:limit]

        except Exception as e:
            logger.error(f"[KNOWLEDGE] get_ranked_procedures failed: {e}")
            return []

    # ── Traits: get_traits_for_prompt ──────────────────────────────────

    def get_traits_for_prompt(self, query_embedding=None, limit: int = 20) -> List[dict]:
        """
        Three-tier retrieval for prompt injection (ported from UserTraitService):

        1. Core traits (decay_class=permanent, kind IN (trait, preference)) — always included
        2. Semantic KNN from knowledge_vec (if query_embedding provided)
        3. Wildcard: top by confidence for remaining budget

        Returns formatted trait dicts.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                max_traits = min(limit, _MAX_TRAITS_IN_PROMPT)

                # Tier 1: Core traits (permanent, always present)
                cursor.execute("""
                    SELECT rowid, key, value, confidence, kind, entity
                    FROM knowledge
                    WHERE decay_class = 'permanent'
                      AND kind IN ('trait', 'preference')
                      AND deleted_at IS NULL
                      AND confidence > ?
                    ORDER BY confidence DESC
                """, (_INJECTION_THRESHOLD,))
                core_rows = cursor.fetchall()

                core_traits = []
                core_keys = set()
                for row in core_rows:
                    core_traits.append({
                        'key': row[1],
                        'value': row[2],
                        'confidence': row[3],
                        'kind': row[4],
                        'label': _confidence_label(row[3]),
                    })
                    core_keys.add(row[1])

                remaining_slots = max_traits - len(core_traits)
                semantic_traits = []
                wildcard_traits = []

                if remaining_slots > 0 and query_embedding is not None:
                    # Tier 2: Semantic matches via KNN
                    blob = pack_embedding(query_embedding)
                    if blob:
                        cursor.execute("""
                            SELECT v.rowid, v.distance
                            FROM knowledge_vec v
                            WHERE v.embedding MATCH ? AND k = ?
                            ORDER BY v.distance
                        """, (blob, _SEMANTIC_RETRIEVAL_K))
                        vec_results = cursor.fetchall()

                        for v_row in vec_results:
                            rid = v_row[0]
                            cursor.execute("""
                                SELECT key, value, confidence, kind
                                FROM knowledge
                                WHERE rowid = ?
                                  AND kind IN ('trait', 'preference', 'fact')
                                  AND deleted_at IS NULL
                                  AND confidence > ?
                                  AND decay_class != 'permanent'
                            """, (rid, _INJECTION_THRESHOLD))
                            krow = cursor.fetchone()
                            if krow and krow[0] not in core_keys:
                                semantic_traits.append({
                                    'key': krow[0],
                                    'value': krow[1],
                                    'confidence': krow[2],
                                    'kind': krow[3],
                                    'label': _confidence_label(krow[2]),
                                })

                    # Tier 3: Identity wildcards
                    semantic_keys = {t['key'] for t in semantic_traits}
                    exclude_keys = core_keys | semantic_keys
                    exclude_placeholders = ','.join('?' for _ in exclude_keys)

                    if exclude_keys:
                        cursor.execute(f"""
                            SELECT key, value, confidence, kind
                            FROM knowledge
                            WHERE kind IN ('trait', 'preference')
                              AND decay_class != 'permanent'
                              AND deleted_at IS NULL
                              AND confidence >= ?
                              AND key NOT IN ({exclude_placeholders})
                            ORDER BY confidence DESC, evidence_count DESC
                            LIMIT ?
                        """, [_WILDCARD_CONFIDENCE] + list(exclude_keys) + [_WILDCARD_SLOTS])
                    else:
                        cursor.execute("""
                            SELECT key, value, confidence, kind
                            FROM knowledge
                            WHERE kind IN ('trait', 'preference')
                              AND decay_class != 'permanent'
                              AND deleted_at IS NULL
                              AND confidence >= ?
                            ORDER BY confidence DESC, evidence_count DESC
                            LIMIT ?
                        """, (_WILDCARD_CONFIDENCE, _WILDCARD_SLOTS))

                    for wrow in cursor.fetchall():
                        wildcard_traits.append({
                            'key': wrow[0],
                            'value': wrow[1],
                            'confidence': wrow[2],
                            'kind': wrow[3],
                            'label': _confidence_label(wrow[2]),
                        })

                    # Cap semantic to fill remaining after wildcards
                    semantic_cap = remaining_slots - len(wildcard_traits)
                    semantic_traits = semantic_traits[:max(0, semantic_cap)]

                elif remaining_slots > 0:
                    # No embedding — get highest confidence non-core
                    cursor.execute("""
                        SELECT key, value, confidence, kind
                        FROM knowledge
                        WHERE kind IN ('trait', 'preference', 'fact')
                          AND decay_class != 'permanent'
                          AND deleted_at IS NULL
                          AND confidence > ?
                        ORDER BY confidence DESC
                        LIMIT ?
                    """, (_INJECTION_THRESHOLD, remaining_slots))

                    for row in cursor.fetchall():
                        if row[0] not in core_keys:
                            semantic_traits.append({
                                'key': row[0],
                                'value': row[1],
                                'confidence': row[2],
                                'kind': row[3],
                                'label': _confidence_label(row[2]),
                            })

                cursor.close()

                return core_traits + semantic_traits + wildcard_traits

        except Exception as e:
            logger.error(f"[KNOWLEDGE] get_traits_for_prompt failed: {e}")
            return []

    # ── Filtered: get_by_kind ──────────────────────────────────────────

    def get_by_kind(self, kind: str, entity: str = None, limit: int = 50, min_confidence: float = 0.0) -> List[dict]:
        """Simple filtered query for a specific kind."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                if entity:
                    cursor.execute("""
                        SELECT rowid, *
                        FROM knowledge
                        WHERE kind = ? AND entity = ? AND deleted_at IS NULL AND confidence >= ?
                        ORDER BY confidence DESC
                        LIMIT ?
                    """, (kind, entity, min_confidence, limit))
                else:
                    cursor.execute("""
                        SELECT rowid, *
                        FROM knowledge
                        WHERE kind = ? AND deleted_at IS NULL AND confidence >= ?
                        ORDER BY confidence DESC
                        LIMIT ?
                    """, (kind, min_confidence, limit))

                rows = cursor.fetchall()
                cursor.close()
                return [self._row_to_dict(r) for r in rows]

        except Exception as e:
            logger.error(f"[KNOWLEDGE] get_by_kind failed for kind={kind}: {e}")
            return []
