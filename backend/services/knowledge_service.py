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
with a single service operating on the ``knowledge`` table.

All knowledge entries share a common schema: (kind, entity, key, value, data,
decay_class, confidence, source, evidence_count).

Retrieval uses Reciprocal Rank Fusion (RRF) across three signals: exact key match,
FTS5 full-text search, and sqlite-vec KNN.
"""

import json
import logging
import math
import re
import threading
from typing import List, Optional


from services.database_service import get_shared_db_service
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)

# ── Stop words (NLTK-backed, lazy-loaded) ─────────────────────────────

_nltk_stop_words = None
_stop_words_lock = threading.Lock()


def _get_stop_words() -> frozenset:
    """Return the NLTK English stop word set, downloading the corpus on first use."""
    global _nltk_stop_words
    if _nltk_stop_words is not None:
        return _nltk_stop_words
    with _stop_words_lock:
        if _nltk_stop_words is not None:
            return _nltk_stop_words
        import nltk
        try:
            from nltk.corpus import stopwords
            _nltk_stop_words = frozenset(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _nltk_stop_words = frozenset(stopwords.words('english'))
    return _nltk_stop_words


# ── Trait validation (deterministic, zero LLM) ────────────────────────
# Ported from UserTraitService — catches garbage traits from weak models.

_PLACEHOLDER_RE = re.compile(
    r'^(?:unknown|n/?a|none|null|undefined|not specified|unspecified|empty|default)$',
    re.IGNORECASE,
)

_MAX_TOPIC_SLUG_SEGMENTS = 3


def _extract_content_words(text: str, stop_words: frozenset = None) -> list:
    """Extract meaningful content words (not stop words, len > 1)."""
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    if stop_words is None:
        stop_words = _get_stop_words()
    return [w for w in words if w not in stop_words and len(w) > 1]


def _validate_trait(key: str, value: str) -> Optional[str]:
    """Validate a trait before storage. Returns rejection reason or None if valid."""
    if not key or len(key) < 2:
        return 'key_too_short'

    stop_words = _get_stop_words()
    key_segments = [w for w in re.split(r'[_\-\s]+', key.lower()) if w]
    content_key = [w for w in key_segments if w not in stop_words and len(w) > 1]
    if not content_key:
        return 'key_is_stop_words'

    if not value or len(value.strip()) < 3:
        return 'value_too_short'
    if len(value) > 500:
        return 'value_too_long'

    if _PLACEHOLDER_RE.match(value.strip()):
        return 'placeholder_value'

    content_words_value = _extract_content_words(value, stop_words)
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
        topic_content = [s for s in slug_segments if s not in stop_words and len(s) > 1]
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
        'very_slow':  0.00015, # ~110 days from 0.85 → floor (methodology records)
        'slow':       0.002,   # ~8 days from 0.85 → floor
        'standard':   0.005,   # ~3 days from 0.85 → floor
        'fast':       0.015,   # ~1 day from 0.85 → floor
        'ephemeral':  0.040,   # ~20 hours from 0.85 → floor
    }

    VALID_KINDS = {'trait', 'fact', 'procedure', 'preference', 'rule', 'metric', 'moment'}
    VALID_DECAY_CLASSES = {'permanent', 'very_slow', 'slow', 'standard', 'fast', 'ephemeral'}

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

    def _sync_fts(self, conn, rowid: int, key: str = None, value: str = None,
                  kind: str = None, entity: str = None, search_queries: str = None):
        """Sync the FTS index for a knowledge row.

        When called with only rowid (e.g. from doc2query background thread),
        reads the current row values from the DB before syncing.
        When key/value/kind/entity are provided, search_queries can also be
        passed directly to avoid an extra DB roundtrip.
        """
        try:
            cursor = conn.cursor()

            # If key/value/kind/entity not provided, read from DB
            if key is None:
                cursor.execute(
                    "SELECT key, value, kind, entity, search_queries FROM knowledge WHERE rowid = ?",
                    (rowid,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return
                key, value, kind, entity = row[0], row[1], row[2], row[3]
                search_queries = row[4]

            # Delete old entry (if exists)
            cursor.execute(
                "DELETE FROM knowledge_fts WHERE rowid = ?",
                (rowid,)
            )
            # Insert new entry including search_queries column
            cursor.execute(
                "INSERT INTO knowledge_fts(rowid, key, value, kind, entity, search_queries) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rowid, key, value or '', kind, entity, search_queries or '')
            )
            cursor.close()
        except Exception as e:
            logger.warning(f"[KNOWLEDGE] FTS sync failed for rowid={rowid}: {e}")

    def _remove_fts(self, conn, rowid: int):
        """Remove a row from the FTS index."""
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_fts WHERE rowid = ?", (rowid,))
            cursor.close()
        except Exception as e:
            logger.warning(f"[KNOWLEDGE] FTS removal failed for rowid={rowid}: {e}")

    def _schedule_doc2query(self, rowid: int, key: str, value: str):
        """Fire-and-forget: generate search queries for a knowledge entry and persist them."""
        import threading

        def _run():
            try:
                from services.doc2query_service import get_doc2query_service
                d2q = get_doc2query_service()
                if not d2q.is_available():
                    return
                queries = d2q.generate_queries(f"{key}: {value}")
                if not queries:
                    return
                import json as _json
                with self.db.connection() as conn:
                    conn.execute(
                        "UPDATE knowledge SET search_queries = ? WHERE rowid = ?",
                        (_json.dumps(queries), rowid)
                    )
                    self._sync_fts(conn, rowid)
            except Exception as e:
                logger.warning(f"[KNOWLEDGE] doc2query failed for rowid={rowid}: {e}")

        threading.Thread(target=_run, daemon=True).start()

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
        source: str = None,
        embedding=None,
    ) -> Optional[int]:
        """
        Store or reinforce a knowledge entry. UPSERT on (entity, key).

        On conflict (same entity+key exists):
        - Same value re-observed: reinforce (boost confidence with diminishing returns)
        - Different value AND new confidence > old * 2: overwrite
        - Otherwise: just reinforce evidence_count and timestamp

        For kind=trait: applies trait validation before storage.

        Returns the integer rowid of the stored/updated record, or None on failure.
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
            _schedule_d2q_args = None  # (rowid, key, value) — set when doc2query should run

            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Check for existing entry
                cursor.execute("""
                    SELECT rowid, id, kind, value, data, confidence, evidence_count
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
                        if confidence >= old_confidence * 0.8:
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
                            source = COALESCE(?, source),
                            updated_at = datetime('now')
                        WHERE rowid = ?
                    """, (update_value, update_data, new_confidence, new_evidence,
                          source, row_id))

                    # Update embedding if value changed
                    if value_changed and embedding is None and value:
                        embedding = self._generate_embedding(f"{key}: {value}")
                    if value_changed or embedding:
                        self._store_vec(conn, row_id, embedding)

                    # Sync FTS if value changed
                    if value_changed:
                        self._sync_fts(conn, row_id, key, update_value, kind, entity)
                        # Re-generate search queries for the new value
                        if value:
                            _schedule_d2q_args = (row_id, key, value)

                    cursor.close()

                else:
                    # New entry
                    if embedding is None and value:
                        embedding = self._generate_embedding(f"{key}: {value}")

                    cursor.execute("""
                        INSERT INTO knowledge (kind, entity, key, value, data, decay_class,
                                               confidence, source, evidence_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (kind, entity, key, value, data_json, decay_class,
                          confidence, source))

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

                    # Schedule doc2query after connection commits
                    if value:
                        _schedule_d2q_args = (row_id, key, value)

            # Fire doc2query AFTER the connection context exits (row committed)
            if _schedule_d2q_args:
                self._schedule_doc2query(*_schedule_d2q_args)

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
        _context=None,
    ) -> List[dict]:
        """
        Hybrid retrieval with 3 signals fused via Reciprocal Rank Fusion (RRF, k=60).

        Signals:
          1. Exact key match
          2. FTS5 full-text match
          3. Vector KNN (requires embedding the query)

        Returns rows sorted by fused RRF score descending.
        
        When FTS and exact-key both return zero results, vector-only scores are
        boosted 2.5x to ensure semantic matches are properly ranked.
        """
        rrf_k = 60
        scores = {}  # rowid -> float
        row_cache = {}  # rowid -> dict
        
        # Track which signals returned results for vector-only boost
        exact_count = 0
        fts_count = 0
        vec_count = 0

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
                exact_rows = cursor.fetchall()
                exact_count = len(exact_rows)
                for rank, row in enumerate(exact_rows):
                    rid = row[0]
                    row_cache[rid] = self._row_to_dict(row)
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)

                # ── Signal 2: FTS5 match ───────────────────────────────
                try:
                    # Sanitize query for FTS: remove special characters
                    fts_query = re.sub(r'[^\w\s]', '', query).strip()
                    if fts_query:
                        # Filter stop words before building FTS terms so that
                        # purely stop-word queries fall through to the vector-only
                        # 2.5x boost path instead of returning low-quality matches.
                        fts_words = [
                            w for w in fts_query.split()
                            if w and w.lower() not in _get_stop_words()
                        ]
                        # Use prefix search for partial matches
                        fts_terms = ' OR '.join(f'"{w}"*' for w in fts_words if w)
                        if fts_terms:
                            cursor.execute("""
                                SELECT f.rowid, f.rank
                                FROM knowledge_fts f
                                WHERE knowledge_fts MATCH ?
                                ORDER BY f.rank
                                LIMIT 30
                            """, (fts_terms,))
                            fts_rowids = [(r[0], r[1]) for r in cursor.fetchall()]
                            fts_count = len(fts_rowids)

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
                            vec_count = len(vec_results)

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

                # ── Vector-only boost ──────────────────────────────────
                # When FTS and exact-key both return zero results, boost vector-only
                # scores to ensure semantic matches are properly ranked.
                # Example: "sister named Elena" vs stored "siblings" - zero word overlap
                # means only vector search fires, but should still rank highly.
                if exact_count == 0 and fts_count == 0 and vec_count > 0:
                    vector_only_boost = 2.5
                    logger.debug(
                        f"[KNOWLEDGE] Vector-only recall (exact={exact_count}, fts={fts_count}, vec={vec_count}), "
                        f"applying {vector_only_boost}x boost"
                    )
                    scores = {rid: score * vector_only_boost for rid, score in scores.items()}

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

        Allowed fields: value, data, confidence, decay_class.
        Automatically syncs FTS and embedding if value changes.
        """
        allowed = {'value', 'data', 'confidence', 'decay_class'}
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

    def forget_all_by_entity(self, entity: str, kind: Optional[str] = None) -> int:
        """
        Bulk soft-delete every active row for an entity (optionally filtered by
        kind) and keep the FTS shadow table in sync.

        Used by cascade-delete callers (e.g. moments) so they don't have to
        reach into private helpers like ``_remove_fts``. Returns the number of
        rows that were soft-deleted.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                if kind is not None:
                    cursor.execute(
                        "SELECT rowid FROM knowledge "
                        "WHERE entity = ? AND kind = ? AND deleted_at IS NULL",
                        (entity, kind),
                    )
                else:
                    cursor.execute(
                        "SELECT rowid FROM knowledge "
                        "WHERE entity = ? AND deleted_at IS NULL",
                        (entity,),
                    )
                rowids = [r[0] for r in cursor.fetchall()]

                if not rowids:
                    cursor.close()
                    return 0

                placeholders = ','.join('?' for _ in rowids)
                cursor.execute(
                    f"UPDATE knowledge SET deleted_at = datetime('now') "
                    f"WHERE rowid IN ({placeholders})",
                    rowids,
                )
                affected = cursor.rowcount

                for rid in rowids:
                    self._remove_fts(conn, rid)

                cursor.close()

            logger.info(
                f"[KNOWLEDGE] forget_all_by_entity soft-deleted {affected} row(s) "
                f"for entity={entity} kind={kind or '*'}"
            )
            return affected

        except Exception as e:
            logger.error(
                f"[KNOWLEDGE] forget_all_by_entity failed for entity={entity} kind={kind}: {e}"
            )
            return 0

    # ── Core: strengthen ───────────────────────────────────────────────

    def strengthen(self, entity: str, key: str, episode_id: str = None) -> bool:
        """Increment evidence_count, boost confidence with diminishing returns."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rowid, confidence, evidence_count
                    FROM knowledge
                    WHERE entity = ? AND key = ? AND deleted_at IS NULL
                """, (entity, key))
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return False

                row_id, old_conf, old_evidence = row
                new_evidence = old_evidence + 1
                boost = 0.05 / math.log2(new_evidence + 1)
                new_conf = min(1.0, old_conf + boost)

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

        Applies per-class decay rate to all active entries.
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
                            confidence = MAX(0.05, confidence - ?),
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
                cursor.fetchall()  # drain cursor

                cursor.close()

            if total_updated > 0 or deleted_count > 0:
                logger.info(
                    f"[KNOWLEDGE] Decay cycle: {total_updated} decayed, {deleted_count} soft-deleted"
                )

            return total_updated

        except Exception as e:
            logger.error(f"[KNOWLEDGE] decay_cycle failed: {e}")
            return 0

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

    # ── Hard delete by rowid ───────────────────────────────────────────

    def hard_delete_by_id(self, row_id: int) -> bool:
        """Hard-delete a knowledge entry by rowid. Cascades to knowledge_vec and FTS."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT rowid FROM knowledge WHERE rowid = ?", (row_id,))
                if not cursor.fetchone():
                    cursor.close()
                    return False
                self._remove_fts(conn, row_id)
                cursor.execute("DELETE FROM knowledge_vec WHERE rowid = ?", (row_id,))
                cursor.execute("DELETE FROM knowledge WHERE rowid = ?", (row_id,))
                cursor.close()
                return True
        except Exception as e:
            logger.error(f"[KNOWLEDGE] hard_delete_by_id({row_id}) failed: {e}")
            return False

    # ── Find similar traits ────────────────────────────────────────────

    def find_similar_traits(self, embedding: list, exclude_id: int, limit: int = 3) -> list:
        """Find traits/preferences semantically similar to the given embedding."""
        try:
            blob = pack_embedding(embedding)
            if not blob:
                return []
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT v.rowid, v.distance
                    FROM knowledge_vec v
                    WHERE v.embedding MATCH ? AND k = 20
                    ORDER BY v.distance
                """, (blob,))
                vec_rows = cursor.fetchall()
                results = []
                for rid, dist in vec_rows:
                    if dist >= 0.25 or rid == exclude_id:
                        continue
                    cursor.execute("""
                        SELECT rowid, key, value, confidence, entity
                        FROM knowledge
                        WHERE rowid = ?
                          AND kind IN ('trait', 'preference')
                          AND deleted_at IS NULL
                    """, (rid,))
                    krow = cursor.fetchone()
                    if krow:
                        results.append({
                            'id': krow[0], 'key': krow[1], 'value': krow[2],
                            'confidence': krow[3], 'entity': krow[4],
                        })
                    if len(results) >= limit:
                        break
                cursor.close()
                return results
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] find_similar_traits failed: {e}")
            return []

    # ── Update confidence by rowid ─────────────────────────────────────

    def update_confidence(self, row_id: int, new_confidence: float) -> bool:
        """Update the confidence of a knowledge entry by rowid."""
        from services.time_utils import utc_now
        new_confidence = max(0.0, min(1.0, new_confidence))
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE knowledge SET confidence = ?, updated_at = ? WHERE rowid = ?",
                    (new_confidence, utc_now().isoformat(), row_id)
                )
                conn.commit()
                affected = cursor.rowcount > 0
                cursor.close()
                return affected
        except Exception as e:
            logger.error(f"[KNOWLEDGE] update_confidence({row_id}) failed: {e}")
            return False

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
