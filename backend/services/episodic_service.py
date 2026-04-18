# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episodic Service — Unified storage, CRUD, and retrieval for episodes.

Combines the former EpisodicStorageService and EpisodicRetrievalService into
a single cohesive service. Provides episode persistence, hybrid search
(vector + FTS5), composite scoring, and memory consolidation.
"""

import json
import logging
import math
import struct
import uuid
from datetime import datetime
from typing import Optional, List

try:
    import nltk
    from nltk import pos_tag, word_tokenize, RegexpParser
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

from services.database_service import DatabaseService, DictCursor
from services.embedding_utils import pack_embedding
from services.time_utils import utc_now, parse_utc


class EpisodicService:
    """Manages episode storage, CRUD, retrieval, and hybrid search."""

    def __init__(self, database_service: DatabaseService, config: dict = None):
        """Initialize the episodic service.

        Args:
            database_service: DatabaseService instance for connection management.
            config: Optional config dict with retrieval weights and tuning params.
        """
        self.db_service = database_service
        self.config = config or {}
        self.embedding_dimensions = self.config.get('embedding_dimensions', 256)
        self.weights = self.config.get('inference_weights', {
            'vector_similarity': 4,
            'retrieval_weight': 3,
            'arousal_salience': 2,
            'emotional_congruence': 1,
            'temporal_proximity': 1,
        })
        # Freshness decay rate (lambda)
        self.decay_rate = self.config.get('freshness_decay_rate', 0.05)
        # Reconsolidation boost
        self.reconsolidation_boost = self.config.get('reconsolidation_boost', 0.2)

    # ── Storage / CRUD ───────────────────────────────────────────────

    def store_episode(self, episode_data: dict, *, embedding=None) -> str:
        """Store a new episode in the database.

        Required fields: gist, salience, channel.

        Args:
            episode_data: Episode fields dict. May include an 'embedding' key
                for backwards compatibility, but the ``embedding`` kwarg takes
                precedence when both are supplied.
            embedding: Optional embedding vector (list of floats). When provided,
                supersedes episode_data.get('embedding').

        Returns:
            UUID of the created episode.

        Raises:
            ValueError: If any required field is missing.
        """
        required_fields = ['gist', 'salience', 'channel']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            episode_id = str(uuid.uuid4())
            # Kwarg takes precedence; fall back to the dict key for callers
            # that embedded the vector inside episode_data (old path).
            if embedding is None:
                embedding = episode_data.get('embedding')

            transcript_id_start = episode_data.get('transcript_id_start')
            transcript_id_end = episode_data.get('transcript_id_end')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # When >50% transcript overlap with an existing episode, store the new
                # episode anyway and create a super episode linking both.
                overlapping_episodes = []
                if transcript_id_start is not None and transcript_id_end is not None:
                    new_span = transcript_id_end - transcript_id_start + 1
                    cursor.execute("""
                        SELECT id, transcript_id_start, transcript_id_end,
                               transcript_ids, gist
                        FROM episodes
                        WHERE transcript_id_start IS NOT NULL
                          AND transcript_id_end IS NOT NULL
                          AND transcript_id_start <= ?
                          AND transcript_id_end >= ?
                          AND deleted_at IS NULL
                          AND (consolidated_from IS NULL OR consolidated_from = '[]')
                    """, (transcript_id_end, transcript_id_start))
                    for overlap_row in cursor.fetchall():
                        existing_start = overlap_row[1]
                        existing_end = overlap_row[2]
                        overlap_start = max(transcript_id_start, existing_start)
                        overlap_end = min(transcript_id_end, existing_end)
                        overlap_count = max(0, overlap_end - overlap_start + 1)
                        if new_span > 0 and overlap_count / new_span > 0.5:
                            overlapping_episodes.append({
                                'id': overlap_row[0],
                                'transcript_id_start': existing_start,
                                'transcript_id_end': existing_end,
                                'transcript_ids': overlap_row[3],
                                'gist': overlap_row[4] or '',
                            })

                cursor.execute("""
                    INSERT INTO episodes (
                        id, gist, salience, channel,
                        transcript_ids, transcript_id_start, transcript_id_end,
                        emotional_valence, emotional_arousal,
                        consolidated_from, storage_strength, retrieval_weight
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['channel'],
                    json.dumps(episode_data.get('transcript_ids', [])),
                    transcript_id_start,
                    transcript_id_end,
                    episode_data.get('emotional_valence'),
                    episode_data.get('emotional_arousal'),
                    json.dumps(episode_data.get('consolidated_from', [])),
                    episode_data.get('storage_strength', 1.0),
                    episode_data.get('retrieval_weight', 1.0),
                ))

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                # Sync FTS index (external-content table requires explicit insert)
                rowid = cursor.execute(
                    "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
                if rowid:
                    conn.execute(
                        "INSERT INTO episodes_fts(rowid, gist) VALUES (?, ?)",
                        (rowid[0], episode_data['gist']),
                    )

                # Create super episodes for each overlapping existing episode
                for existing in overlapping_episodes:
                    existing_ids_raw = existing['transcript_ids']
                    if isinstance(existing_ids_raw, str):
                        try:
                            existing_ids = json.loads(existing_ids_raw)
                        except Exception:
                            existing_ids = []
                    else:
                        existing_ids = existing_ids_raw or []

                    new_ids = episode_data.get('transcript_ids', [])
                    if isinstance(new_ids, str):
                        try:
                            new_ids = json.loads(new_ids)
                        except Exception:
                            new_ids = []

                    merged_ids = sorted(set(existing_ids) | set(new_ids))
                    super_start = min(
                        existing['transcript_id_start'],
                        transcript_id_start,
                    )
                    super_end = max(
                        existing['transcript_id_end'],
                        transcript_id_end,
                    )
                    existing_gist = existing['gist'][:200]
                    new_gist = episode_data['gist'][:200]
                    super_gist = f"{existing_gist} | {new_gist}"
                    super_id = str(uuid.uuid4())

                    cursor.execute("""
                        INSERT INTO episodes (
                            id, gist, salience, channel,
                            transcript_ids, transcript_id_start, transcript_id_end,
                            emotional_valence, emotional_arousal,
                            consolidated_from, storage_strength, retrieval_weight
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        super_id,
                        super_gist,
                        episode_data['salience'],
                        episode_data['channel'],
                        json.dumps(merged_ids),
                        super_start,
                        super_end,
                        episode_data.get('emotional_valence'),
                        episode_data.get('emotional_arousal'),
                        json.dumps([existing['id'], episode_id]),
                        1.0,
                        1.0,
                    ))

                    super_rowid = cursor.execute(
                        "SELECT rowid FROM episodes WHERE id = ?", (super_id,)
                    ).fetchone()
                    if super_rowid:
                        conn.execute(
                            "INSERT INTO episodes_fts(rowid, gist) VALUES (?, ?)",
                            (super_rowid[0], super_gist),
                        )

                    logging.info(
                        f"Created super episode {super_id} linking "
                        f"{existing['id']} + {episode_id}"
                    )

                cursor.close()

                logging.info(f"Stored episode {episode_id} for channel '{episode_data['channel']}'")

                return episode_id

        except Exception as e:
            logging.error(f"Failed to store episode: {e}")
            raise

    def _store_embedding(self, conn, episode_id: str, embedding):
        """Store an embedding blob in the ``episodes_vec`` virtual table."""
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return

            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )
            cursor.close()
        except Exception as e:
            logging.warning(f"Failed to store episode embedding: {e}")

    def update_episode(self, episode_id: str, fields: dict, embedding=None) -> None:
        """Update an existing episode with the provided field values.

        Args:
            episode_id: The episode UUID.
            fields: Column→value mapping to apply. Empty dict is a no-op.
            embedding: Optional new embedding vector to store in episodes_vec.

        Raises:
            Exception: Propagates any database error to the caller.
        """
        if not fields and embedding is None:
            return

        with self.db_service.connection() as conn:
            if fields:
                set_clauses = [f"{key} = ?" for key in fields]
                set_clauses.append("updated_at = datetime('now')")
                values = list(fields.values()) + [episode_id]
                query = f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?"
                conn.execute(query, values)

            if embedding is not None:
                self._store_embedding(conn, episode_id, embedding)

    def soft_delete(self, episode_id: str) -> None:
        """Soft-delete an episode by setting its ``deleted_at`` timestamp.

        Raises:
            Exception: Propagates any database error to the caller.
        """
        with self.db_service.connection() as conn:
            conn.execute("""
                UPDATE episodes
                SET deleted_at = datetime('now')
                WHERE id = ? AND deleted_at IS NULL
            """, (episode_id,))

    def soft_delete_episode(self, episode_id: str) -> bool:
        """Soft-delete an episode. Returns True on success, False otherwise."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE episodes
                    SET deleted_at = datetime('now')
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                rows_deleted = cursor.rowcount
                cursor.close()

                if rows_deleted > 0:
                    logging.info(f"Soft deleted episode {episode_id}")
                    return True
                else:
                    logging.warning(f"Episode {episode_id} not found or already deleted")
                    return False

        except Exception as e:
            logging.error(f"Failed to soft delete episode: {e}")
            return False

    def set_consolidated_into(self, leaf_id: str, super_id: str) -> None:
        """Set the consolidated_into back-pointer on a leaf episode.

        Args:
            leaf_id: UUID of the leaf episode to update.
            super_id: Row id (integer) or UUID of the super-episode. The value
                is stored as-is; callers should pass the rowid integer.

        Raises:
            Exception: Propagates any database error to the caller.
        """
        with self.db_service.connection() as conn:
            conn.execute(
                "UPDATE episodes SET consolidated_into = ? WHERE id = ?",
                (super_id, leaf_id),
            )

    def get_episode_by_id(self, episode_id: str) -> Optional[dict]:
        """Retrieve a single non-deleted episode by its UUID.

        Also triggers an access count + storage strength boost.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, gist, salience, channel,
                           created_at, updated_at, last_accessed_at, access_count,
                           transcript_ids, transcript_id_start, transcript_id_end,
                           emotional_valence, emotional_arousal,
                           consolidated_from, consolidated_into,
                           storage_strength, retrieval_weight
                    FROM episodes
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                # Update access tracking
                self._update_activation_score(episode_id)

                episode = {
                    'id': str(row[0]),
                    'gist': row[1],
                    'salience': row[2],
                    'channel': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'last_accessed_at': row[6],
                    'access_count': row[7],
                    'transcript_ids': row[8] if row[8] is not None else '[]',
                    'transcript_id_start': row[9],
                    'transcript_id_end': row[10],
                    'emotional_valence': row[11],
                    'emotional_arousal': row[12],
                    'consolidated_from': row[13] if row[13] is not None else '[]',
                    'consolidated_into': row[14],
                    'storage_strength': row[15] if row[15] is not None else 1.0,
                    'retrieval_weight': row[16] if row[16] is not None else 1.0,
                }

                return episode

        except Exception as e:
            logging.error(f"Failed to get episode by ID: {e}")
            return None

    def _update_activation_score(self, episode_id: str):
        """Boost storage_strength and reset retrieval_weight on access."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE episodes
                    SET access_count = access_count + 1,
                        last_accessed_at = datetime('now'),
                        storage_strength = MIN(storage_strength + 0.1, 10.0),
                        retrieval_weight = MIN(retrieval_weight + 0.3, 1.0)
                    WHERE id = ?
                """, (episode_id,))

                cursor.close()

        except Exception as e:
            logging.error(f"Failed to update activation score: {e}")

    def _analyze_query(self, query_text: str) -> dict:
        """Analyze query text to extract entities, keywords, and noun phrases.

        Uses NLTK POS tagging when available; falls back to whitespace split.
        """
        if not _NLTK_AVAILABLE or not query_text:
            tokens = query_text.split() if query_text else []
            return {'entities': tokens, 'keywords': tokens, 'noun_phrases': []}

        try:
            tokens = word_tokenize(query_text)
            tagged = pos_tag(tokens)

            grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
            parser = RegexpParser(grammar)
            tree = parser.parse(tagged)

            noun_phrases = []
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                phrase = ' '.join(word for word, tag in subtree.leaves())
                noun_phrases.append(phrase)

            stop_tags = {'CC', 'CD', 'DT', 'EX', 'IN', 'MD', 'PDT', 'POS',
                         'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB'}
            content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG',
                            'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
            entity_tags = {'NNP', 'NNPS'}

            entities = [word for word, tag in tagged if tag in entity_tags]
            keywords = [word for word, tag in tagged
                        if tag in content_tags and tag not in stop_tags and len(word) > 1]

            return {'entities': entities, 'keywords': keywords, 'noun_phrases': noun_phrases}

        except Exception as e:
            logging.warning(f"NLTK query analysis failed, falling back: {e}")
            tokens = query_text.split()
            return {'entities': tokens, 'keywords': tokens, 'noun_phrases': []}

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve_episodes(
        self,
        query_text: str,
        radius: float = 0.3,
        *,
        query_embedding: Optional[List[float]] = None,
        return_telemetry: bool = False,
    ):
        """Retrieve episodes via hybrid vector + FTS search.

        Parameters
        ----------
        query_text
            Free text query.
        radius
            Caller-chosen INPUT radius (pre adaptive-shrink). The memory
            skill may already have composed this from narrow/expand
            factors before calling — this method only applies the
            population-aware adaptive shrink on top.
        query_embedding
            Pre-computed query embedding. When provided, the embedding
            is used as-is and no extra call is made to the embedding
            service. Memory skill passes this after computing q_now
            once per recall call so redundancy/drift detection and
            retrieval share the same vector.
        return_telemetry
            When True, returns a ``(episodes, telemetry)`` tuple where
            ``telemetry`` is a dict with keys:
              - ``episode_count``: total live episode count
              - ``input_radius``: the ``radius`` arg as passed in
              - ``adaptive_shrink_divisor``: population-aware divisor
              - ``effective_radius``: the radius actually used for the
                vector distance filter
              - ``vector_candidates``: count of vector candidates
                returned by sqlite-vec (pre radius filter)
              - ``survivors_after_radius``: count remaining after the
                radius threshold is applied
              - ``fts_candidates``: count of FTS matches
              - ``final_rrf_count``: count of RRF-merged candidates
              - ``top_distances``: list of up to 5 nearest vector
                distances (float) for the top survivors
            When False, returns the plain ``List[dict]`` for backwards
            compatibility with existing callers.
        """
        telemetry: dict = {
            'episode_count': 0,
            'input_radius': radius,
            'adaptive_shrink_divisor': 1.0,
            'effective_radius': radius,
            'vector_candidates': 0,
            'survivors_after_radius': 0,
            'fts_candidates': 0,
            'final_rrf_count': 0,
            'top_distances': [],
        }
        try:
            if query_embedding is None:
                query_embedding = self._generate_embedding(query_text)
            episode_count = self._count_episodes()
            adaptive_divisor = 1 + 0.1 * math.log2(episode_count + 2)
            effective_radius = radius / adaptive_divisor

            telemetry['episode_count'] = episode_count
            telemetry['adaptive_shrink_divisor'] = adaptive_divisor
            telemetry['effective_radius'] = effective_radius

            candidates = self._hybrid_retrieve(
                query_embedding, query_text, effective_radius, telemetry=telemetry
            )

            if not candidates:
                if return_telemetry:
                    return [], telemetry
                return []

            query_data = {
                'text': query_text,
                'embedding': query_embedding,
                'emotional_valence': None,
                'emotional_arousal': None,
            }
            ranked = self._rerank_with_composite_score(candidates, query_data, self.weights)
            self._apply_reconsolidation(ranked)

            telemetry['final_rrf_count'] = len(ranked)
            top_dists = []
            for ep in ranked[:5]:
                vd = ep.get('vector_distance')
                if vd is not None:
                    try:
                        top_dists.append(round(float(vd), 4))
                    except (TypeError, ValueError):
                        pass
            telemetry['top_distances'] = top_dists

            if return_telemetry:
                return ranked, telemetry
            return ranked

        except Exception as e:
            logging.error(f"Failed to retrieve episodes: {e}")
            if return_telemetry:
                return [], telemetry
            return []

    def _count_episodes(self) -> int:
        try:
            with self.db_service.connection() as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL").fetchone()[0]
        except Exception:
            return 0

    def _apply_reconsolidation(self, episodes: List[dict]) -> None:
        """Apply memory reconsolidation to retrieved episodes.

        Debounce: skips episodes reconsolidated within the last 10 minutes.
        """
        debounce_minutes = self.config.get('reconsolidation_debounce_minutes', 10)

        store = None
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
        except Exception:
            pass

        for episode in episodes:
            try:
                episode_id = episode.get('id')

                # Debounce: prefer MemoryStore (fast TTL), fall back to DB last_accessed_at
                if store:
                    debounce_key = f"reconsolidation:{episode_id}"
                    if store.get(debounce_key):
                        logging.debug(f"Skipping reconsolidation for episode {episode_id} (debounced)")
                        continue
                    store.set(debounce_key, "1", ex=debounce_minutes * 60)
                else:
                    last_accessed = episode.get('last_accessed_at')
                    if last_accessed:
                        last_dt = parse_utc(last_accessed)
                        if (utc_now() - last_dt).total_seconds() < debounce_minutes * 60:
                            logging.debug(f"Skipping reconsolidation for episode {episode_id} (DB debounced)")
                            continue

                self._update_activation_score(episode_id)

                current_salience = episode.get('salience', 5)
                boost_scaled = self.reconsolidation_boost * 10
                new_salience = min(10, current_salience + boost_scaled)

                now = utc_now()
                new_access_count = episode.get('access_count', 0) + 1

                self.update_episode(episode_id, {
                    'salience': new_salience,
                    'last_accessed_at': now.isoformat(),
                    'access_count': new_access_count,
                })

                episode['salience'] = new_salience
                episode['last_accessed_at'] = now
                episode['access_count'] = new_access_count

                logging.debug(
                    f"Reconsolidated episode {episode_id}: "
                    f"salience {current_salience} -> {new_salience}, "
                    f"access_count -> {new_access_count}"
                )

            except Exception as e:
                logging.warning(f"Failed to reconsolidate episode {episode.get('id')}: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a query."""
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            return emb_service.generate_embedding(text)
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            raise

    def _hybrid_retrieve(self, query_embedding: List[float], query_text: str,
                        effective_radius: float, telemetry: Optional[dict] = None) -> List[dict]:
        try:
            with self.db_service.connection() as conn:
                cursor = DictCursor(conn.cursor())

                vector_ceiling = 200
                vector_query = """
                    SELECT e.id, e.gist, e.salience, e.channel, e.created_at,
                           e.last_accessed_at,
                           COALESCE(e.retrieval_weight, 1.0) AS retrieval_weight,
                           v.distance AS vector_distance,
                           e.emotional_valence, e.emotional_arousal,
                           e.consolidated_into
                    FROM episodes e
                    JOIN episodes_vec v ON v.rowid = e.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                      AND e.deleted_at IS NULL
                    ORDER BY v.distance
                """
                cursor.execute(vector_query, [pack_embedding(query_embedding), vector_ceiling])
                all_vector_results = cursor.fetchall()
                vector_results = [r for r in all_vector_results
                                  if r.get('vector_distance') is not None
                                  and r['vector_distance'] <= effective_radius]

                if telemetry is not None:
                    telemetry['vector_candidates'] = len(all_vector_results)
                    telemetry['survivors_after_radius'] = len(vector_results)

                import re as _re
                fts_safe = _re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text)
                fts_safe = _re.sub(r'\s+', ' ', fts_safe).strip()
                fts_terms = ' '.join(f'"{w}"' for w in fts_safe.split() if w)

                fts_results = []
                if fts_terms:
                    fts_query = """
                        SELECT e.id, e.gist, e.salience, e.channel, e.created_at,
                               e.last_accessed_at,
                               COALESCE(e.retrieval_weight, 1.0) AS retrieval_weight,
                               episodes_fts.rank AS text_rank,
                               e.emotional_valence, e.emotional_arousal,
                               e.consolidated_into
                        FROM episodes_fts
                        JOIN episodes e ON e.rowid = episodes_fts.rowid
                        WHERE episodes_fts MATCH ?
                          AND e.deleted_at IS NULL
                        ORDER BY episodes_fts.rank
                        LIMIT 200
                    """
                    cursor.execute(fts_query, [fts_terms])
                    all_fts = cursor.fetchall()
                    fts_results = [r for r in all_fts
                                   if r.get('text_rank') is not None and r['text_rank'] > -50]

                if telemetry is not None:
                    telemetry['fts_candidates'] = len(fts_results)

                candidates = self._merge_with_rrf(vector_results, fts_results)
                return candidates

        except Exception as e:
            logging.error(f"Hybrid retrieval failed: {e}")
            return []

    def _merge_with_rrf(self, vector_results: list, fts_results: list,
                       k: int = 60) -> List[dict]:
        """Merge vector and full-text results using Reciprocal Rank Fusion."""
        episodes = {}

        for rank, row in enumerate(vector_results, 1):
            episode_id = str(row['id'])
            if episode_id not in episodes:
                episodes[episode_id] = {
                    'id': episode_id,
                    'gist': row['gist'],
                    'salience': row['salience'],
                    'channel': row['channel'],
                    'created_at': row['created_at'],
                    'retrieval_weight': row.get('retrieval_weight', 1.0),
                    'last_accessed_at': row['last_accessed_at'],
                    'emotional_valence': row.get('emotional_valence'),
                    'emotional_arousal': row.get('emotional_arousal'),
                    'consolidated_into': row.get('consolidated_into'),
                    'vector_distance': row.get('vector_distance'),
                    'text_rank': None,
                    'rrf_score': 0
                }
            episodes[episode_id]['rrf_score'] += 1.0 / (k + rank)

        for rank, row in enumerate(fts_results, 1):
            episode_id = str(row['id'])
            if episode_id not in episodes:
                episodes[episode_id] = {
                    'id': episode_id,
                    'gist': row['gist'],
                    'salience': row['salience'],
                    'channel': row['channel'],
                    'created_at': row['created_at'],
                    'retrieval_weight': row.get('retrieval_weight', 1.0),
                    'last_accessed_at': row['last_accessed_at'],
                    'emotional_valence': row.get('emotional_valence'),
                    'emotional_arousal': row.get('emotional_arousal'),
                    'consolidated_into': row.get('consolidated_into'),
                    'vector_distance': None,
                    'text_rank': row.get('text_rank'),
                    'rrf_score': 0
                }
            else:
                episodes[episode_id]['text_rank'] = row.get('text_rank')

            episodes[episode_id]['rrf_score'] += 1.0 / (k + rank)

        candidates = sorted(episodes.values(), key=lambda x: x['rrf_score'], reverse=True)
        return candidates

    def _rerank_with_composite_score(self, candidates: List[dict],
                                     query_data: dict, weights: dict) -> List[dict]:
        """Stage 2: Rerank candidates using composite scoring."""
        scored_episodes = []

        for episode in candidates:
            vector_sim = self._calculate_vector_similarity(
                query_data.get('embedding'), episode.get('vector_distance')
            )
            effective_freshness = self._calculate_effective_freshness(
                episode['salience'], episode['created_at'], episode.get('last_accessed_at')
            )
            retrieval = self._calculate_activation_score(
                episode.get('retrieval_weight', 1.0), effective_freshness
            )

            ep_arousal = episode.get('emotional_arousal')
            arousal_salience = float(ep_arousal) if ep_arousal is not None else 0.5

            query_valence = query_data.get('emotional_valence')
            ep_valence = episode.get('emotional_valence')
            if query_valence is not None and ep_valence is not None:
                emotional_congruence = 1.0 - abs(float(query_valence) - float(ep_valence)) / 2.0
            else:
                emotional_congruence = 0.5

            temporal_proximity = effective_freshness

            composite_score = (
                vector_sim * weights.get('vector_similarity', 4) +
                retrieval * weights.get('retrieval_weight', 3) +
                arousal_salience * 10 * weights.get('arousal_salience', 2) +
                emotional_congruence * 10 * weights.get('emotional_congruence', 1) +
                temporal_proximity * 10 * weights.get('temporal_proximity', 1)
            )

            episode['composite_score'] = composite_score
            episode['score_breakdown'] = {
                'vector_similarity': vector_sim,
                'retrieval_weight': retrieval,
                'arousal_salience': arousal_salience,
                'emotional_congruence': emotional_congruence,
                'temporal_proximity': temporal_proximity,
            }
            scored_episodes.append(episode)

        scored_episodes.sort(key=lambda x: x['composite_score'], reverse=True)
        return scored_episodes

    # ── Scoring helpers ──────────────────────────────────────────────

    def _calculate_vector_similarity(self, query_embedding: List[float],
                                    distance: float) -> float:
        """Convert cosine distance to a similarity score on a 1-10 scale."""
        if distance is None or query_embedding is None:
            return 5.0
        similarity = max(0, 10 - (distance * 5))
        return similarity

    def _calculate_effective_freshness(self, salience: float, created_at: datetime,
                                       last_accessed_at: datetime = None) -> float:
        """Calculate effective freshness using exponential decay."""
        try:
            reference_time = last_accessed_at if last_accessed_at else created_at
            if isinstance(reference_time, str):
                reference_time = parse_utc(reference_time)
            delta_hours = (utc_now() - reference_time).total_seconds() / 3600.0

            effective_decay = self.decay_rate * (1.0 - salience / 10.0)
            freshness = math.exp(-effective_decay * delta_hours)

            return round(max(0.0, min(freshness, 1.0)), 3)

        except Exception as e:
            logging.warning(f"Failed to calculate effective freshness: {e}")
            return 0.5

    def _calculate_activation_score(self, base_activation: float,
                                   effective_freshness: float) -> float:
        """Calculate activation score combining base activation and freshness (1-10 scale)."""
        combined_score = (base_activation * 0.5) + (effective_freshness * 10 * 0.5)
        return min(10.0, max(1.0, combined_score))

    def format_for_prompt(self, episodes: List[dict]) -> str:
        if not episodes:
            return ''

        lines = []
        for episode in episodes:
            gist = episode.get('gist', '')
            if not gist:
                continue

            consolidated_from = episode.get('consolidated_from')
            if isinstance(consolidated_from, str):
                try:
                    consolidated_from = json.loads(consolidated_from)
                except Exception:
                    consolidated_from = []

            if consolidated_from:
                date_range = self._get_consolidated_date_range(consolidated_from, episode)
                lines.append(f"{date_range} — {gist}")
            else:
                created_at = episode.get('created_at', '')
                try:
                    dt = parse_utc(created_at)
                    date_str = dt.strftime('%Y-%m-%d')
                except Exception:
                    date_str = str(created_at)[:10]
                lines.append(f"{date_str} — {gist}")

        return '\n'.join(lines)

    def _get_consolidated_date_range(self, source_ids: list, episode: dict) -> str:
        fallback_created = episode.get('created_at', '')
        try:
            fallback_dt = parse_utc(fallback_created)
            fallback_str = fallback_dt.strftime('%Y-%m-%d')
        except Exception:
            fallback_str = str(fallback_created)[:10]

        if not source_ids:
            return fallback_str

        source_ids = [s for s in source_ids if s != episode.get('id')]
        if not source_ids:
            return fallback_str

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                placeholders = ','.join('?' for _ in source_ids)
                cursor.execute(
                    f"SELECT MIN(created_at), MAX(created_at) FROM episodes WHERE id IN ({placeholders})",
                    source_ids
                )
                row = cursor.fetchone()
                cursor.close()

                if not row or row[0] is None:
                    return fallback_str

                min_dt = parse_utc(row[0])
                max_dt = parse_utc(row[1])
                min_str = min_dt.strftime('%Y-%m-%d')
                max_str = max_dt.strftime('%Y-%m-%d')

                if min_str == max_str:
                    return min_str
                return f"{min_str} to {max_str}"

        except Exception as e:
            logging.warning(f"Failed to get consolidated date range: {e}")
            return fallback_str


# ── Module-level novelty helpers ─────────────────────────────────────────────


def _fetch_novelty_comparison_set(channel: str) -> list[bytes]:
    """Return embedding blobs for the (recent ∪ most-activated) apex episodes.

    Comparison set = dedup(last-100 by created_at) ∪ (top-100 by access_count),
    apex-only (consolidated_into IS NULL), same channel, not deleted.

    Returns a list of raw binary blobs suitable for compute_novelty().
    Returns [] on any failure — novelty will be 1.0 (fully novel).
    """
    from services.database_service import get_shared_db_service
    from services.episodic_constants import NOVELTY_RECENT_LIMIT, NOVELTY_ACTIVATION_LIMIT

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id FROM episodes
                WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL
                ORDER BY created_at DESC LIMIT ?
                """,
                (channel, NOVELTY_RECENT_LIMIT),
            )
            recent_ids = {row[0] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT id FROM episodes
                WHERE channel = ? AND deleted_at IS NULL AND consolidated_into IS NULL
                ORDER BY access_count DESC LIMIT ?
                """,
                (channel, NOVELTY_ACTIVATION_LIMIT),
            )
            top_ids = {row[0] for row in cursor.fetchall()}

            all_ids = list(recent_ids | top_ids)
            if not all_ids:
                cursor.close()
                return []

            placeholders = ','.join('?' * len(all_ids))
            cursor.execute(
                f"""
                SELECT ev.embedding
                FROM episodes_vec ev
                JOIN episodes e ON e.rowid = ev.rowid
                WHERE e.id IN ({placeholders})
                  AND ev.embedding IS NOT NULL
                """,
                all_ids,
            )
            blobs = [row[0] for row in cursor.fetchall() if row[0]]
            cursor.close()

        return blobs

    except Exception as exc:
        logging.warning(f"[NOVELTY] _fetch_novelty_comparison_set failed: {exc}")
        return []


def _unpack_blob(blob: bytes) -> list[float]:
    """Unpack a sqlite-vec binary blob into a list of floats."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f'{n}f', blob))


def _cosine_sim_blobs(blob_a: bytes, blob_b: bytes) -> float:
    """Compute cosine similarity between two sqlite-vec binary blobs.

    Embeddings from EmbeddingService are L2-normalised, so cosine similarity
    is just the dot product of the unpacked float vectors.  Returns 0.0 if
    either blob is malformed or the lengths differ.
    """
    try:
        import numpy as np
        vec_a = np.array(_unpack_blob(blob_a), dtype=np.float32)
        vec_b = np.array(_unpack_blob(blob_b), dtype=np.float32)
        if vec_a.shape != vec_b.shape or vec_a.shape[0] == 0:
            return 0.0
        # Embeddings are pre-normalised — dot product equals cosine sim.
        return float(np.dot(vec_a, vec_b))
    except Exception:
        return 0.0


def find_super_candidates(channel: str) -> list[list[str]]:
    """Return lists of episode IDs that form tight semantic clusters.

    A cluster qualifies when:
      - it contains at least SUPER_EPISODE_MIN_CLUSTER episodes,
      - every pairwise cosine similarity across all members >= SUPER_EPISODE_THRESHOLD.

    Algorithm: greedy pairwise scan over the apex pool (consolidated_into IS NULL,
    deleted_at IS NULL).  First-found wins; episodes already claimed by an emitted
    cluster are not re-evaluated.

    Args:
        channel: The episode channel to cluster.

    Returns:
        List of ID-lists (strings), each list being one tight cluster.
        Clusters are non-overlapping and deterministic (sorted IDs within each list).
    """
    from services.database_service import get_shared_db_service
    from services.episodic_constants import SUPER_EPISODE_THRESHOLD, SUPER_EPISODE_MIN_CLUSTER

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            # Fetch apex episodes with their embeddings.
            cursor.execute(
                """
                SELECT e.id, ev.embedding
                FROM episodes e
                JOIN episodes_vec ev ON ev.rowid = e.rowid
                WHERE e.channel = ?
                  AND e.consolidated_into IS NULL
                  AND e.deleted_at IS NULL
                  AND ev.embedding IS NOT NULL
                ORDER BY e.created_at ASC
                """,
                (channel,),
            )
            rows = cursor.fetchall()
            cursor.close()
    except Exception as exc:
        logging.warning(f"[SUPER_CLUSTER] find_super_candidates query failed: {exc}")
        return []

    if not rows:
        return []

    # Build parallel id/embedding lists.
    ep_ids: list[str] = [str(r[0]) for r in rows]
    ep_embs: list[bytes] = [r[1] for r in rows]
    n = len(ep_ids)

    # Pre-compute all pairwise cosine similarities (upper triangle only).
    # For typical apex pools (<200 episodes) this is fast (~40k ops @ 256-d).
    sims: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sims[(i, j)] = _cosine_sim_blobs(ep_embs[i], ep_embs[j])

    def _pair_sim(i: int, j: int) -> float:
        return sims[(i, j)] if i < j else sims[(j, i)]

    claimed: set[int] = set()
    clusters: list[list[str]] = []

    for i in range(n):
        if i in claimed:
            continue

        # Candidates: all unclaimed j where cosine(i, j) >= threshold.
        candidates = [
            j for j in range(n)
            if j != i and j not in claimed and _pair_sim(i, j) >= SUPER_EPISODE_THRESHOLD
        ]

        if len(candidates) < SUPER_EPISODE_MIN_CLUSTER - 1:
            # Not enough neighbours even before the all-pairs check.
            continue

        # Verify the cluster is fully tight — all pairs within {i} ∪ candidates
        # must also exceed the threshold.
        members = [i] + candidates
        tight = True
        for a in range(len(members)):
            if not tight:
                break
            for b in range(a + 1, len(members)):
                if _pair_sim(members[a], members[b]) < SUPER_EPISODE_THRESHOLD:
                    tight = False
                    break

        if not tight or len(members) < SUPER_EPISODE_MIN_CLUSTER:
            continue

        # Emit cluster with sorted IDs for determinism.
        cluster_ids = sorted(ep_ids[m] for m in members)
        clusters.append(cluster_ids)
        for m in members:
            claimed.add(m)

    return clusters


def compute_novelty(new_embedding, prior_embeddings: list[bytes]) -> float:
    """Return semantic novelty of new_embedding vs the prior comparison set.

    novelty = 1.0 - max(cosine_sim(new, prior) for prior in prior_embeddings)
    Clamped to [0.0, 1.0]. Returns 1.0 (fully novel) if prior_embeddings is empty.

    Args:
        new_embedding: The new embedding as a list/array of floats (L2-normalized).
        prior_embeddings: List of raw binary blobs from _fetch_novelty_comparison_set().

    Returns:
        Float in [0.0, 1.0]. Higher = more novel.
    """
    if not prior_embeddings:
        return 1.0

    try:
        import numpy as np

        if isinstance(new_embedding, bytes):
            new_vec = np.array(_unpack_blob(new_embedding), dtype=np.float32)
        else:
            new_vec = np.array(new_embedding, dtype=np.float32)

        # L2-normalize (embeddings from EmbeddingService are already normalized,
        # but defend against callers who pass raw vectors).
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm

        max_sim = 0.0
        for blob in prior_embeddings:
            try:
                prior_vec = np.array(_unpack_blob(blob), dtype=np.float32)
                if prior_vec.shape != new_vec.shape:
                    continue
                sim = float(np.dot(new_vec, prior_vec))
                if sim > max_sim:
                    max_sim = sim
            except Exception:
                continue

        return max(0.0, min(1.0, 1.0 - max_sim))

    except Exception as exc:
        logging.warning(f"[NOVELTY] compute_novelty failed: {exc}")
        return 1.0
