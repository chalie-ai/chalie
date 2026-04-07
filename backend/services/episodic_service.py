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
(vector + FTS5), composite scoring, and memory reconsolidation.
"""

import json
import logging
import math
import uuid
from datetime import datetime
from typing import Optional, List, Dict

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


_reconsolidation_pending: set = set()


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
            'outcome_relevance': 2,
            'entity_overlap': 3,
            'goal_tag_overlap': 2,
            'arousal_salience': 2,
            'emotional_congruence': 1,
            'temporal_proximity': 1,
        })
        # Freshness decay rate (lambda)
        self.decay_rate = self.config.get('freshness_decay_rate', 0.05)
        # Reconsolidation boost
        self.reconsolidation_boost = self.config.get('reconsolidation_boost', 0.2)

    # ── Storage / CRUD ───────────────────────────────────────────────

    def store_episode(self, episode_data: dict) -> str:
        """Store a new episode in the database.

        Returns:
            UUID of the created episode.

        Raises:
            ValueError: If any required field is missing.
        """
        required_fields = ['intent', 'context', 'action', 'emotion', 'outcome',
                          'gist', 'salience', 'channel']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            episode_id = str(uuid.uuid4())
            embedding = episode_data.get('embedding')

            transcript_id_start = episode_data.get('transcript_id_start')
            transcript_id_end = episode_data.get('transcript_id_end')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Deduplication: skip if >50% transcript ID overlap with existing episode
                if transcript_id_start is not None and transcript_id_end is not None:
                    new_span = transcript_id_end - transcript_id_start + 1
                    cursor.execute("""
                        SELECT id, transcript_id_start, transcript_id_end FROM episodes
                        WHERE transcript_id_start IS NOT NULL
                          AND transcript_id_end IS NOT NULL
                          AND transcript_id_start <= ?
                          AND transcript_id_end >= ?
                          AND deleted_at IS NULL
                    """, (transcript_id_end, transcript_id_start))
                    for overlap_row in cursor.fetchall():
                        existing_start = overlap_row[1]
                        existing_end = overlap_row[2]
                        overlap_start = max(transcript_id_start, existing_start)
                        overlap_end = min(transcript_id_end, existing_end)
                        overlap_count = max(0, overlap_end - overlap_start + 1)
                        if new_span > 0 and overlap_count / new_span > 0.5:
                            cursor.close()
                            logging.info(
                                f"Skipping duplicate episode for transcript range "
                                f"[{transcript_id_start}, {transcript_id_end}] — "
                                f">50% overlap with existing episode {overlap_row[0]}"
                            )
                            return overlap_row[0]

                cursor.execute("""
                    INSERT INTO episodes (
                        id, intent, context, action, emotion, outcome, gist,
                        salience, channel,
                        salience_factors, open_loops,
                        transcript_ids, transcript_id_start, transcript_id_end,
                        entities, goal_tags, emotional_valence, emotional_arousal,
                        consolidated_from, storage_strength, retrieval_weight
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    json.dumps(episode_data['intent']),
                    json.dumps(episode_data['context']) if isinstance(episode_data['context'], (dict, list)) else episode_data['context'],
                    episode_data['action'],
                    json.dumps(episode_data['emotion']),
                    episode_data['outcome'],
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['channel'],
                    json.dumps(episode_data.get('salience_factors', {})),
                    json.dumps(episode_data.get('open_loops', [])),
                    json.dumps(episode_data.get('transcript_ids', [])),
                    transcript_id_start,
                    transcript_id_end,
                    json.dumps(episode_data.get('entities', [])),
                    json.dumps(episode_data.get('goal_tags', [])),
                    episode_data.get('emotional_valence'),
                    episode_data.get('emotional_arousal'),
                    json.dumps(episode_data.get('consolidated_from', [])),
                    episode_data.get('storage_strength', 1.0),
                    episode_data.get('retrieval_weight', 1.0),
                ))

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

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

    def update_episode(self, episode_id: str, updates: dict) -> bool:
        """Update an existing episode with the provided field values.

        Handles JSON serialization for structured fields and optionally
        refreshes the embedding in ``episodes_vec``.

        Returns:
            ``True`` if at least one row was updated, ``False`` on error.
        """
        if not updates:
            return True

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                set_clauses = []
                values = []
                embedding = None

                for key, value in updates.items():
                    if key == 'embedding':
                        embedding = value
                        continue
                    if key in ['intent', 'context', 'emotion', 'salience_factors', 'open_loops']:
                        set_clauses.append(f"{key} = ?")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)

                set_clauses.append("updated_at = datetime('now')")
                values.append(episode_id)

                query = f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)

                rows_updated = cursor.rowcount

                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                cursor.close()

                logging.info(f"Updated episode {episode_id}")
                return rows_updated > 0

        except Exception as e:
            logging.error(f"Failed to update episode: {e}")
            return False

    def soft_delete_episode(self, episode_id: str) -> bool:
        """Soft-delete an episode by setting its ``deleted_at`` timestamp."""
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

    def get_episode_by_id(self, episode_id: str) -> Optional[dict]:
        """Retrieve a single non-deleted episode by its UUID.

        Also triggers a reconsolidation update (access count + activation score).
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, intent, context, action, emotion, outcome, gist,
                           salience, channel,
                           created_at, updated_at, last_accessed_at, access_count,
                           salience_factors, open_loops,
                           transcript_ids, transcript_id_start, transcript_id_end,
                           entities, goal_tags, emotional_valence, emotional_arousal,
                           consolidated_from, storage_strength, retrieval_weight
                    FROM episodes
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                # Update access tracking
                self._update_activation_score(episode_id)
                self.mark_for_reconsolidation(episode_id)

                episode = {
                    'id': str(row[0]),
                    'intent': row[1],
                    'context': row[2],
                    'action': row[3],
                    'emotion': row[4],
                    'outcome': row[5],
                    'gist': row[6],
                    'salience': row[7],
                    'channel': row[8],
                    'created_at': row[9],
                    'updated_at': row[10],
                    'last_accessed_at': row[11],
                    'access_count': row[12],
                    'salience_factors': row[13] if len(row) > 13 else {},
                    'open_loops': row[14] if len(row) > 14 else [],
                    'transcript_ids': row[15] if len(row) > 15 else '[]',
                    'transcript_id_start': row[16] if len(row) > 16 else None,
                    'transcript_id_end': row[17] if len(row) > 17 else None,
                    'entities': row[18] if len(row) > 18 else '[]',
                    'goal_tags': row[19] if len(row) > 19 else '[]',
                    'emotional_valence': row[20] if len(row) > 20 else None,
                    'emotional_arousal': row[21] if len(row) > 21 else None,
                    'consolidated_from': row[22] if len(row) > 22 else '[]',
                    'storage_strength': row[23] if len(row) > 23 else 1.0,
                    'retrieval_weight': row[24] if len(row) > 24 else 1.0,
                }

                return episode

        except Exception as e:
            logging.error(f"Failed to get episode by ID: {e}")
            return None

    def _update_activation_score(self, episode_id: str):
        """Reconsolidation on access: boost storage_strength and reset retrieval_weight."""
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

    def mark_for_reconsolidation(self, episode_id: str) -> None:
        _reconsolidation_pending.add(episode_id)

    def process_pending_reconsolidation(self, current_context: str = "") -> int:
        if not _reconsolidation_pending:
            return 0

        pending = list(_reconsolidation_pending)
        _reconsolidation_pending.clear()

        processed = 0
        for episode_id in pending:
            try:
                self._reconsolidate_episode(episode_id, current_context)
                processed += 1
            except Exception as e:
                logging.warning(f"Reconsolidation failed for {episode_id}: {e}")

        return processed

    def _reconsolidate_episode(self, episode_id: str, current_context: str) -> None:
        if not current_context:
            return

        episode = self.get_episode_by_id(episode_id)
        if not episode:
            return

        gist = episode.get('gist', '')
        outcome = episode.get('outcome', '')
        if not gist and not outcome:
            return

        episode_summary = f"Gist: {gist}\nOutcome: {outcome}"

        try:
            from services.background_llm_queue import create_background_llm_proxy

            llm = create_background_llm_proxy("episodic-reconsolidation")
            system_prompt = (
                "You are a memory reconsolidation assistant. "
                "Respond with JSON only: {\"verdict\": \"contradiction\"|\"extension\"|\"none\", "
                "\"resolved_loops\": [\"...\"], \"correction\": \"...\"}"
            )
            user_message = (
                f"Episode memory:\n{episode_summary}\n\n"
                f"Current context:\n{current_context}\n\n"
                "Does the current context contradict this episode, extend it (resolve open loops), or neither?"
            )
            response = llm.send_message(system_prompt, user_message)
        except Exception as e:
            logging.warning(f"Reconsolidation LLM call failed for {episode_id}: {e}")
            return

        if not response:
            return

        try:
            raw = response.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
        except Exception as e:
            logging.warning(f"Reconsolidation LLM parse failed for {episode_id}: {e}")
            return

        verdict = result.get("verdict", "none")

        if verdict == "contradiction":
            correction = result.get("correction", "")
            self.update_episode(episode_id, {"reliability": "contradicted"})
            logging.info(f"Reconsolidation: episode {episode_id} marked contradicted")

            if correction:
                open_loops = episode.get('open_loops', [])
                if isinstance(open_loops, str):
                    try:
                        open_loops = json.loads(open_loops)
                    except Exception:
                        open_loops = []
                try:
                    self.store_episode({
                        'intent': episode.get('intent', {}),
                        'context': f"Reconsolidation of episode {episode_id}",
                        'action': 'reconsolidation_correction',
                        'emotion': episode.get('emotion', {}),
                        'outcome': correction,
                        'gist': f"Corrected: {correction[:120]}",
                        'salience': episode.get('salience', 5),
                        'channel': episode.get('channel', ''),
                        'open_loops': open_loops,
                        'salience_factors': {'source': 'reconsolidation', 'original_episode': episode_id},
                        'storage_strength': 1.0,
                        'retrieval_weight': 1.0,
                    })
                except Exception as e:
                    logging.warning(f"Failed to store corrected episode for {episode_id}: {e}")

        elif verdict == "extension":
            resolved = result.get("resolved_loops", [])
            if resolved:
                open_loops = episode.get('open_loops', [])
                if isinstance(open_loops, str):
                    try:
                        open_loops = json.loads(open_loops)
                    except Exception:
                        open_loops = []
                updated_loops = [l for l in open_loops if l not in resolved]
                self.update_episode(episode_id, {"open_loops": updated_loops})
                logging.info(
                    f"Reconsolidation: episode {episode_id} extended, "
                    f"resolved {len(resolved)} open loop(s)"
                )

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve_episodes(self, query_text: str, radius: float = 0.3) -> List[dict]:
        try:
            query_analysis = self._analyze_query(query_text)
            query_embedding = self._generate_embedding(query_text)
            episode_count = self._count_episodes()
            effective_radius = radius / (1 + 0.1 * math.log2(episode_count + 2))
            candidates = self._hybrid_retrieve(query_embedding, query_text, effective_radius)

            if not candidates:
                return []

            query_data = {
                'text': query_text,
                'embedding': query_embedding,
                'entities': query_analysis['entities'],
                'goal_tags': [],
                'emotional_valence': None,
                'emotional_arousal': None,
            }
            ranked = self._rerank_with_composite_score(candidates, query_data, self.weights)
            self._apply_reconsolidation(ranked)

            return ranked

        except Exception as e:
            logging.error(f"Failed to retrieve episodes: {e}")
            return []

    def _count_episodes(self) -> int:
        try:
            with self.db_service.connection() as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL").fetchone()[0]
        except Exception:
            return 0

    def _analyze_query(self, query_text: str) -> dict:
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

                if store:
                    debounce_key = f"reconsolidation:{episode_id}"
                    if store.get(debounce_key):
                        logging.debug(f"Skipping reconsolidation for episode {episode_id} (debounced)")
                        continue
                    store.set(debounce_key, "1", ex=debounce_minutes * 60)

                self._update_activation_score(episode_id)
                self.mark_for_reconsolidation(episode_id)

                # Touch-on-read: increment retrieval_count for tool_reflection episodes
                salience_factors = episode.get('salience_factors', {})
                if isinstance(salience_factors, str):
                    import json as _json
                    salience_factors = _json.loads(salience_factors)
                if salience_factors.get('source') == 'tool_reflection':
                    salience_factors['retrieval_count'] = salience_factors.get('retrieval_count', 0) + 1
                    self.update_episode(episode_id, {
                        'salience_factors': salience_factors
                    })

                current_salience = episode.get('salience', 5)
                boost_scaled = self.reconsolidation_boost * 10
                new_salience = min(10, current_salience + boost_scaled)

                self.update_episode(episode_id, {
                    'salience': new_salience
                })

                episode['salience'] = new_salience
                episode['last_accessed_at'] = utc_now()
                episode['access_count'] = episode.get('access_count', 0) + 1

                logging.debug(
                    f"Reconsolidated episode {episode_id}: "
                    f"salience {current_salience} -> {new_salience}, "
                    f"access_count -> {episode['access_count']}"
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
                        effective_radius: float) -> List[dict]:
        try:
            with self.db_service.connection() as conn:
                cursor = DictCursor(conn.cursor())

                vector_ceiling = 200
                vector_query = """
                    SELECT e.id, e.intent, e.context, e.action, e.emotion, e.outcome, e.gist,
                           e.salience, e.channel, e.created_at,
                           e.last_accessed_at, e.salience_factors, e.open_loops,
                           COALESCE(e.retrieval_weight, 1.0) AS retrieval_weight,
                           v.distance AS vector_distance,
                           COALESCE(e.entities, '[]') AS entities,
                           COALESCE(e.goal_tags, '[]') AS goal_tags,
                           e.emotional_valence, e.emotional_arousal
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

                import re as _re
                fts_safe = _re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text)
                fts_safe = _re.sub(r'\s+', ' ', fts_safe).strip()
                fts_terms = ' '.join(f'"{w}"' for w in fts_safe.split() if w)

                fts_results = []
                if fts_terms:
                    fts_query = """
                        SELECT e.id, e.intent, e.context, e.action, e.emotion, e.outcome, e.gist,
                               e.salience, e.channel, e.created_at,
                               e.last_accessed_at, e.salience_factors, e.open_loops,
                               COALESCE(e.retrieval_weight, 1.0) AS retrieval_weight,
                               episodes_fts.rank AS text_rank,
                               COALESCE(e.entities, '[]') AS entities,
                               COALESCE(e.goal_tags, '[]') AS goal_tags,
                               e.emotional_valence, e.emotional_arousal
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
                    'intent': row['intent'],
                    'context': row['context'],
                    'action': row['action'],
                    'emotion': row['emotion'],
                    'outcome': row['outcome'],
                    'gist': row['gist'],
                    'salience': row['salience'],
                    'channel': row['channel'],
                    'created_at': row['created_at'],
                    'retrieval_weight': row.get('retrieval_weight', 1.0),
                    'last_accessed_at': row['last_accessed_at'],
                    'salience_factors': row.get('salience_factors', {}),
                    'open_loops': row.get('open_loops', []),
                    'entities': row.get('entities', '[]'),
                    'goal_tags': row.get('goal_tags', '[]'),
                    'emotional_valence': row.get('emotional_valence'),
                    'emotional_arousal': row.get('emotional_arousal'),
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
                    'intent': row['intent'],
                    'context': row['context'],
                    'action': row['action'],
                    'emotion': row['emotion'],
                    'outcome': row['outcome'],
                    'gist': row['gist'],
                    'salience': row['salience'],
                    'channel': row['channel'],
                    'created_at': row['created_at'],
                    'retrieval_weight': row.get('retrieval_weight', 1.0),
                    'last_accessed_at': row['last_accessed_at'],
                    'salience_factors': row.get('salience_factors', {}),
                    'open_loops': row.get('open_loops', []),
                    'entities': row.get('entities', '[]'),
                    'goal_tags': row.get('goal_tags', '[]'),
                    'emotional_valence': row.get('emotional_valence'),
                    'emotional_arousal': row.get('emotional_arousal'),
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
            outcome_relevance = self._calculate_outcome_relevance(
                query_data['text'], episode['outcome']
            )

            episode_entities = self._parse_json_list(episode.get('entities', '[]'))
            episode_goal_tags = self._parse_json_list(episode.get('goal_tags', '[]'))

            entity_overlap = self._jaccard(query_data.get('entities', []), episode_entities)
            goal_tag_overlap = self._jaccard(query_data.get('goal_tags', []), episode_goal_tags)

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
                outcome_relevance * weights.get('outcome_relevance', 2) +
                entity_overlap * 10 * weights.get('entity_overlap', 3) +
                goal_tag_overlap * 10 * weights.get('goal_tag_overlap', 2) +
                arousal_salience * 10 * weights.get('arousal_salience', 2) +
                emotional_congruence * 10 * weights.get('emotional_congruence', 1) +
                temporal_proximity * 10 * weights.get('temporal_proximity', 1)
            )

            episode['composite_score'] = composite_score
            episode['score_breakdown'] = {
                'vector_similarity': vector_sim,
                'retrieval_weight': retrieval,
                'outcome_relevance': outcome_relevance,
                'entity_overlap': entity_overlap,
                'goal_tag_overlap': goal_tag_overlap,
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

    def _calculate_outcome_relevance(self, query_text: str, outcome: str) -> float:
        """Calculate outcome relevance using keyword overlap (1-10 scale)."""
        if not query_text or not outcome:
            return 5.0

        query_tokens = set(query_text.lower().split())
        outcome_tokens = set(outcome.lower().split())

        intersection = len(query_tokens & outcome_tokens)
        union = len(query_tokens | outcome_tokens)

        overlap = intersection / union if union > 0 else 0
        return 1 + (overlap * 9)

    @staticmethod
    def _jaccard(set_a, set_b) -> float:
        if not set_a or not set_b:
            return 0.0
        a, b = set(set_a), set(set_b)
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _parse_json_list(value) -> list:
        if isinstance(value, list):
            return value
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except Exception:
            return []

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


def backfill_episode_transcript_ids() -> int:
    """Best-effort backfill of transcript_ids for existing episodes that have none.

    Matches episodes to transcript entries by timestamp proximity (±5 minutes,
    same topic). Returns the number of episodes updated.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, channel, created_at
                FROM episodes
                WHERE deleted_at IS NULL
                  AND (transcript_ids IS NULL OR transcript_ids = '[]')
                  AND transcript_id_start IS NULL
            """)
            episodes_to_backfill = cursor.fetchall()

            if not episodes_to_backfill:
                cursor.close()
                return 0

            updated = 0
            for ep_id, channel, created_at_str in episodes_to_backfill:
                try:
                    cursor.execute("""
                        SELECT id FROM transcript
                        WHERE channel = ?
                          AND created_at BETWEEN datetime(?, '-5 minutes')
                                             AND datetime(?, '+5 minutes')
                        ORDER BY id ASC
                    """, (channel, created_at_str, created_at_str))
                    matching = cursor.fetchall()
                    if not matching:
                        continue

                    ids = [r[0] for r in matching]
                    cursor.execute("""
                        UPDATE episodes
                        SET transcript_ids = ?,
                            transcript_id_start = ?,
                            transcript_id_end = ?
                        WHERE id = ?
                    """, (json.dumps(ids), min(ids), max(ids), ep_id))
                    updated += 1
                except Exception:
                    pass

            cursor.close()

        logging.info(f"[EPISODIC] Backfilled transcript_ids for {updated} episodes")
        return updated

    except Exception as e:
        logging.warning(f"[EPISODIC] backfill_episode_transcript_ids failed: {e}")
        return 0
