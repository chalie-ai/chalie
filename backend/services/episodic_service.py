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
import struct
import uuid
from datetime import datetime
from typing import Optional, List, Dict

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
            'topic_overlap': 2,
            'intent_overlap': 3,
            'activation_score': 3,
            'outcome_relevance': 2
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
                          'gist', 'salience', 'freshness', 'topic']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            episode_id = str(uuid.uuid4())
            embedding = episode_data.get('embedding')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO episodes (
                        id, intent, context, action, emotion, outcome, gist,
                        salience, freshness, topic, exchange_id,
                        activation_score, salience_factors, open_loops
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    json.dumps(episode_data['intent']),
                    json.dumps(episode_data['context']),
                    episode_data['action'],
                    json.dumps(episode_data['emotion']),
                    episode_data['outcome'],
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['freshness'],
                    episode_data['topic'],
                    episode_data.get('exchange_id'),
                    1.0,
                    json.dumps(episode_data.get('salience_factors', {})),
                    json.dumps(episode_data.get('open_loops', []))
                ))

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                cursor.close()

                logging.info(f"Stored episode {episode_id} for topic '{episode_data['topic']}'")

                # Notify curiosity pursuit service for conversational reinforcement
                try:
                    from services.curiosity_pursuit_service import CuriosityPursuitService
                    CuriosityPursuitService().on_new_episode(episode_data)
                except Exception:
                    pass  # Non-fatal

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
                           salience, freshness, topic, exchange_id,
                           created_at, updated_at, last_accessed_at, access_count,
                           activation_score, salience_factors, open_loops
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
                    'intent': row[1],
                    'context': row[2],
                    'action': row[3],
                    'emotion': row[4],
                    'outcome': row[5],
                    'gist': row[6],
                    'salience': row[7],
                    'freshness': row[8],
                    'topic': row[9],
                    'exchange_id': row[10],
                    'created_at': row[11],
                    'updated_at': row[12],
                    'last_accessed_at': row[13],
                    'access_count': row[14],
                    'activation_score': row[15],
                    'salience_factors': row[16] if len(row) > 16 else {},
                    'open_loops': row[17] if len(row) > 17 else []
                }

                return episode

        except Exception as e:
            logging.error(f"Failed to get episode by ID: {e}")
            return None

    def _update_activation_score(self, episode_id: str):
        """Update activation score based on access frequency and recency (ACT-R model)."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE episodes
                    SET access_count = access_count + 1,
                        last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (episode_id,))

                cursor.execute("""
                    UPDATE episodes
                    SET activation_score = 1.0
                        + (access_count * 0.1)
                        + CASE
                            WHEN last_accessed_at IS NOT NULL THEN
                                (1.0 / (1.0 + (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', last_accessed_at) AS REAL)) / 86400.0))
                            ELSE 0
                          END
                    WHERE id = ?
                """, (episode_id,))

                cursor.close()

        except Exception as e:
            logging.error(f"Failed to update activation score: {e}")

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve_episodes(self, query_text: str, topic: str = None,
                         intent: str = None, limit: int = 3,
                         weights: dict = None, semantic_concepts: List[Dict] = None,
                         query_embedding=None) -> List[dict]:
        """Retrieve relevant episodes using hybrid search and composite scoring."""
        try:
            scoring_weights = weights or self.weights

            if query_embedding is None:
                query_embedding = self._generate_embedding(query_text)

            prefilter_limit = self.config.get('prefilter_candidates', 50)
            candidates = self._hybrid_retrieve(
                query_embedding, query_text, topic, prefilter_limit
            )

            if not candidates:
                logging.info("No candidate episodes found")
                return []

            query_data = {
                'text': query_text,
                'topic': topic,
                'intent': intent,
                'embedding': query_embedding
            }
            ranked_episodes = self._rerank_with_composite_score(
                candidates, query_data, scoring_weights, semantic_concepts=semantic_concepts
            )

            top_episodes = ranked_episodes[:limit]
            self._apply_reconsolidation(top_episodes)

            return top_episodes

        except Exception as e:
            logging.error(f"Failed to retrieve episodes: {e}")
            return []

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
                        topic: str, limit: int) -> List[dict]:
        """Stage 1: Hybrid prefilter using vector similarity + full-text search."""
        try:
            with self.db_service.connection() as conn:
                cursor = DictCursor(conn.cursor())

                vector_query = """
                    SELECT e.id, e.intent, e.context, e.action, e.emotion, e.outcome, e.gist,
                           e.salience, e.freshness, e.topic, e.created_at, e.activation_score,
                           e.last_accessed_at, e.salience_factors, e.open_loops,
                           COALESCE(e.reliability, 'reliable') AS reliability,
                           v.distance AS vector_distance
                    FROM episodes e
                    JOIN episodes_vec v ON v.rowid = e.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                      AND e.deleted_at IS NULL
                """
                vector_params = [pack_embedding(query_embedding), limit]

                if topic:
                    vector_query += " AND e.topic = ?"
                    vector_params.append(topic)

                vector_query += " ORDER BY v.distance"

                cursor.execute(vector_query, vector_params)
                vector_results = cursor.fetchall()

                fts_query = """
                    SELECT e.id, e.intent, e.context, e.action, e.emotion, e.outcome, e.gist,
                           e.salience, e.freshness, e.topic, e.created_at, e.activation_score,
                           e.last_accessed_at, e.salience_factors, e.open_loops,
                           COALESCE(e.reliability, 'reliable') AS reliability,
                           episodes_fts.rank AS text_rank
                    FROM episodes_fts
                    JOIN episodes e ON e.rowid = episodes_fts.rowid
                    WHERE episodes_fts MATCH ?
                      AND e.deleted_at IS NULL
                """
                import re as _re
                fts_safe = _re.sub(r'[:\(\)\*\^"\\?,\'.]', ' ', query_text)
                fts_safe = _re.sub(r'\s+', ' ', fts_safe).strip()
                fts_params = [fts_safe or '*']

                if topic:
                    fts_query += " AND e.topic = ?"
                    fts_params.append(topic)

                fts_query += " ORDER BY episodes_fts.rank LIMIT ?"
                fts_params.append(limit)

                cursor.execute(fts_query, fts_params)
                fts_results = cursor.fetchall()

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
                    'freshness': row['freshness'],
                    'topic': row['topic'],
                    'created_at': row['created_at'],
                    'activation_score': row['activation_score'],
                    'last_accessed_at': row['last_accessed_at'],
                    'salience_factors': row.get('salience_factors', {}),
                    'open_loops': row.get('open_loops', []),
                    'reliability': row.get('reliability', 'reliable'),
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
                    'freshness': row['freshness'],
                    'topic': row['topic'],
                    'created_at': row['created_at'],
                    'activation_score': row['activation_score'],
                    'last_accessed_at': row['last_accessed_at'],
                    'salience_factors': row.get('salience_factors', {}),
                    'open_loops': row.get('open_loops', []),
                    'reliability': row.get('reliability', 'reliable'),
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
                                     query_data: dict, weights: dict,
                                     semantic_concepts: List[Dict] = None) -> List[dict]:
        """Stage 2: Rerank candidates using composite scoring."""
        scored_episodes = []

        for episode in candidates:
            vector_sim = self._calculate_vector_similarity(
                query_data.get('embedding'), episode.get('vector_distance')
            )
            topic_overlap = self._calculate_topic_overlap(
                query_data.get('topic'), episode['topic']
            )
            intent_overlap = self._calculate_intent_overlap(
                query_data.get('intent'), episode['intent']
            )
            effective_freshness = self._calculate_effective_freshness(
                episode['salience'], episode['created_at'], episode.get('last_accessed_at')
            )
            activation = self._calculate_activation_score(
                episode['activation_score'], effective_freshness
            )
            outcome_relevance = self._calculate_outcome_relevance(
                query_data['text'], episode['outcome']
            )

            semantic_boost = 0
            if semantic_concepts:
                semantic_boost = self._calculate_semantic_boost(
                    episode, semantic_concepts
                )

            composite_score = (
                vector_sim * weights.get('vector_similarity', 4) +
                topic_overlap * weights.get('topic_overlap', 2) +
                intent_overlap * weights.get('intent_overlap', 3) +
                activation * weights.get('activation_score', 3) +
                outcome_relevance * weights.get('outcome_relevance', 2) +
                semantic_boost * weights.get('semantic_boost', 2)
            )

            episode['composite_score'] = composite_score
            episode['score_breakdown'] = {
                'vector_similarity': vector_sim,
                'topic_overlap': topic_overlap,
                'intent_overlap': intent_overlap,
                'activation': activation,
                'outcome_relevance': outcome_relevance,
                'semantic_boost': semantic_boost
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

    def _calculate_topic_overlap(self, query_topic: str, episode_topic: str) -> float:
        """Calculate topic overlap score (2-10 scale)."""
        if not query_topic:
            return 5.0

        if query_topic.lower() == episode_topic.lower():
            return 10.0
        elif query_topic.lower() in episode_topic.lower() or episode_topic.lower() in query_topic.lower():
            return 7.0
        else:
            return 2.0

    def _calculate_intent_overlap(self, query_intent: str, episode_intent: dict) -> float:
        """Calculate intent overlap score (1-10 scale)."""
        if not query_intent:
            return 5.0

        if isinstance(episode_intent, dict):
            intent_type = episode_intent.get('type', '')
            intent_direction = episode_intent.get('direction', '')
            episode_intent_str = f"{intent_type} {intent_direction}"
        else:
            episode_intent_str = str(episode_intent)

        query_tokens = set(query_intent.lower().split())
        episode_tokens = set(episode_intent_str.lower().split())

        if not query_tokens or not episode_tokens:
            return 5.0

        intersection = len(query_tokens & episode_tokens)
        union = len(query_tokens | episode_tokens)

        jaccard = intersection / union if union > 0 else 0
        return 1 + (jaccard * 9)

    def _calculate_effective_freshness(self, salience: float, created_at: datetime,
                                       last_accessed_at: datetime = None) -> float:
        """Calculate effective freshness using exponential decay."""
        try:
            reference_time = last_accessed_at if last_accessed_at else created_at
            if isinstance(reference_time, str):
                reference_time = parse_utc(reference_time)
            delta_hours = (utc_now() - reference_time).total_seconds() / 3600.0

            effective_decay = self.decay_rate * (1.0 - salience)
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

    def _calculate_semantic_boost(self, episode: dict, concepts: List[Dict]) -> float:
        """Calculate boost (0-10 scale) based on concept mention in episode."""
        if not concepts:
            return 0.0

        episode_text = f"{episode.get('gist', '')} {episode.get('outcome', '')}".lower()

        matched_concepts = 0
        for concept in concepts:
            concept_name = concept.get('concept_name', concept.get('name', '')).lower()
            if concept_name in episode_text:
                matched_concepts += 1

        boost = min(10, (matched_concepts / len(concepts)) * 10) if concepts else 0
        return boost
