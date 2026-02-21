# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episodic Retrieval Service - Episode query and hybrid search.
Responsibility: Retrieval layer only (SRP).
"""

import math
from datetime import datetime
from typing import Optional, List, Dict
from services.database_service import DatabaseService
from services.episodic_storage_service import EpisodicStorageService
import logging


class EpisodicRetrievalService:
    """Manages episode retrieval with hybrid search and scoring."""

    def __init__(self, database_service: DatabaseService, config: dict = None):
        """
        Initialize retrieval service.

        Args:
            database_service: DatabaseService instance for connection management
            config: Optional config dict with weights
        """
        self.db_service = database_service
        self.storage_service = EpisodicStorageService(database_service)
        self.config = config or {}
        self.embedding_dimensions = self.config.get('embedding_dimensions', 256)
        self.weights = self.config.get('inference_weights', {
            'vector_similarity': 4,
            'topic_overlap': 2,
            'intent_overlap': 3,
            'activation_score': 3,
            'outcome_relevance': 2
        })
        # Freshness decay rate (λ)
        self.decay_rate = self.config.get('freshness_decay_rate', 0.05)
        # Reconsolidation boost
        self.reconsolidation_boost = self.config.get('reconsolidation_boost', 0.2)

    def retrieve_episodes(self, query_text: str, topic: str = None,
                         intent: str = None, limit: int = 3,
                         weights: dict = None, semantic_concepts: List[Dict] = None) -> List[dict]:
        """
        Retrieve relevant episodes using hybrid search and composite scoring.

        Args:
            query_text: User query text for semantic search
            topic: Optional topic filter
            intent: Optional intent filter
            limit: Number of episodes to return
            weights: Optional custom weights for scoring factors

        Returns:
            List of episode dicts sorted by composite score
        """
        try:
            # Use custom weights if provided
            scoring_weights = weights or self.weights

            # Generate embedding for query
            query_embedding = self._generate_embedding(query_text)

            # Stage 1: Hybrid prefilter (fast)
            prefilter_limit = self.config.get('prefilter_candidates', 50)
            candidates = self._hybrid_retrieve(
                query_embedding, query_text, topic, prefilter_limit
            )

            if not candidates:
                logging.info("No candidate episodes found")
                return []

            # Stage 2: Composite reranking (precise)
            query_data = {
                'text': query_text,
                'topic': topic,
                'intent': intent,
                'embedding': query_embedding
            }
            ranked_episodes = self._rerank_with_composite_score(
                candidates, query_data, scoring_weights, semantic_concepts=semantic_concepts
            )

            # Get top N episodes
            top_episodes = ranked_episodes[:limit]

            # Stage 3: Memory reconsolidation (strengthen retrieved memories)
            self._apply_reconsolidation(top_episodes)

            return top_episodes

        except Exception as e:
            logging.error(f"Failed to retrieve episodes: {e}")
            return []

    def _apply_reconsolidation(self, episodes: List[dict]) -> None:
        """
        Apply memory reconsolidation to retrieved episodes.

        Retrieved memories get strengthened (salience boost), their
        last_accessed_at timestamp is updated, and access_count is incremented.

        Debounce: skips episodes reconsolidated within the last 10 minutes
        to prevent the same episodes being UPDATE'd 15+ times per request chain.

        Args:
            episodes: List of episode dicts to reconsolidate
        """
        debounce_minutes = self.config.get('reconsolidation_debounce_minutes', 10)

        # Redis-based debounce check
        redis_conn = None
        try:
            from services.redis_client import RedisClientService
            redis_conn = RedisClientService.create_connection()
        except Exception:
            pass

        for episode in episodes:
            try:
                episode_id = episode.get('id')

                # Debounce: skip if reconsolidated recently
                if redis_conn:
                    debounce_key = f"reconsolidation:{episode_id}"
                    if redis_conn.get(debounce_key):
                        logging.debug(f"Skipping reconsolidation for episode {episode_id} (debounced)")
                        continue
                    # Set debounce flag
                    redis_conn.set(debounce_key, "1", ex=debounce_minutes * 60)

                # Use storage service's internal activation score update
                # This increments access_count and updates last_accessed_at
                self.storage_service._update_activation_score(episode_id)

                # Touch-on-read: increment retrieval_count for tool_reflection episodes
                salience_factors = episode.get('salience_factors', {})
                if isinstance(salience_factors, str):
                    import json as _json
                    salience_factors = _json.loads(salience_factors)
                if salience_factors.get('source') == 'tool_reflection':
                    salience_factors['retrieval_count'] = salience_factors.get('retrieval_count', 0) + 1
                    self.storage_service.update_episode(episode_id, {
                        'salience_factors': salience_factors
                    })

                # Then apply salience boost (1-10 scale)
                current_salience = episode.get('salience', 5)
                # Scale reconsolidation_boost from 0-1 range to 1-10 range
                boost_scaled = self.reconsolidation_boost * 10
                new_salience = min(10, current_salience + boost_scaled)

                self.storage_service.update_episode(episode_id, {
                    'salience': new_salience
                })

                # Update in-memory episode dict for return value
                episode['salience'] = new_salience
                episode['last_accessed_at'] = datetime.now()
                episode['access_count'] = episode.get('access_count', 0) + 1

                logging.debug(
                    f"Reconsolidated episode {episode_id}: "
                    f"salience {current_salience} → {new_salience}, "
                    f"access_count → {episode['access_count']}"
                )

            except Exception as e:
                logging.warning(f"Failed to reconsolidate episode {episode.get('id')}: {e}")
                # Continue with other episodes even if one fails

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector with dimensions matching config

        Raises:
            Exception if embedding generation fails
        """
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            return emb_service.generate_embedding(text)

        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            raise

    def _hybrid_retrieve(self, query_embedding: List[float], query_text: str,
                        topic: str, limit: int) -> List[dict]:
        """
        Stage 1: Hybrid prefilter using vector similarity + full-text search.

        Args:
            query_embedding: Query embedding vector
            query_text: Query text for full-text search
            topic: Optional topic filter
            limit: Number of candidates to return

        Returns:
            List of candidate episodes with similarity scores
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Set HNSW ef_search parameter for query
                ef_search = self.config.get('hnsw_ef_search', 100)
                cursor.execute(f"SET hnsw.ef_search = {ef_search}")

                # Vector similarity search (include last_accessed_at for freshness calculation)
                vector_query = """
                    SELECT id, intent, context, action, emotion, outcome, gist,
                           salience, freshness, topic, created_at, activation_score,
                           last_accessed_at, salience_factors, open_loops,
                           (embedding <=> %s::vector) AS vector_distance
                    FROM episodes
                    WHERE deleted_at IS NULL
                """
                vector_params = [query_embedding]

                if topic:
                    vector_query += " AND topic = %s"
                    vector_params.append(topic)

                vector_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
                vector_params.extend([query_embedding, limit])

                cursor.execute(vector_query, vector_params)
                vector_results = cursor.fetchall()

                # Full-text search on gist and action (intent is now JSONB)
                fts_query = """
                    SELECT id, intent, context, action, emotion, outcome, gist,
                           salience, freshness, topic, created_at, activation_score,
                           last_accessed_at, salience_factors, open_loops,
                           ts_rank(to_tsvector('english', gist || ' ' || action),
                                   plainto_tsquery('english', %s)) AS text_rank
                    FROM episodes
                    WHERE deleted_at IS NULL
                      AND (to_tsvector('english', gist || ' ' || action) @@
                           plainto_tsquery('english', %s))
                """
                fts_params = [query_text, query_text]

                if topic:
                    fts_query += " AND topic = %s"
                    fts_params.append(topic)

                fts_query += " ORDER BY text_rank DESC LIMIT %s"
                fts_params.append(limit)

                cursor.execute(fts_query, fts_params)
                fts_results = cursor.fetchall()

                cursor.close()

                # Merge results using Reciprocal Rank Fusion (RRF)
                candidates = self._merge_with_rrf(vector_results, fts_results)

                return candidates

        except Exception as e:
            logging.error(f"Hybrid retrieval failed: {e}")
            return []

    def _merge_with_rrf(self, vector_results: list, fts_results: list,
                       k: int = 60) -> List[dict]:
        """
        Merge vector and full-text results using Reciprocal Rank Fusion.

        Args:
            vector_results: Results from vector search
            fts_results: Results from full-text search
            k: RRF constant (default 60)

        Returns:
            Merged list of candidates
        """
        # Build episode dict from results
        episodes = {}

        # Process vector results
        for rank, row in enumerate(vector_results, 1):
            episode_id = str(row[0])
            if episode_id not in episodes:
                episodes[episode_id] = {
                    'id': episode_id,
                    'intent': row[1],
                    'context': row[2],
                    'action': row[3],
                    'emotion': row[4],
                    'outcome': row[5],
                    'gist': row[6],
                    'salience': row[7],
                    'freshness': row[8],
                    'topic': row[9],
                    'created_at': row[10],
                    'activation_score': row[11],
                    'last_accessed_at': row[12],
                    'salience_factors': row[13] if len(row) > 13 else {},
                    'open_loops': row[14] if len(row) > 14 else [],
                    'vector_distance': row[15] if len(row) > 15 else None,
                    'text_rank': None,
                    'rrf_score': 0
                }
            episodes[episode_id]['rrf_score'] += 1.0 / (k + rank)

        # Process FTS results
        for rank, row in enumerate(fts_results, 1):
            episode_id = str(row[0])
            if episode_id not in episodes:
                episodes[episode_id] = {
                    'id': episode_id,
                    'intent': row[1],
                    'context': row[2],
                    'action': row[3],
                    'emotion': row[4],
                    'outcome': row[5],
                    'gist': row[6],
                    'salience': row[7],
                    'freshness': row[8],
                    'topic': row[9],
                    'created_at': row[10],
                    'activation_score': row[11],
                    'last_accessed_at': row[12],
                    'salience_factors': row[13] if len(row) > 13 else {},
                    'open_loops': row[14] if len(row) > 14 else [],
                    'vector_distance': None,
                    'text_rank': row[15] if len(row) > 15 else None,
                    'rrf_score': 0
                }
            else:
                episodes[episode_id]['text_rank'] = row[15] if len(row) > 15 else None

            episodes[episode_id]['rrf_score'] += 1.0 / (k + rank)

        # Sort by RRF score
        candidates = sorted(episodes.values(), key=lambda x: x['rrf_score'], reverse=True)
        return candidates

    def _rerank_with_composite_score(self, candidates: List[dict],
                                     query_data: dict, weights: dict,
                                     semantic_concepts: List[Dict] = None) -> List[dict]:
        """
        Stage 2: Rerank candidates using composite scoring.

        Args:
            candidates: List of candidate episodes from prefilter
            query_data: Dict with query text, topic, intent, embedding
            weights: Weight dict for scoring factors

        Returns:
            Reranked list of episodes
        """
        scored_episodes = []

        for episode in candidates:
            # Calculate individual scores (1-10 scale)
            vector_sim = self._calculate_vector_similarity(
                query_data.get('embedding'), episode.get('vector_distance')
            )
            topic_overlap = self._calculate_topic_overlap(
                query_data.get('topic'), episode['topic']
            )
            intent_overlap = self._calculate_intent_overlap(
                query_data.get('intent'), episode['intent']
            )
            # Calculate effective freshness dynamically
            effective_freshness = self._calculate_effective_freshness(
                episode['salience'], episode['created_at'], episode.get('last_accessed_at')
            )
            activation = self._calculate_activation_score(
                episode['activation_score'], effective_freshness
            )
            outcome_relevance = self._calculate_outcome_relevance(
                query_data['text'], episode['outcome']
            )

            # Calculate semantic concept match boost (0-10 scale)
            semantic_boost = 0
            if semantic_concepts:
                semantic_boost = self._calculate_semantic_boost(
                    episode,
                    semantic_concepts
                )

            # Weighted composite score (add semantic_boost)
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

        # Sort by composite score
        scored_episodes.sort(key=lambda x: x['composite_score'], reverse=True)
        return scored_episodes

    def _calculate_vector_similarity(self, query_embedding: List[float],
                                    distance: float) -> float:
        """Convert cosine distance to similarity score (1-10 scale)."""
        if distance is None or query_embedding is None:
            return 5.0  # Neutral score if no vector data
        # Cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: distance 0 -> score 10, distance 1 -> score 5
        similarity = max(0, 10 - (distance * 5))
        return similarity

    def _calculate_topic_overlap(self, query_topic: str, episode_topic: str) -> float:
        """Calculate topic overlap score (1-10 scale)."""
        if not query_topic:
            return 5.0  # Neutral score if no topic filter

        if query_topic.lower() == episode_topic.lower():
            return 10.0  # Exact match
        elif query_topic.lower() in episode_topic.lower() or episode_topic.lower() in query_topic.lower():
            return 7.0  # Partial match
        else:
            return 2.0  # No match

    def _calculate_intent_overlap(self, query_intent: str, episode_intent: dict) -> float:
        """
        Calculate intent overlap score (1-10 scale).

        Intent is now JSONB: {"type": "exploration|...", "direction": "open-ended|..."}
        """
        if not query_intent:
            return 5.0  # Neutral score if no intent provided

        # Handle JSONB intent structure
        if isinstance(episode_intent, dict):
            intent_type = episode_intent.get('type', '')
            intent_direction = episode_intent.get('direction', '')
            episode_intent_str = f"{intent_type} {intent_direction}"
        else:
            episode_intent_str = str(episode_intent)

        # Tokenize and compute Jaccard similarity
        query_tokens = set(query_intent.lower().split())
        episode_tokens = set(episode_intent_str.lower().split())

        if not query_tokens or not episode_tokens:
            return 5.0

        intersection = len(query_tokens & episode_tokens)
        union = len(query_tokens | episode_tokens)

        jaccard = intersection / union if union > 0 else 0
        return 1 + (jaccard * 9)  # Scale to 1-10

    def _calculate_effective_freshness(self, salience: float, created_at: datetime,
                                       last_accessed_at: datetime = None) -> float:
        """
        Calculate effective freshness using exponential decay.

        Formula: freshness = e^(-effective_decay × Δt_hours)
        Where: effective_decay = decay_rate × (1 - salience)

        High salience slows decay (salience-modulated decay rate).

        Args:
            salience: Salience score in [0.1, 1.0]
            created_at: Episode creation timestamp
            last_accessed_at: Last access timestamp (None if never accessed)

        Returns:
            Effective freshness in [0.0, 1.0], rounded to 3 decimals
        """
        try:
            # Time since last access in hours
            reference_time = last_accessed_at if last_accessed_at else created_at
            delta_hours = (datetime.now() - reference_time).total_seconds() / 3600.0

            # Salience slows decay (high salience = slower decay)
            effective_decay = self.decay_rate * (1.0 - salience)

            # Exponential decay
            freshness = math.exp(-effective_decay * delta_hours)

            return round(max(0.0, min(freshness, 1.0)), 3)

        except Exception as e:
            logging.warning(f"Failed to calculate effective freshness: {e}")
            return 0.5

    def _calculate_activation_score(self, base_activation: float,
                                   effective_freshness: float) -> float:
        """
        Calculate activation score combining base activation and freshness (1-10 scale).

        Args:
            base_activation: Base activation from access patterns
            effective_freshness: Effective freshness in [0, 1]

        Returns:
            Activation score in [1, 10]
        """
        # Combine activation and freshness (weight both equally)
        combined_score = (base_activation * 0.5) + (effective_freshness * 10 * 0.5)

        # Scale to 1-10
        return min(10.0, max(1.0, combined_score))

    def _calculate_outcome_relevance(self, query_text: str, outcome: str) -> float:
        """Calculate outcome relevance using keyword matching (1-10 scale)."""
        if not query_text or not outcome:
            return 5.0

        # Simple keyword overlap
        query_tokens = set(query_text.lower().split())
        outcome_tokens = set(outcome.lower().split())

        intersection = len(query_tokens & outcome_tokens)
        union = len(query_tokens | outcome_tokens)

        overlap = intersection / union if union > 0 else 0
        return 1 + (overlap * 9)  # Scale to 1-10

    def _calculate_semantic_boost(self, episode: dict, concepts: List[Dict]) -> float:
        """
        Calculate boost (0-10 scale) based on concept mention in episode.

        Strategy: Keyword matching on concept names in episode gist/outcome
        """
        if not concepts:
            return 0.0

        episode_text = f"{episode.get('gist', '')} {episode.get('outcome', '')}".lower()

        matched_concepts = 0
        for concept in concepts:
            concept_name = concept.get('concept_name', concept.get('name', '')).lower()
            # Check if concept name appears in episode
            if concept_name in episode_text:
                matched_concepts += 1

        # Scale: max 5 concepts -> 0 to 10
        boost = min(10, (matched_concepts / len(concepts)) * 10) if concepts else 0
        return boost
