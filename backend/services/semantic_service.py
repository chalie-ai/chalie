# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Semantic Service — Unified storage and retrieval for semantic concepts.

Combines the former SemanticStorageService and SemanticRetrievalService into
a single cohesive service. Provides concept persistence, hybrid search
(vector similarity + graph properties), spreading activation, and
confidence-filtered retrieval.
"""

import json
import logging
import math
import random
import struct
import uuid
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np

from services.database_service import DatabaseService
from services.embedding_utils import pack_embedding


def _json_default(obj):
    """Handle UUID and other non-serializable types."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SemanticService:
    """Manages semantic concept storage, retrieval, and graph operations."""

    def __init__(self, db_service, embedding_service=None,
                 storage_service=None, config: dict = None):
        """Initialize the semantic service.

        Args:
            db_service: DatabaseService instance for connection management.
            embedding_service: Optional embedding service for vector operations.
            storage_service: Ignored (backward compat with old SemanticRetrievalService
                which accepted a separate storage_service). This class is the storage.
            config: Optional config dict. Loaded from agent config when None.
        """
        self.db_service = db_service

        if embedding_service is None:
            try:
                from services.embedding_service import get_embedding_service
                self.embedding_service = get_embedding_service()
            except Exception:
                self.embedding_service = None
        else:
            self.embedding_service = embedding_service

        if config is not None:
            self.config = config
        else:
            try:
                from services.config_service import ConfigService
                self.config = ConfigService.get_agent_config("semantic-memory")
            except Exception:
                self.config = {}

        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.4)

    # ── Storage / CRUD ───────────────────────────────────────────────

    def _store_embedding(self, conn, concept_id: str, embedding):
        """Store embedding in the companion vec table."""
        try:
            blob = pack_embedding(embedding)
            if blob is None:
                return

            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM semantic_concepts WHERE id = ?", (concept_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO semantic_concepts_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )
            cursor.close()
        except Exception as e:
            logging.warning(f"Failed to store concept embedding: {e}")

    def store_concept(self, concept_data: dict) -> str:
        """Store a new concept in the database.

        Args:
            concept_data: Concept dict with fields:
                         concept_name, concept_type, definition, embedding,
                         abstraction_level, domain, confidence, source_episodes

        Returns:
            UUID of the created concept.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ['concept_name', 'concept_type', 'definition']
        for field in required_fields:
            if field not in concept_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            concept_id = str(uuid.uuid4())
            embedding = concept_data.get('embedding')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO semantic_concepts (
                        id, concept_name, concept_type, definition,
                        abstraction_level, domain, confidence, source_episodes,
                        context_constraints, examples
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept_id,
                    concept_data['concept_name'],
                    concept_data['concept_type'],
                    concept_data['definition'],
                    concept_data.get('abstraction_level', 3),
                    concept_data.get('domain'),
                    concept_data.get('confidence', 0.5),
                    json.dumps(concept_data.get('source_episodes', []), default=_json_default),
                    json.dumps(concept_data.get('context_constraints', {}), default=_json_default),
                    json.dumps(concept_data.get('examples', []), default=_json_default)
                ))

                if embedding is not None:
                    self._store_embedding(conn, concept_id, embedding)

                cursor.close()

                logging.info(f"Stored concept {concept_id}: '{concept_data['concept_name']}' ({concept_data['concept_type']})")
                return str(concept_id)

        except Exception as e:
            logging.error(f"Failed to store concept: {e}")
            raise

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """Retrieve a concept by ID."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, concept_name, concept_type, definition,
                        abstraction_level, domain, strength, activation_score,
                        access_count, consolidation_count, confidence, source_episodes,
                        verification_status, context_constraints, examples,
                        first_learned_at, last_accessed_at, last_reinforced_at,
                        utility_score, decay_resistance, created_at, updated_at
                    FROM semantic_concepts
                    WHERE id = ? AND deleted_at IS NULL
                """, (concept_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                return {
                    'id': str(row[0]),
                    'concept_name': row[1],
                    'concept_type': row[2],
                    'definition': row[3],
                    'embedding': None,
                    'abstraction_level': row[4],
                    'domain': row[5],
                    'strength': row[6],
                    'activation_score': row[7],
                    'access_count': row[8],
                    'consolidation_count': row[9],
                    'confidence': row[10],
                    'source_episodes': row[11],
                    'verification_status': row[12],
                    'context_constraints': row[13],
                    'examples': row[14],
                    'first_learned_at': row[15],
                    'last_accessed_at': row[16],
                    'last_reinforced_at': row[17],
                    'utility_score': row[18],
                    'decay_resistance': row[19],
                    'created_at': row[20],
                    'updated_at': row[21]
                }

        except Exception as e:
            logging.error(f"Failed to get concept {concept_id}: {e}")
            return None

    def strengthen_concept(self, concept_id: str, episode_id: str) -> bool:
        """Strengthen a concept (increase strength by reinforcement factor)."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT strength, source_episodes, consolidation_count, decay_resistance
                    FROM semantic_concepts
                    WHERE id = ? AND deleted_at IS NULL
                """, (concept_id,))

                row = cursor.fetchone()
                if not row:
                    logging.warning(f"Concept {concept_id} not found for strengthening")
                    cursor.close()
                    return False

                current_strength, source_episodes_raw, consolidation_count, decay_resistance = row

                if isinstance(source_episodes_raw, str):
                    source_episodes = json.loads(source_episodes_raw)
                else:
                    source_episodes = source_episodes_raw or []

                new_strength = min(10.0, current_strength * 1.1)

                if episode_id not in source_episodes:
                    source_episodes.append(episode_id)

                new_consolidation_count = consolidation_count + 1
                new_decay_resistance = min(1.0, 0.5 + 0.05 * math.log(new_consolidation_count + 1))

                cursor.execute("""
                    UPDATE semantic_concepts
                    SET
                        strength = ?,
                        source_episodes = ?,
                        consolidation_count = ?,
                        decay_resistance = ?,
                        last_reinforced_at = datetime('now'),
                        updated_at = datetime('now')
                    WHERE id = ?
                """, (
                    new_strength,
                    json.dumps(source_episodes, default=_json_default),
                    new_consolidation_count,
                    new_decay_resistance,
                    concept_id
                ))

                cursor.close()

                logging.info(f"Strengthened concept {concept_id}: {current_strength:.2f} -> {new_strength:.2f}")
                return True

        except Exception as e:
            logging.error(f"Failed to strengthen concept {concept_id}: {e}")
            return False

    def store_relationship(self, relationship_data: dict) -> str:
        """Store a relationship between two concepts."""
        required_fields = ['source_concept_id', 'target_concept_id', 'relationship_type']
        for field in required_fields:
            if field not in relationship_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, strength, source_episodes
                    FROM semantic_relationships
                    WHERE source_concept_id = ?
                      AND target_concept_id = ?
                      AND relationship_type = ?
                      AND deleted_at IS NULL
                """, (
                    relationship_data['source_concept_id'],
                    relationship_data['target_concept_id'],
                    relationship_data['relationship_type']
                ))

                existing = cursor.fetchone()

                if existing:
                    relationship_id, current_strength, source_episodes_raw = existing

                    if isinstance(source_episodes_raw, str):
                        source_episodes = json.loads(source_episodes_raw)
                    else:
                        source_episodes = source_episodes_raw or []

                    new_strength = min(1.0, current_strength * 1.1)

                    new_episodes = relationship_data.get('source_episodes', [])
                    for ep in new_episodes:
                        if ep not in source_episodes:
                            source_episodes.append(ep)

                    cursor.execute("""
                        UPDATE semantic_relationships
                        SET strength = ?, source_episodes = ?, updated_at = datetime('now')
                        WHERE id = ?
                    """, (new_strength, json.dumps(source_episodes, default=_json_default), relationship_id))

                    logging.info(f"Strengthened relationship {relationship_id}: {current_strength:.2f} -> {new_strength:.2f}")
                else:
                    relationship_id = str(uuid.uuid4())

                    cursor.execute("""
                        INSERT INTO semantic_relationships (
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        relationship_id,
                        relationship_data['source_concept_id'],
                        relationship_data['target_concept_id'],
                        relationship_data['relationship_type'],
                        relationship_data.get('strength', 0.5),
                        relationship_data.get('bidirectional', False),
                        json.dumps(relationship_data.get('source_episodes', []), default=_json_default),
                        relationship_data.get('confidence', 0.5)
                    ))

                    logging.info(f"Created relationship {relationship_id}: {relationship_data['relationship_type']}")

                cursor.close()
                return str(relationship_id)

        except Exception as e:
            logging.error(f"Failed to store relationship: {e}")
            raise

    def get_all_concepts(self) -> List[dict]:
        """Retrieve all non-deleted concepts with embeddings."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        sc.id, sc.concept_name, sc.concept_type, sc.definition,
                        sc.abstraction_level, sc.domain, sc.strength, sc.activation_score,
                        sc.access_count, sc.consolidation_count, sc.confidence, sc.source_episodes,
                        sc.verification_status, sc.context_constraints, sc.examples,
                        sc.first_learned_at, sc.last_accessed_at, sc.last_reinforced_at,
                        sc.utility_score, sc.decay_resistance, sc.created_at, sc.updated_at,
                        COALESCE(sc.reliability, 'reliable') AS reliability,
                        v.embedding
                    FROM semantic_concepts sc
                    JOIN semantic_concepts_vec v ON v.rowid = sc.rowid
                    WHERE sc.deleted_at IS NULL
                    ORDER BY sc.strength DESC, sc.confidence DESC
                """)

                rows = cursor.fetchall()
                cursor.close()

                concepts = []
                for row in rows:
                    raw_emb = row[23] if len(row) > 23 else None
                    if raw_emb and isinstance(raw_emb, (bytes, bytearray)):
                        n = len(raw_emb) // 4
                        embedding = list(struct.unpack(f'{n}f', raw_emb))
                    else:
                        embedding = None
                    concepts.append({
                        'id': str(row[0]),
                        'concept_name': row[1],
                        'concept_type': row[2],
                        'definition': row[3],
                        'embedding': embedding,
                        'abstraction_level': row[4],
                        'domain': row[5],
                        'strength': row[6],
                        'activation_score': row[7],
                        'access_count': row[8],
                        'consolidation_count': row[9],
                        'confidence': row[10],
                        'source_episodes': row[11],
                        'verification_status': row[12],
                        'context_constraints': row[13],
                        'examples': row[14],
                        'first_learned_at': row[15],
                        'last_accessed_at': row[16],
                        'last_reinforced_at': row[17],
                        'utility_score': row[18],
                        'decay_resistance': row[19],
                        'created_at': row[20],
                        'updated_at': row[21],
                        'reliability': row[22] if len(row) > 22 else 'reliable'
                    })

                logging.debug(f"Retrieved {len(concepts)} concepts")
                return concepts

        except Exception as e:
            logging.error(f"Failed to get all concepts: {e}")
            return []

    def get_relationships(self, concept_id: str, direction: str = 'outgoing') -> List[dict]:
        """Get relationships for a concept."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if direction == 'outgoing':
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE source_concept_id = ? AND deleted_at IS NULL
                    """, (concept_id,))
                elif direction == 'incoming':
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE target_concept_id = ? AND deleted_at IS NULL
                    """, (concept_id,))
                else:  # both
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE (source_concept_id = ? OR target_concept_id = ?)
                          AND deleted_at IS NULL
                    """, (concept_id, concept_id))

                rows = cursor.fetchall()
                cursor.close()

                relationships = []
                for row in rows:
                    relationships.append({
                        'id': str(row[0]),
                        'source_concept_id': str(row[1]),
                        'target_concept_id': str(row[2]),
                        'target_id': str(row[2]),
                        'relationship_type': row[3],
                        'strength': row[4],
                        'bidirectional': row[5],
                        'source_episodes': row[6],
                        'confidence': row[7]
                    })

                return relationships

        except Exception as e:
            logging.error(f"Failed to get relationships for {concept_id}: {e}")
            return []

    # ── Retrieval ────────────────────────────────────────────────────

    def retrieve_concepts(self, query: str, limit: int = 10, query_embedding=None) -> List[Dict[str, Any]]:
        """Retrieve relevant concepts using hybrid search with confidence filtering."""
        if query_embedding is None:
            if self.embedding_service is None:
                return []
            query_embedding = self.embedding_service.generate_embedding(query)

        all_concepts = self.get_all_concepts()

        if not all_concepts:
            return []

        scored_concepts = self._hybrid_search(query_embedding, all_concepts)

        confident_concepts = [
            concept for concept in scored_concepts
            if concept.get('confidence', 0) >= self.min_confidence_threshold
        ]

        if not confident_concepts:
            return []

        confident_concepts.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        filtered_concepts = confident_concepts[:limit]

        if filtered_concepts:
            concept_ids = [c['id'] for c in filtered_concepts]
            self._track_access(concept_ids)

        return filtered_concepts

    def _hybrid_search(self, query_embedding: np.ndarray, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple scoring factors."""
        weights = self.config.get('inference_weights', {
            'vector_similarity': 5,
            'strength': 3,
            'activation_score': 2,
            'utility_score': 2,
            'confidence': 1
        })

        scored_concepts = []

        for concept in concepts:
            concept_embedding = concept.get('embedding')
            if concept_embedding is None:
                continue

            vector_similarity = self._cosine_similarity(query_embedding, concept_embedding)

            strength = concept.get('strength', 0.5)
            activation_score = concept.get('activation_score', 0.0)
            utility_score = concept.get('utility_score', 0.0)
            confidence = concept.get('confidence', 0.5)

            hybrid_score = (
                weights['vector_similarity'] * vector_similarity +
                weights['strength'] * strength +
                weights['activation_score'] * activation_score +
                weights['utility_score'] * utility_score +
                weights['confidence'] * confidence
            )

            total_weight = sum(weights.values())
            hybrid_score = hybrid_score / total_weight

            concept_copy = concept.copy()
            concept_copy['hybrid_score'] = hybrid_score
            concept_copy['vector_similarity'] = vector_similarity

            scored_concepts.append(concept_copy)

        return scored_concepts

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if isinstance(vec1, str):
            vec1 = json.loads(vec1)
        if isinstance(vec2, str):
            vec2 = json.loads(vec2)
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        similarity = max(0.0, min(1.0, similarity))

        return float(similarity)

    def spreading_activation(self, seed_concepts: List[str], max_depth: int = 2) -> List[Dict[str, Any]]:
        """Perform spreading activation from seed concepts via BFS."""
        activation_levels = {}

        for concept_id in seed_concepts:
            activation_levels[concept_id] = 1.0

        frontier = deque()
        for concept_id in seed_concepts:
            frontier.append((concept_id, 0, 1.0))

        visited = set(seed_concepts)
        decay_factor = 0.7
        activation_threshold = 0.3

        while frontier:
            current_id, depth, activation = frontier.popleft()

            if depth >= max_depth:
                continue

            relationships = self.get_relationships(current_id)

            for relationship in relationships:
                target_id = relationship.get('target_id')
                relationship_strength = relationship.get('strength', 0.5)

                if target_id is None:
                    continue

                new_activation = activation * decay_factor

                if relationship_strength < 0.5:
                    if random.random() >= 0.15:
                        continue
                else:
                    new_activation = new_activation * relationship_strength

                if new_activation < activation_threshold:
                    continue

                if target_id in activation_levels:
                    activation_levels[target_id] = max(activation_levels[target_id], new_activation)
                else:
                    activation_levels[target_id] = new_activation

                if target_id not in visited:
                    visited.add(target_id)
                    frontier.append((target_id, depth + 1, new_activation))

        activated_concepts = []
        for concept_id, activation_score in activation_levels.items():
            concept = self.get_concept(concept_id)
            if concept:
                concept_copy = concept.copy()
                concept_copy['activation_score'] = activation_score
                activated_concepts.append(concept_copy)

        activated_concepts.sort(key=lambda x: x.get('activation_score', 0), reverse=True)

        if activated_concepts:
            concept_ids = [c['id'] for c in activated_concepts]
            self._track_access(concept_ids)

        return activated_concepts

    def _track_access(self, concept_ids: List[str]) -> None:
        """Track concept access for utility calculation."""
        if not concept_ids:
            return

        if not self.db_service:
            logging.warning("Database service not available for access tracking")
            return

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                for concept_id in concept_ids:
                    cursor.execute("""
                        UPDATE semantic_concepts
                        SET
                            access_count = access_count + 1,
                            last_accessed_at = datetime('now'),
                            updated_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (concept_id,))

                cursor.close()
                logging.debug(f"Tracked access for {len(concept_ids)} concepts")

        except Exception as e:
            logging.error(f"Failed to track access: {e}")
