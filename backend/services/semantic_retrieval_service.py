# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Semantic Retrieval Service

Provides hybrid search combining vector similarity with semantic graph properties.
Implements confidence filtering and spreading activation for enhanced retrieval.
"""

import json
import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional
from collections import deque
from services.embedding_service import get_embedding_service
from services.semantic_storage_service import SemanticStorageService
from services.config_service import ConfigService


class SemanticRetrievalService:
    """
    Handles semantic retrieval using hybrid search with confidence filtering.
    """

    def __init__(self, db_service, embedding_service = None, storage_service: SemanticStorageService = None):
        self.db_service = db_service
        self.embedding_service = embedding_service or get_embedding_service()
        self.storage_service = storage_service or SemanticStorageService(db_service)
        self.config = ConfigService.get_agent_config("semantic-memory")
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.4)

    def retrieve_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant concepts using hybrid search with confidence filtering.

        Args:
            query: Natural language query
            limit: Maximum number of concepts to retrieve

        Returns:
            List of concept dictionaries with scores, filtered by confidence threshold
        """
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)

        # Get all concepts from storage
        all_concepts = self.storage_service.get_all_concepts()

        if not all_concepts:
            return []

        # Perform hybrid search
        scored_concepts = self._hybrid_search(query_embedding, all_concepts)

        # Apply confidence filtering - mimics "can't remember" experience
        confident_concepts = [
            concept for concept in scored_concepts
            if concept.get('confidence', 0) >= self.min_confidence_threshold
        ]

        # If all concepts are below threshold, return empty (can't remember)
        if not confident_concepts:
            return []

        # Sort by hybrid score and limit results
        confident_concepts.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        filtered_concepts = confident_concepts[:limit]

        # Track access for utility calculation
        if filtered_concepts:
            concept_ids = [c['id'] for c in filtered_concepts]
            self._track_access(concept_ids)

        return filtered_concepts

    def _hybrid_search(self, query_embedding: np.ndarray, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining multiple scoring factors.

        Scoring factors (from config):
        - Vector similarity: weight = 5
        - Strength: weight = 3
        - Activation score: weight = 2
        - Utility score: weight = 2
        - Confidence: weight = 1

        Args:
            query_embedding: Query vector
            concepts: List of concept dictionaries

        Returns:
            List of concepts with added hybrid_score field
        """
        weights = self.config.get('inference_weights', {
            'vector_similarity': 5,
            'strength': 3,
            'activation_score': 2,
            'utility_score': 2,
            'confidence': 1
        })

        scored_concepts = []

        for concept in concepts:
            # Get concept embedding
            concept_embedding = concept.get('embedding')
            if concept_embedding is None:
                # Skip concepts without embeddings
                continue

            # Calculate vector similarity (cosine similarity)
            vector_similarity = self._cosine_similarity(query_embedding, concept_embedding)

            # Get semantic properties (normalized 0-1)
            strength = concept.get('strength', 0.5)
            activation_score = concept.get('activation_score', 0.0)
            utility_score = concept.get('utility_score', 0.0)
            confidence = concept.get('confidence', 0.5)

            # Calculate weighted hybrid score
            hybrid_score = (
                weights['vector_similarity'] * vector_similarity +
                weights['strength'] * strength +
                weights['activation_score'] * activation_score +
                weights['utility_score'] * utility_score +
                weights['confidence'] * confidence
            )

            # Normalize by total weight
            total_weight = sum(weights.values())
            hybrid_score = hybrid_score / total_weight

            # Add score to concept
            concept_copy = concept.copy()
            concept_copy['hybrid_score'] = hybrid_score
            concept_copy['vector_similarity'] = vector_similarity

            scored_concepts.append(concept_copy)

        return scored_concepts

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        # pgvector may return embeddings as strings via psycopg2
        if isinstance(vec1, str):
            vec1 = json.loads(vec1)
        if isinstance(vec2, str):
            vec2 = json.loads(vec2)
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, similarity))

        return float(similarity)

    def spreading_activation(self, seed_concepts: List[str], max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Perform spreading activation from seed concepts.

        Implements BFS traversal through the semantic graph with:
        - Activation decay of 0.7 per depth level
        - 15% random chance for weak relationships (strength < 0.5) to activate (creative leaps)
        - Activation threshold of 0.3 (below this, stop spreading)

        Args:
            seed_concepts: List of concept IDs to start activation from
            max_depth: Maximum traversal depth (default: 2)

        Returns:
            List of activated concepts with activation scores
        """
        # Track activation levels for each concept
        activation_levels = {}

        # Initialize seed concepts with full activation (1.0)
        for concept_id in seed_concepts:
            activation_levels[concept_id] = 1.0

        # BFS queue: (concept_id, current_depth, activation_level)
        frontier = deque()
        for concept_id in seed_concepts:
            frontier.append((concept_id, 0, 1.0))

        # Track visited to avoid reprocessing
        visited = set(seed_concepts)

        # Decay factor per depth level
        decay_factor = 0.7

        # Activation threshold - below this, stop spreading
        activation_threshold = 0.3

        # BFS traversal
        while frontier:
            current_id, depth, activation = frontier.popleft()

            # Stop if max depth reached
            if depth >= max_depth:
                continue

            # Get relationships from current concept
            relationships = self.storage_service.get_relationships(current_id)

            for relationship in relationships:
                target_id = relationship.get('target_id')
                relationship_strength = relationship.get('strength', 0.5)

                if target_id is None:
                    continue

                # Calculate activation for target concept
                # Apply decay factor
                new_activation = activation * decay_factor

                # Weak relationships (strength < 0.5) have 15% random chance to activate
                # This enables "creative leaps"
                if relationship_strength < 0.5:
                    if random.random() >= 0.15:
                        # 85% chance to skip weak relationships
                        continue
                else:
                    # Strong relationships: weight activation by strength
                    new_activation = new_activation * relationship_strength

                # Check activation threshold
                if new_activation < activation_threshold:
                    continue

                # Update activation level (use max if concept already activated)
                if target_id in activation_levels:
                    activation_levels[target_id] = max(activation_levels[target_id], new_activation)
                else:
                    activation_levels[target_id] = new_activation

                # Add to frontier if not visited
                if target_id not in visited:
                    visited.add(target_id)
                    frontier.append((target_id, depth + 1, new_activation))

        # Build result list with concept details
        activated_concepts = []
        for concept_id, activation_score in activation_levels.items():
            # Get concept details from storage
            concept = self.storage_service.get_concept(concept_id)
            if concept:
                concept_copy = concept.copy()
                concept_copy['activation_score'] = activation_score
                activated_concepts.append(concept_copy)

        # Sort by activation score (highest first)
        activated_concepts.sort(key=lambda x: x.get('activation_score', 0), reverse=True)

        # Track access for utility calculation
        if activated_concepts:
            concept_ids = [c['id'] for c in activated_concepts]
            self._track_access(concept_ids)

        return activated_concepts

    def _track_access(self, concept_ids: List[str]) -> None:
        """
        Track concept access for utility calculation.

        Args:
            concept_ids: List of concept IDs that were accessed
        """
        if not concept_ids:
            return

        if not self.db_service:
            logging.warning("Database service not available for access tracking")
            return

        conn = None
        try:
            conn = self.db_service.get_connection()
            cursor = conn.cursor()

            # Batch update access tracking
            for concept_id in concept_ids:
                cursor.execute("""
                    UPDATE semantic_concepts
                    SET
                        access_count = access_count + 1,
                        last_accessed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (concept_id,))

            conn.commit()
            cursor.close()
            logging.debug(f"Tracked access for {len(concept_ids)} concepts")

        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Failed to track access: {e}")
        finally:
            if conn:
                self.db_service.release_connection(conn)
