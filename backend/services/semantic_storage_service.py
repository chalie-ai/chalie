"""
Semantic Storage Service - CRUD operations for semantic concepts.
Responsibility: Storage layer only (SRP).
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from services.database_service import DatabaseService


def _json_default(obj):
    """Handle UUID and other non-serializable types from PostgreSQL."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SemanticStorageService:
    """Manages semantic concept storage and retrieval operations."""

    def __init__(self, database_service: DatabaseService):
        """
        Initialize storage service.

        Args:
            database_service: DatabaseService instance for connection management
        """
        self.db_service = database_service

    def store_concept(self, concept_data: dict) -> str:
        """
        Store a new concept in the database.

        Args:
            concept_data: Concept dict with fields:
                         concept_name, concept_type, definition, embedding,
                         abstraction_level, domain, confidence, source_episodes

        Returns:
            UUID of the created concept

        Raises:
            ValueError if required fields are missing
            Exception if storage fails
        """
        required_fields = ['concept_name', 'concept_type', 'definition']
        for field in required_fields:
            if field not in concept_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO semantic_concepts (
                        concept_name, concept_type, definition, embedding,
                        abstraction_level, domain, confidence, source_episodes,
                        context_constraints, examples
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    concept_data['concept_name'],
                    concept_data['concept_type'],
                    concept_data['definition'],
                    concept_data.get('embedding'),
                    concept_data.get('abstraction_level', 3),
                    concept_data.get('domain'),
                    concept_data.get('confidence', 0.5),
                    json.dumps(concept_data.get('source_episodes', []), default=_json_default),
                    json.dumps(concept_data.get('context_constraints', {}), default=_json_default),
                    json.dumps(concept_data.get('examples', []), default=_json_default)
                ))

                concept_id = cursor.fetchone()[0]
                cursor.close()

                logging.info(f"Stored concept {concept_id}: '{concept_data['concept_name']}' ({concept_data['concept_type']})")
                return str(concept_id)

        except Exception as e:
            logging.error(f"Failed to store concept: {e}")
            raise

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """
        Retrieve a concept by ID.

        Args:
            concept_id: UUID of concept to retrieve

        Returns:
            Concept dict or None if not found
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, concept_name, concept_type, definition, embedding,
                        abstraction_level, domain, strength, activation_score,
                        access_count, consolidation_count, confidence, source_episodes,
                        verification_status, context_constraints, examples,
                        first_learned_at, last_accessed_at, last_reinforced_at,
                        utility_score, decay_resistance, created_at, updated_at
                    FROM semantic_concepts
                    WHERE id = %s AND deleted_at IS NULL
                """, (concept_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                embedding = row[4]
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)

                return {
                    'id': str(row[0]),
                    'concept_name': row[1],
                    'concept_type': row[2],
                    'definition': row[3],
                    'embedding': embedding,
                    'abstraction_level': row[5],
                    'domain': row[6],
                    'strength': row[7],
                    'activation_score': row[8],
                    'access_count': row[9],
                    'consolidation_count': row[10],
                    'confidence': row[11],
                    'source_episodes': row[12],
                    'verification_status': row[13],
                    'context_constraints': row[14],
                    'examples': row[15],
                    'first_learned_at': row[16],
                    'last_accessed_at': row[17],
                    'last_reinforced_at': row[18],
                    'utility_score': row[19],
                    'decay_resistance': row[20],
                    'created_at': row[21],
                    'updated_at': row[22]
                }

        except Exception as e:
            logging.error(f"Failed to get concept {concept_id}: {e}")
            return None

    def strengthen_concept(self, concept_id: str, episode_id: str) -> bool:
        """
        Strengthen a concept (increase strength by reinforcement factor).

        Args:
            concept_id: UUID of concept to strengthen
            episode_id: UUID of episode providing reinforcement

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Get current concept data
                cursor.execute("""
                    SELECT strength, source_episodes, consolidation_count, decay_resistance
                    FROM semantic_concepts
                    WHERE id = %s AND deleted_at IS NULL
                """, (concept_id,))

                row = cursor.fetchone()
                if not row:
                    logging.warning(f"Concept {concept_id} not found for strengthening")
                    cursor.close()
                    return False

                current_strength, source_episodes, consolidation_count, decay_resistance = row

                # Strengthening formula: new_strength = min(10.0, old_strength × 1.1)
                new_strength = min(10.0, current_strength * 1.1)

                # Add episode to source_episodes if not already present
                if episode_id not in source_episodes:
                    source_episodes.append(episode_id)

                # Increment consolidation count
                new_consolidation_count = consolidation_count + 1

                # Update decay_resistance based on consolidation (0.5 + 0.05 × log(consolidation_count + 1))
                import math
                new_decay_resistance = min(1.0, 0.5 + 0.05 * math.log(new_consolidation_count + 1))

                # Update concept
                cursor.execute("""
                    UPDATE semantic_concepts
                    SET
                        strength = %s,
                        source_episodes = %s,
                        consolidation_count = %s,
                        decay_resistance = %s,
                        last_reinforced_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    new_strength,
                    json.dumps(source_episodes, default=_json_default),
                    new_consolidation_count,
                    new_decay_resistance,
                    concept_id
                ))

                cursor.close()

                logging.info(f"Strengthened concept {concept_id}: {current_strength:.2f} → {new_strength:.2f}")
                return True

        except Exception as e:
            logging.error(f"Failed to strengthen concept {concept_id}: {e}")
            return False

    def store_relationship(self, relationship_data: dict) -> str:
        """
        Store a relationship between two concepts.

        Args:
            relationship_data: Dict with source_concept_id, target_concept_id,
                              relationship_type, strength, confidence, source_episodes

        Returns:
            UUID of the created relationship
        """
        required_fields = ['source_concept_id', 'target_concept_id', 'relationship_type']
        for field in required_fields:
            if field not in relationship_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Check if relationship already exists (UNIQUE constraint)
                cursor.execute("""
                    SELECT id, strength, source_episodes
                    FROM semantic_relationships
                    WHERE source_concept_id = %s
                      AND target_concept_id = %s
                      AND relationship_type = %s
                      AND deleted_at IS NULL
                """, (
                    relationship_data['source_concept_id'],
                    relationship_data['target_concept_id'],
                    relationship_data['relationship_type']
                ))

                existing = cursor.fetchone()

                if existing:
                    # Strengthen existing relationship
                    relationship_id, current_strength, source_episodes = existing
                    new_strength = min(1.0, current_strength * 1.1)

                    # Add new episodes
                    new_episodes = relationship_data.get('source_episodes', [])
                    for ep in new_episodes:
                        if ep not in source_episodes:
                            source_episodes.append(ep)

                    cursor.execute("""
                        UPDATE semantic_relationships
                        SET strength = %s, source_episodes = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (new_strength, json.dumps(source_episodes, default=_json_default), relationship_id))

                    logging.info(f"Strengthened relationship {relationship_id}: {current_strength:.2f} → {new_strength:.2f}")
                else:
                    # Create new relationship
                    cursor.execute("""
                        INSERT INTO semantic_relationships (
                            source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        relationship_data['source_concept_id'],
                        relationship_data['target_concept_id'],
                        relationship_data['relationship_type'],
                        relationship_data.get('strength', 0.5),
                        relationship_data.get('bidirectional', False),
                        json.dumps(relationship_data.get('source_episodes', []), default=_json_default),
                        relationship_data.get('confidence', 0.5)
                    ))

                    relationship_id = cursor.fetchone()[0]
                    logging.info(f"Created relationship {relationship_id}: {relationship_data['relationship_type']}")

                cursor.close()
                return str(relationship_id)

        except Exception as e:
            logging.error(f"Failed to store relationship: {e}")
            raise

    def get_all_concepts(self) -> List[dict]:
        """
        Retrieve all non-deleted concepts with embeddings.

        Returns:
            List of concept dictionaries
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        id, concept_name, concept_type, definition, embedding,
                        abstraction_level, domain, strength, activation_score,
                        access_count, consolidation_count, confidence, source_episodes,
                        verification_status, context_constraints, examples,
                        first_learned_at, last_accessed_at, last_reinforced_at,
                        utility_score, decay_resistance, created_at, updated_at
                    FROM semantic_concepts
                    WHERE deleted_at IS NULL AND embedding IS NOT NULL
                    ORDER BY strength DESC, confidence DESC
                """)

                rows = cursor.fetchall()
                cursor.close()

                concepts = []
                for row in rows:
                    embedding = row[4]
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    concepts.append({
                        'id': str(row[0]),
                        'concept_name': row[1],
                        'concept_type': row[2],
                        'definition': row[3],
                        'embedding': embedding,
                        'abstraction_level': row[5],
                        'domain': row[6],
                        'strength': row[7],
                        'activation_score': row[8],
                        'access_count': row[9],
                        'consolidation_count': row[10],
                        'confidence': row[11],
                        'source_episodes': row[12],
                        'verification_status': row[13],
                        'context_constraints': row[14],
                        'examples': row[15],
                        'first_learned_at': row[16],
                        'last_accessed_at': row[17],
                        'last_reinforced_at': row[18],
                        'utility_score': row[19],
                        'decay_resistance': row[20],
                        'created_at': row[21],
                        'updated_at': row[22]
                    })

                logging.debug(f"Retrieved {len(concepts)} concepts")
                return concepts

        except Exception as e:
            logging.error(f"Failed to get all concepts: {e}")
            return []

    def get_relationships(self, concept_id: str, direction: str = 'outgoing') -> List[dict]:
        """
        Get relationships for a concept.

        Args:
            concept_id: UUID of concept
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of relationship dicts with target_id field for compatibility
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if direction == 'outgoing':
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE source_concept_id = %s AND deleted_at IS NULL
                    """, (concept_id,))
                elif direction == 'incoming':
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE target_concept_id = %s AND deleted_at IS NULL
                    """, (concept_id,))
                else:  # both
                    cursor.execute("""
                        SELECT
                            id, source_concept_id, target_concept_id, relationship_type,
                            strength, bidirectional, source_episodes, confidence
                        FROM semantic_relationships
                        WHERE (source_concept_id = %s OR target_concept_id = %s)
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
                        'target_id': str(row[2]),  # Alias for compatibility with spreading activation
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
