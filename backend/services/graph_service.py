"""
Graph Service - Query interface for semantic concept graph.
Provides read-only access to concepts and relationships.
"""

import logging
from typing import Optional, List, Dict, Any
from services.database_service import DatabaseService


class GraphService:
    """
    Query interface for the semantic concept graph.
    Provides read-only methods for retrieving concepts and relationships.
    """

    def __init__(self, db_service: DatabaseService):
        """
        Initialize graph service.

        Args:
            db_service: DatabaseService instance for connection management
        """
        self.db_service = db_service

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all non-deleted concepts with embeddings.

        Returns:
            List of concept dictionaries with all fields
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
                    concepts.append({
                        'id': str(row[0]),
                        'concept_name': row[1],
                        'concept_type': row[2],
                        'definition': row[3],
                        'embedding': row[4],
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

                logging.debug(f"Retrieved {len(concepts)} concepts from graph")
                return concepts

        except Exception as e:
            logging.error(f"Failed to get all concepts: {e}")
            return []

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single concept by ID.

        Args:
            concept_id: UUID of the concept

        Returns:
            Concept dictionary or None if not found
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

                return {
                    'id': str(row[0]),
                    'concept_name': row[1],
                    'concept_type': row[2],
                    'definition': row[3],
                    'embedding': row[4],
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

    def get_relationships(self, concept_id: str, direction: str = 'outgoing') -> List[Dict[str, Any]]:
        """
        Get relationships for a concept with target concept info.

        Args:
            concept_id: UUID of the concept
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of relationship dictionaries with target concept details
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                if direction == 'outgoing':
                    cursor.execute("""
                        SELECT
                            r.id, r.source_concept_id, r.target_concept_id,
                            r.relationship_type, r.strength, r.bidirectional,
                            r.source_episodes, r.confidence,
                            c.concept_name, c.concept_type, c.definition
                        FROM semantic_relationships r
                        JOIN semantic_concepts c ON r.target_concept_id = c.id
                        WHERE r.source_concept_id = %s
                          AND r.deleted_at IS NULL
                          AND c.deleted_at IS NULL
                    """, (concept_id,))
                elif direction == 'incoming':
                    cursor.execute("""
                        SELECT
                            r.id, r.source_concept_id, r.target_concept_id,
                            r.relationship_type, r.strength, r.bidirectional,
                            r.source_episodes, r.confidence,
                            c.concept_name, c.concept_type, c.definition
                        FROM semantic_relationships r
                        JOIN semantic_concepts c ON r.source_concept_id = c.id
                        WHERE r.target_concept_id = %s
                          AND r.deleted_at IS NULL
                          AND c.deleted_at IS NULL
                    """, (concept_id,))
                else:  # both
                    cursor.execute("""
                        SELECT
                            r.id, r.source_concept_id, r.target_concept_id,
                            r.relationship_type, r.strength, r.bidirectional,
                            r.source_episodes, r.confidence,
                            c.concept_name, c.concept_type, c.definition
                        FROM semantic_relationships r
                        LEFT JOIN semantic_concepts c ON
                            CASE
                                WHEN r.source_concept_id = %s THEN r.target_concept_id = c.id
                                ELSE r.source_concept_id = c.id
                            END
                        WHERE (r.source_concept_id = %s OR r.target_concept_id = %s)
                          AND r.deleted_at IS NULL
                          AND c.deleted_at IS NULL
                    """, (concept_id, concept_id, concept_id))

                rows = cursor.fetchall()
                cursor.close()

                relationships = []
                for row in rows:
                    relationships.append({
                        'id': str(row[0]),
                        'source_concept_id': str(row[1]),
                        'target_concept_id': str(row[2]),
                        'target_id': str(row[2]),  # Alias for compatibility
                        'relationship_type': row[3],
                        'strength': row[4],
                        'bidirectional': row[5],
                        'source_episodes': row[6],
                        'confidence': row[7],
                        'target_name': row[8],
                        'target_type': row[9],
                        'target_definition': row[10]
                    })

                logging.debug(f"Retrieved {len(relationships)} relationships for concept {concept_id}")
                return relationships

        except Exception as e:
            logging.error(f"Failed to get relationships for {concept_id}: {e}")
            return []
