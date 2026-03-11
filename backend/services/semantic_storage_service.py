"""
Semantic Storage Service - CRUD operations for semantic concepts.
Responsibility: Storage layer only (SRP).
"""

import logging
import json
import struct
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from services.database_service import DatabaseService


def _json_default(obj):
    """Handle UUID and other non-serializable types."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list/tuple of floats into a binary blob for sqlite-vec."""
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, (list, tuple)):
        return struct.pack(f'{len(embedding)}f', *embedding)
    return embedding


class SemanticStorageService:
    """Manages semantic concept storage and retrieval operations."""

    def __init__(self, database_service: DatabaseService):
        """
        Initialize storage service.

        Args:
            database_service: DatabaseService instance for connection management
        """
        self.db_service = database_service

    def _store_embedding(self, conn, concept_id: str, embedding):
        """Store embedding in the companion vec table."""
        try:
            blob = _pack_embedding(embedding)
            if blob is None:
                return

            # Get the rowid of the concept for vec table linking
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

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, concept_id, embedding)

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
                    'embedding': None,  # Embeddings live in vec table, not base table
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
                    WHERE id = ? AND deleted_at IS NULL
                """, (concept_id,))

                row = cursor.fetchone()
                if not row:
                    logging.warning(f"Concept {concept_id} not found for strengthening")
                    cursor.close()
                    return False

                current_strength, source_episodes_raw, consolidation_count, decay_resistance = row

                # Parse source_episodes from JSON text
                if isinstance(source_episodes_raw, str):
                    source_episodes = json.loads(source_episodes_raw)
                else:
                    source_episodes = source_episodes_raw or []

                # Strengthening formula: new_strength = min(10.0, old_strength * 1.1)
                new_strength = min(10.0, current_strength * 1.1)

                # Add episode to source_episodes if not already present
                if episode_id not in source_episodes:
                    source_episodes.append(episode_id)

                # Increment consolidation count
                new_consolidation_count = consolidation_count + 1

                # Update decay_resistance based on consolidation (0.5 + 0.05 * log(consolidation_count + 1))
                import math
                new_decay_resistance = min(1.0, 0.5 + 0.05 * math.log(new_consolidation_count + 1))

                # Update concept
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
                    # Strengthen existing relationship
                    relationship_id, current_strength, source_episodes_raw = existing

                    # Parse source_episodes from JSON text
                    if isinstance(source_episodes_raw, str):
                        source_episodes = json.loads(source_episodes_raw)
                    else:
                        source_episodes = source_episodes_raw or []

                    new_strength = min(1.0, current_strength * 1.1)

                    # Add new episodes
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
                    # Create new relationship
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
        """
        Retrieve all non-deleted concepts with embeddings.

        Returns:
            List of concept dictionaries
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Join with vec table to only return concepts that have embeddings
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
                    # Unpack embedding blob from vec table (float32, 768d)
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
