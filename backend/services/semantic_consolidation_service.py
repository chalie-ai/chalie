# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Semantic Consolidation Service - Extracts concepts from episodes.
Responsibility: Episode analysis and concept extraction (SRP).
"""

import logging
import json
from typing import Dict, List, Optional
from services.semantic_storage_service import SemanticStorageService
from services.config_service import ConfigService


class SemanticConsolidationService:
    """Extracts semantic concepts and relationships from episodic memory."""

    def __init__(
        self,
        ollama_service,
        storage_service: SemanticStorageService,
        config_service: ConfigService
    ):
        """
        Initialize consolidation service.

        Args:
            ollama_service: LLM service for inference
            storage_service: SemanticStorageService for persistence
            config_service: ConfigService for semantic memory config
        """
        self.ollama = ollama_service
        self.storage = storage_service
        self.config = config_service.get_agent_config("semantic-memory")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load semantic extraction prompt template."""
        try:
            from services.config_service import ConfigService
            return ConfigService.get_agent_prompt("semantic-extraction")
        except Exception as e:
            logging.error(f"Failed to load prompt template: {e}")
            raise

    def extract_from_episode(self, episode: dict) -> dict:
        """
        Extract concepts and relationships from an episode.

        Args:
            episode: Episode dict with gist, action, outcome, intent, context

        Returns:
            Dict with 'concepts' and 'relationships' lists
        """
        try:
            # Build episode content for extraction
            episode_content = self._build_episode_content(episode)

            # Fill prompt template
            prompt = self.prompt_template.replace('{{episode_content}}', episode_content)

            # Call LLM for extraction
            response = self.ollama.send_message("", prompt).text

            # Parse JSON response
            extracted = json.loads(response)

            # Validate structure
            if 'concepts' not in extracted or 'relationships' not in extracted:
                logging.warning(f"Invalid extraction structure for episode {episode.get('id', 'unknown')}")
                return {'concepts': [], 'relationships': []}

            logging.info(f"Extracted {len(extracted['concepts'])} concepts, {len(extracted['relationships'])} relationships")
            return extracted

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse extraction JSON: {e}")
            return {'concepts': [], 'relationships': []}
        except Exception as e:
            logging.error(f"Failed to extract from episode: {e}")
            return {'concepts': [], 'relationships': []}

    def _build_episode_content(self, episode: dict) -> str:
        """Build episode content string for extraction."""
        parts = []

        if episode.get('gist'):
            parts.append(f"Summary: {episode['gist']}")

        if episode.get('intent'):
            parts.append(f"Intent: {json.dumps(episode['intent'])}")

        if episode.get('context'):
            parts.append(f"Context: {json.dumps(episode['context'])}")

        if episode.get('action'):
            parts.append(f"Action: {episode['action']}")

        if episode.get('outcome'):
            parts.append(f"Outcome: {episode['outcome']}")

        return "\n\n".join(parts)

    def consolidate_concept(self, concept: dict, episode_id: str) -> str:
        """
        Consolidate a concept (match existing or create new).

        Args:
            concept: Extracted concept dict (name, type, definition, abstraction_level, domain)
            episode_id: Source episode UUID

        Returns:
            UUID of concept (existing or new)
        """
        try:
            # Generate embedding for concept
            from services.embedding_service import EmbeddingService
            emb_service = EmbeddingService()
            concept_text = f"{concept['name']} ({concept['type']}): {concept['definition']}"
            embedding = emb_service.generate_embedding(concept_text)

            # Search for similar existing concepts
            existing_concept = self._find_similar_concept(
                concept_name=concept['name'],
                embedding=embedding,
                concept_type=concept['type']
            )

            if existing_concept:
                # Strengthen existing concept
                self.storage.strengthen_concept(existing_concept['id'], episode_id)
                logging.info(f"Strengthened existing concept: {existing_concept['concept_name']}")
                return existing_concept['id']
            else:
                # Create new concept
                concept_data = {
                    'concept_name': concept['name'],
                    'concept_type': concept['type'],
                    'definition': concept['definition'],
                    'embedding': embedding,
                    'abstraction_level': concept.get('abstraction_level', 3),
                    'domain': concept.get('domain'),
                    'confidence': 0.5,
                    'source_episodes': [episode_id]
                }

                concept_id = self.storage.store_concept(concept_data)
                logging.info(f"Created new concept: {concept['name']}")
                return concept_id

        except Exception as e:
            logging.error(f"Failed to consolidate concept '{concept.get('name', 'unknown')}': {e}")
            raise

    def _find_similar_concept(self, concept_name: str, embedding: list, concept_type: str) -> Optional[dict]:
        """
        Find similar concept using hybrid search (name match + vector similarity).

        Args:
            concept_name: Name of concept
            embedding: Embedding vector
            concept_type: Type of concept

        Returns:
            Existing concept dict or None
        """
        conn = None
        try:
            conn = self.storage.db_service.get_connection()
            cursor = conn.cursor()

            # Hybrid search: exact name match OR high cosine similarity (>0.85)
            cursor.execute("""
                SELECT
                    id, concept_name, concept_type, definition, embedding,
                    abstraction_level, domain, strength, confidence
                FROM semantic_concepts
                WHERE deleted_at IS NULL
                  AND concept_type = %s
                  AND (
                    LOWER(concept_name) = LOWER(%s)
                    OR (1 - (embedding <=> %s::vector)) > %s
                  )
                ORDER BY
                    CASE WHEN LOWER(concept_name) = LOWER(%s) THEN 0 ELSE 1 END,
                    (1 - (embedding <=> %s::vector)) DESC
                LIMIT 1
            """, (
                concept_type,
                concept_name,
                embedding,
                self.config['similarity_threshold'],
                concept_name,
                embedding
            ))

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
                'confidence': row[8]
            }

        except Exception as e:
            logging.error(f"Failed to find similar concept: {e}")
            return None
        finally:
            if conn:
                self.storage.db_service.release_connection(conn)

    def consolidate_relationship(
        self,
        relationship: dict,
        episode_id: str,
        concept_name_to_id: dict
    ) -> Optional[str]:
        """
        Consolidate a relationship (match existing or create new).

        Args:
            relationship: Extracted relationship dict (source, target, type, strength)
            episode_id: Source episode UUID
            concept_name_to_id: Mapping from concept names to UUIDs

        Returns:
            UUID of relationship (existing or new), or None if concepts not found
        """
        try:
            # Resolve concept IDs from names
            source_id = concept_name_to_id.get(relationship['source'])
            target_id = concept_name_to_id.get(relationship['target'])

            if not source_id or not target_id:
                logging.warning(f"Cannot create relationship: concept not found (source={relationship['source']}, target={relationship['target']})")
                return None

            # Store relationship (storage service handles duplicates)
            relationship_data = {
                'source_concept_id': source_id,
                'target_concept_id': target_id,
                'relationship_type': relationship['type'],
                'strength': relationship.get('strength', 0.5),
                'confidence': 0.5,
                'source_episodes': [episode_id],
                'bidirectional': relationship.get('bidirectional', False)
            }

            relationship_id = self.storage.store_relationship(relationship_data)
            return relationship_id

        except Exception as e:
            logging.error(f"Failed to consolidate relationship: {e}")
            return None
