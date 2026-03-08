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
import struct
from typing import Dict, List, Optional
from services.semantic_storage_service import SemanticStorageService
from services.config_service import ConfigService


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list/tuple of floats into a binary blob for sqlite-vec."""
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, (list, tuple)):
        return struct.pack(f'{len(embedding)}f', *embedding)
    return embedding


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

            # Validate response is not empty
            if not response or not response.strip():
                logging.error(f"Empty response from LLM for episode {episode.get('id', 'unknown')}")
                return {'concepts': [], 'relationships': []}

            # Log response for debugging (truncate if too long)
            response_preview = response[:200] if len(response) > 200 else response
            logging.debug(f"LLM response preview for episode {episode.get('id', 'unknown')}: {response_preview}")

            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json_from_response(response)

            # Parse JSON response
            extracted = json.loads(json_str)

            # Validate structure
            if 'concepts' not in extracted or 'relationships' not in extracted:
                logging.warning(f"Invalid extraction structure for episode {episode.get('id', 'unknown')}")
                return {'concepts': [], 'relationships': []}

            logging.info(f"Extracted {len(extracted['concepts'])} concepts, {len(extracted['relationships'])} relationships")
            return extracted

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse extraction JSON: {e} (response length: {len(response) if response else 0})")
            return {'concepts': [], 'relationships': []}
        except Exception as e:
            logging.error(f"Failed to extract from episode: {e}")
            return {'concepts': [], 'relationships': []}

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response.

        Handles responses wrapped in markdown code blocks or plain JSON.

        Args:
            response: Raw LLM response text

        Returns:
            Valid JSON string
        """
        # Try to extract from markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Try to extract from generic code block
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Return as-is, stripped of whitespace
        return response.strip()

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
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            concept_text = f"{concept['name']} ({concept['type']}): {concept['definition']}"
            embedding = emb_service.generate_embedding(concept_text)

            # Search for similar existing concepts
            existing_concept = self._find_similar_concept(
                concept_name=concept['name'],
                embedding=embedding,
                concept_type=concept['type']
            )

            if existing_concept:
                # Before strengthening, check if the new extraction contradicts the existing concept
                try:
                    from services.contradiction_classifier_service import ContradictionClassifierService
                    from services.uncertainty_service import UncertaintyService
                    db_svc = self.storage.db_service
                    classifier = ContradictionClassifierService(db_service=db_svc)
                    conflict = classifier.check_concept_conflict(
                        concept_name=concept['name'],
                        concept_definition=concept['definition'],
                        existing=existing_concept,
                    )
                    if conflict:
                        unc_svc = UncertaintyService(db_svc)
                        # Skip if this concept already has an open contradiction record
                        existing_uncs = unc_svc.get_uncertainties_for_memory(
                            'concept', existing_concept['id']
                        )
                        already_tracked = any(
                            u.get('uncertainty_type') == 'contradiction'
                            and u.get('state') in ('open', 'surfaced')
                            for u in existing_uncs
                        )
                        if not already_tracked:
                            unc_svc.create_uncertainty(
                                memory_a_type='concept',
                                memory_a_id=existing_concept['id'],
                                memory_b_type=None,
                                memory_b_id=None,
                                uncertainty_type='contradiction',
                                detection_context='consolidation',
                                reasoning=conflict.get('reasoning'),
                                temporal_signal=conflict.get('temporal_signal', False),
                                surface_context=conflict.get('surface_context'),
                            )
                            logging.info(
                                f"[CONSOLIDATION] Contradiction detected on concept "
                                f"'{existing_concept['concept_name']}': {conflict.get('reasoning', '')[:80]}"
                            )
                except Exception as ue:
                    logging.debug(f"[CONSOLIDATION] Contradiction check skipped: {ue}")

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

        Uses sqlite-vec virtual table for vector similarity search, combined with
        exact name matching for hybrid retrieval.

        Args:
            concept_name: Name of concept
            embedding: Embedding vector
            concept_type: Type of concept

        Returns:
            Existing concept dict or None
        """
        try:
            with self.storage.db_service.connection() as conn:
                cursor = conn.cursor()

                similarity_threshold = self.config['similarity_threshold']
                packed = _pack_embedding(embedding)

                # Step 1: Check for exact name match (highest priority)
                cursor.execute("""
                    SELECT
                        id, concept_name, concept_type, definition,
                        abstraction_level, domain, strength, confidence
                    FROM semantic_concepts
                    WHERE deleted_at IS NULL
                      AND concept_type = ?
                      AND LOWER(concept_name) = LOWER(?)
                    LIMIT 1
                """, (concept_type, concept_name))

                row = cursor.fetchone()
                if row:
                    cursor.close()
                    return {
                        'id': str(row[0]),
                        'concept_name': row[1],
                        'concept_type': row[2],
                        'definition': row[3],
                        'embedding': None,
                        'abstraction_level': row[4],
                        'domain': row[5],
                        'strength': row[6],
                        'confidence': row[7]
                    }

                # Step 2: Vector similarity search via sqlite-vec virtual table
                # sqlite-vec distance is L2 (Euclidean); we retrieve top-K and
                # compute cosine similarity in Python to honour the threshold.
                cursor.execute("""
                    SELECT
                        sc.id, sc.concept_name, sc.concept_type, sc.definition,
                        sc.abstraction_level, sc.domain, sc.strength, sc.confidence,
                        v.distance
                    FROM semantic_concepts_vec v
                    JOIN semantic_concepts sc ON sc.rowid = v.rowid
                    WHERE v.embedding MATCH ?
                      AND k = 5
                      AND sc.deleted_at IS NULL
                      AND sc.concept_type = ?
                """, (packed, concept_type))

                rows = cursor.fetchall()
                cursor.close()

                if not rows:
                    return None

                # Pick best match above threshold.
                # sqlite-vec returns L2 distance; convert to a 0-1 similarity
                # approximation: sim ~ 1 / (1 + distance).
                best = None
                best_sim = 0.0
                for r in rows:
                    distance = r[8] if r[8] is not None else float('inf')
                    sim = 1.0 / (1.0 + distance)
                    if sim > similarity_threshold and sim > best_sim:
                        best_sim = sim
                        best = r

                if best is None:
                    return None

                return {
                    'id': str(best[0]),
                    'concept_name': best[1],
                    'concept_type': best[2],
                    'definition': best[3],
                    'embedding': None,
                    'abstraction_level': best[4],
                    'domain': best[5],
                    'strength': best[6],
                    'confidence': best[7]
                }

        except Exception as e:
            logging.error(f"Failed to find similar concept: {e}")
            return None

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
