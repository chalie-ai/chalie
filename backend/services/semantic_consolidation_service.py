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
from typing import Optional
from services.config_service import ConfigService


class SemanticConsolidationService:
    """Extracts semantic concepts and relationships from episodic memory."""

    def __init__(
        self,
        ollama_service,
        storage_service,
        config_service: ConfigService
    ):
        """
        Initialize consolidation service.

        Args:
            ollama_service: LLM service for inference
            storage_service: KnowledgeService (or legacy SemanticService) for persistence
            config_service: ConfigService for semantic memory config
        """
        self.ollama = ollama_service
        self.storage = storage_service
        self.config = config_service.get_agent_config("semantic-memory")
        self.prompt_template = self._load_prompt_template()

        # Lazy-loaded KnowledgeService (resolved on first use)
        self._knowledge_svc = None

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

    def _get_knowledge_svc(self):
        """Lazy-load KnowledgeService (one per consolidation service instance)."""
        if self._knowledge_svc is None:
            try:
                from services.knowledge_service import KnowledgeService
                # Resolve db_service from the storage object
                db = getattr(self.storage, 'db', None) or getattr(self.storage, 'db_service', None)
                self._knowledge_svc = KnowledgeService(db)
            except Exception as e:
                logging.error(f"[CONSOLIDATION] Failed to init KnowledgeService: {e}")
                return None
        return self._knowledge_svc

    def consolidate_concept(self, concept: dict, episode_id: str) -> str:
        """
        Consolidate a concept (match existing or create new) via KnowledgeService.

        Args:
            concept: Extracted concept dict (name, type, definition, abstraction_level, domain)
            episode_id: Source episode UUID

        Returns:
            ID/key of concept (existing or new)
        """
        try:
            ks = self._get_knowledge_svc()
            if ks is None:
                raise RuntimeError("KnowledgeService unavailable")

            concept_name = concept['name']
            concept_type = concept['type']
            definition = concept['definition']

            # Generate embedding for concept
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            concept_text = f"{concept_name} ({concept_type}): {definition}"
            embedding = emb_service.generate_embedding(concept_text)

            # Search for similar existing concepts via KnowledgeService
            existing_concept = self._find_similar_concept(
                concept_name=concept_name,
                concept_type=concept_type,
            )

            if existing_concept:
                # Strengthen existing concept via KnowledgeService
                ks.strengthen(entity='chalie', key=concept_name, episode_id=episode_id)
                logging.info(f"Strengthened existing concept: {concept_name}")
                return existing_concept.get('id', concept_name)
            else:
                # Create new concept via KnowledgeService
                # KnowledgeService.store() handles embedding, signals, and UPSERT
                data = {
                    'domain': concept.get('domain'),
                    'concept_type': concept_type,
                    'abstraction_level': concept.get('abstraction_level', 3),
                    'source_episodes': [episode_id],
                    'examples': concept.get('examples'),
                    'context_constraints': concept.get('context_constraints'),
                }

                result = ks.store(
                    kind='concept',
                    entity='chalie',
                    key=concept_name,
                    value=definition,
                    data=data,
                    decay_class='standard',
                    confidence=0.5,
                    source='consolidation',
                    embedding=embedding,
                )

                if result:
                    concept_id = result.get('id', concept_name)
                    logging.info(f"Created new concept: {concept_name}")
                    # KnowledgeService.store() already emits new_knowledge signal
                    return concept_id
                else:
                    logging.warning(f"[CONSOLIDATION] KnowledgeService.store returned None for '{concept_name}'")
                    return concept_name

        except Exception as e:
            logging.error(f"Failed to consolidate concept '{concept.get('name', 'unknown')}': {e}")
            raise

    def _find_similar_concept(self, concept_name: str, concept_type: str = None) -> Optional[dict]:
        """
        Find similar concept via KnowledgeService (exact key match + RRF recall).

        Args:
            concept_name: Name of concept
            concept_type: Type of concept (used for post-filtering)

        Returns:
            Existing knowledge row dict or None
        """
        try:
            ks = self._get_knowledge_svc()
            if ks is None:
                return None

            # Step 1: Exact key match (highest priority)
            exact = ks.get(entity='chalie', key=concept_name)
            if exact and exact.get('kind') == 'concept':
                data = exact.get('data') or {}
                if concept_type and data.get('concept_type') and data['concept_type'] != concept_type:
                    pass  # type mismatch, fall through to semantic search
                else:
                    return exact

            # Step 2: Semantic recall via RRF (embedding + FTS + exact)
            results = ks.recall(
                query=concept_name,
                kinds=['concept'],
                entity='chalie',
                limit=5,
            )

            if not results:
                return None

            # Pick best match
            for r in results:
                # RRF score is not directly comparable to cosine sim, but higher is better.
                # Use a heuristic: RRF > 0.01 is roughly equivalent to strong match.
                rrf = r.get('rrf_score', 0)
                if rrf > 0.005:
                    data = r.get('data') or {}
                    if concept_type and data.get('concept_type') and data['concept_type'] != concept_type:
                        continue
                    return r

            return None

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
        Consolidate a relationship via KnowledgeService.

        Args:
            relationship: Extracted relationship dict (source, target, type, strength)
            episode_id: Source episode UUID
            concept_name_to_id: Mapping from concept names to IDs/keys

        Returns:
            Key of relationship, or None if concepts not found
        """
        try:
            ks = self._get_knowledge_svc()
            if ks is None:
                logging.warning("[CONSOLIDATION] KnowledgeService unavailable for relationship storage")
                return None

            source_name = relationship['source']
            target_name = relationship['target']

            # Verify both concepts exist (either as IDs or concept names)
            source_id = concept_name_to_id.get(source_name)
            target_id = concept_name_to_id.get(target_name)

            if not source_id or not target_id:
                logging.warning(
                    f"Cannot create relationship: concept not found "
                    f"(source={source_name}, target={target_name})"
                )
                return None

            rel_type = relationship['type']
            rel_key = f"{source_name}:{rel_type}:{target_name}"

            result = ks.store(
                kind='relationship',
                entity='chalie',
                key=rel_key,
                value=rel_type,
                data={
                    'source_key': source_name,
                    'target_key': target_name,
                    'relationship_type': rel_type,
                    'bidirectional': relationship.get('bidirectional', False),
                    'strength': relationship.get('strength', 0.5),
                    'source_episodes': [episode_id],
                },
                source='consolidation',
            )

            if result:
                return result.get('id', rel_key)
            return None

        except Exception as e:
            logging.error(f"Failed to consolidate relationship: {e}")
            return None
