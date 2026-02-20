"""
Semantic Consolidation Worker - Processes episodes for concept extraction.
Listens to: semantic_consolidation_queue
"""

import logging
import json
from services.database_service import DatabaseService
from services.config_service import ConfigService
from services.llm_service import create_llm_service
from services.semantic_storage_service import SemanticStorageService
from services.semantic_consolidation_service import SemanticConsolidationService


class SemanticConsolidationWorker:
    """Worker that extracts semantic concepts from episodes."""

    def __init__(self):
        """Initialize worker with required services."""
        # Load configs
        from services.database_service import get_merged_db_config

        semantic_config = ConfigService.resolve_agent_config("semantic-memory")
        db_config = get_merged_db_config()

        # Initialize services
        self.db_service = DatabaseService(db_config)
        self.llm_service = create_llm_service(semantic_config)
        self.storage_service = SemanticStorageService(self.db_service)
        self.consolidation_service = SemanticConsolidationService(
            self.llm_service,
            self.storage_service,
            ConfigService
        )

    def process(self, message_data: dict) -> dict:
        """
        Process episode for semantic extraction (STORY-12: supports batch mode).

        Args:
            message_data: Dict with 'episode' key or 'type'='batch_consolidation'

        Returns:
            Dict with processing results
        """
        try:
            # Check if this is a batch consolidation job (STORY-12)
            if message_data.get('type') == 'batch_consolidation':
                return self._process_batch_consolidation()

            # Single episode processing
            episode = message_data.get('episode')
            if not episode:
                logging.error("No episode data in message")
                return {'status': 'error', 'message': 'No episode data'}

            episode_id = episode.get('id')
            logging.info(f"Processing episode {episode_id} for semantic extraction")

            # Extract concepts and relationships
            extracted = self.consolidation_service.extract_from_episode(episode)

            concepts_count = len(extracted.get('concepts', []))
            relationships_count = len(extracted.get('relationships', []))

            if concepts_count == 0:
                logging.info(f"No concepts extracted from episode {episode_id}")
                return {
                    'status': 'success',
                    'episode_id': episode_id,
                    'concepts_created': 0,
                    'relationships_created': 0
                }

            # Consolidate concepts (match or create)
            concept_name_to_id = {}
            for concept in extracted.get('concepts', []):
                try:
                    concept_id = self.consolidation_service.consolidate_concept(
                        concept,
                        episode_id
                    )
                    concept_name_to_id[concept['name']] = concept_id
                except Exception as e:
                    logging.error(f"Failed to consolidate concept '{concept.get('name')}': {e}")

            # Consolidate relationships
            relationships_created = 0
            for relationship in extracted.get('relationships', []):
                try:
                    rel_id = self.consolidation_service.consolidate_relationship(
                        relationship,
                        episode_id,
                        concept_name_to_id
                    )
                    if rel_id:
                        relationships_created += 1
                except Exception as e:
                    logging.error(f"Failed to consolidate relationship: {e}")

            logging.info(
                f"Semantic extraction complete for episode {episode_id}: "
                f"{len(concept_name_to_id)} concepts, {relationships_created} relationships"
            )

            return {
                'status': 'success',
                'episode_id': episode_id,
                'concepts_created': len(concept_name_to_id),
                'relationships_created': relationships_created
            }

        except Exception as e:
            logging.error(f"Semantic consolidation worker error: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def _process_batch_consolidation(self) -> dict:
        """
        Process batch consolidation for idle periods (STORY-12).

        Queries episodes without semantic consolidation and processes up to 20.

        Returns:
            Dict with batch processing results
        """
        logging.info("[SEMANTIC CONSOLIDATION] Processing batch consolidation job")

        try:
            # Query episodes without semantic consolidation
            episodes = []
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, gist, intent, context, action, outcome, salience, topic, salience_factors
                    FROM episodes
                    WHERE (semantic_consolidation_status IS NULL
                           OR semantic_consolidation_status IN ('empty', 'failed'))
                      AND deleted_at IS NULL
                    ORDER BY created_at ASC
                    LIMIT 20
                """)
                columns = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    episodes.append(dict(zip(columns, row)))
                cursor.close()

            if not episodes:
                logging.info("[SEMANTIC CONSOLIDATION] No unconsolidated episodes found")
                return {
                    'status': 'success',
                    'batch_size': 0,
                    'episodes_processed': 0
                }

            logging.info(f"[SEMANTIC CONSOLIDATION] Processing batch of {len(episodes)} episodes")

            total_concepts = 0
            total_relationships = 0
            episodes_processed = 0

            for episode_row in episodes:
                try:
                    # Convert database row to episode dict
                    salience_factors = episode_row.get('salience_factors') or {}
                    if isinstance(salience_factors, str):
                        salience_factors = json.loads(salience_factors)

                    salience = episode_row.get('salience', 5)
                    # Promotion boost: frequently-retrieved tool_reflection episodes
                    # graduate to semantic memory faster (+2 salience)
                    if (salience_factors.get('source') == 'tool_reflection'
                            and salience_factors.get('retrieval_count', 0) >= 3):
                        salience = min(10, salience + 2)

                    episode = {
                        'id': episode_row.get('id'),
                        'gist': episode_row.get('gist'),
                        'intent': episode_row.get('intent'),
                        'context': episode_row.get('context'),
                        'action': episode_row.get('action'),
                        'outcome': episode_row.get('outcome'),
                        'salience': salience,
                        'topic': episode_row.get('topic')
                    }

                    # Process single episode
                    message_data = {'episode': episode}
                    result = self.process(message_data)

                    if result['status'] == 'success':
                        concepts_created = result.get('concepts_created', 0)
                        rels_created = result.get('relationships_created', 0)
                        total_concepts += concepts_created
                        total_relationships += rels_created
                        episodes_processed += 1

                        # Only mark as 'completed' if concepts were actually extracted.
                        # Mark as 'empty' if the LLM returned nothing (likely timeout/error).
                        # 'empty' episodes will be retried on the next batch cycle.
                        status = 'completed' if concepts_created > 0 else 'empty'
                        with self.db_service.connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE episodes SET semantic_consolidation_status = %s WHERE id = %s",
                                (status, episode.get('id'))
                            )
                            cursor.close()

                        if status == 'empty':
                            logging.info(
                                f"[SEMANTIC CONSOLIDATION] Episode {episode.get('id')} extracted 0 concepts, "
                                f"marked as 'empty' for retry"
                            )
                    else:
                        # Explicit failure — mark as 'failed' for retry
                        with self.db_service.connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE episodes SET semantic_consolidation_status = 'failed' WHERE id = %s",
                                (episode.get('id'),)
                            )
                            cursor.close()
                        logging.warning(
                            f"Failed to process episode {episode.get('id')}: "
                            f"{result.get('message')}"
                        )

                except Exception as e:
                    logging.error(
                        f"Error processing episode in batch: {e}",
                        exc_info=True
                    )

            # Update tracker after batch processing
            from services.semantic_consolidation_tracker import SemanticConsolidationTracker
            tracker = SemanticConsolidationTracker()
            tracker.reset_episode_count()

            logging.info(
                f"[SEMANTIC CONSOLIDATION] Batch consolidation complete: "
                f"{episodes_processed}/{len(episodes)} episodes, "
                f"{total_concepts} concepts, {total_relationships} relationships"
            )

            return {
                'status': 'success',
                'batch_size': len(episodes),
                'episodes_processed': episodes_processed,
                'concepts_created': total_concepts,
                'relationships_created': total_relationships
            }

        except Exception as e:
            logging.error(f"Batch consolidation error: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}


# Worker function for consumer.py (matches pattern of other workers)
def semantic_consolidation_worker(job_data: dict) -> str:
    """
    Process episode for semantic extraction.

    Args:
        job_data: Dict with 'episode' key containing episode data

    Returns:
        Result string
    """
    worker = SemanticConsolidationWorker()
    result = worker.process(job_data)

    if result['status'] == 'success':
        return f"Extracted {result.get('concepts_created', 0)} concepts and {result.get('relationships_created', 0)} relationships from episode {result.get('episode_id')}"
    else:
        return f"Error: {result.get('message', 'Unknown error')}"
