"""
Associate Skill — Spreading activation through the semantic graph.

Graph-based concept traversal. Runs spreading activation from seed concepts,
surfacing related ideas through associative links including creative leaps.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def handle_associate(topic: str, params: dict) -> str:
    """
    Run spreading activation from seed concepts through semantic graph.

    Args:
        topic: Current conversation topic
        params: {seeds (required), depth (optional), include_weak (optional)}

    Returns:
        List of activated concepts with paths, scores, and relationship types
    """
    seeds = params.get("seeds", [])
    if not seeds:
        return "[ASSOCIATE] Error: no seed concepts specified."

    # Load config for tunable parameters
    config = _load_config()
    depth = params.get("depth", config.get("max_depth", 2))
    include_weak = params.get("include_weak", True)

    try:
        from services.semantic_retrieval_service import SemanticRetrievalService
        from services.embedding_service import get_embedding_service
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        embedding_service = get_embedding_service()
        retrieval_service = SemanticRetrievalService(db_service, embedding_service)

        # Resolve seed names to concept IDs
        seed_ids = _resolve_seeds(retrieval_service, seeds)

        if not seed_ids:
            return f"[ASSOCIATE] No concepts found matching seeds: {seeds}"

        # Run spreading activation
        activated = retrieval_service.spreading_activation(
            seed_concepts=seed_ids,
            max_depth=depth,
        )

        if not activated:
            return f"[ASSOCIATE] Spreading activation from {seeds} produced no results."

        # Filter weak links if requested
        if not include_weak:
            activated = [
                c for c in activated
                if c.get("activation_score", 0) >= config.get("min_activation_threshold", 0.3)
            ]

        return _format_activated(activated, seeds)

    except Exception as e:
        logger.error(f"[ASSOCIATE] Failed: {e}")
        return f"[ASSOCIATE] Error: {e}"


def _load_config() -> dict:
    """Load associate config from frontal-cortex.json innate_skills section."""
    try:
        from services.config_service import ConfigService

        config = ConfigService.get_agent_config("frontal-cortex")
        return config.get("innate_skills", {}).get("associate", {
            "random_activation_ratio": 0.15,
            "activation_decay": 0.7,
            "min_activation_threshold": 0.3,
            "max_depth": 2,
        })
    except Exception:
        return {
            "random_activation_ratio": 0.15,
            "activation_decay": 0.7,
            "min_activation_threshold": 0.3,
            "max_depth": 2,
        }


def _resolve_seeds(retrieval_service, seeds: List[str]) -> List[str]:
    """Resolve seed concept names/queries to concept IDs."""
    seed_ids = []
    for seed in seeds:
        concepts = retrieval_service.retrieve_concepts(query=seed, limit=1)
        if concepts:
            seed_ids.append(concepts[0]["id"])
        else:
            logger.debug(f"[ASSOCIATE] No concept found for seed: {seed}")
    return seed_ids


def _format_activated(activated: List[Dict], seeds: List[str]) -> str:
    """Format activated concepts with scores and relationship context."""
    lines = [f"[ASSOCIATE] {len(activated)} concepts activated from seeds {seeds}:"]

    for concept in activated:
        name = concept.get("concept_name", concept.get("name", "Unknown"))
        score = concept.get("activation_score", 0)
        strength = concept.get("strength", 0)
        definition = concept.get("definition", "")[:100]

        strength_label = "strong" if strength >= 0.5 else "weak"
        lines.append(
            f"  - {name} (activation={score:.2f}, strength={strength:.2f}, {strength_label})"
        )
        if definition:
            lines.append(f"    {definition}")

    return "\n".join(lines)
