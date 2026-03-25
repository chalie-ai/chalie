"""
Associate Skill — Spreading activation through the knowledge graph.

Graph-based concept traversal. Runs spreading activation from seed concepts,
surfacing related ideas through associative links including creative leaps.
Now operates on the unified knowledge table (kind='relationship').
"""

import logging
import random
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "associate",
    "description": (
        "Spreading activation through the semantic concept graph. "
        "Surfaces related ideas through associative links from seed concepts, "
        "including creative leaps via weak/random associations. "
        "Use when you need to explore what concepts relate to a query, "
        "especially when recall returns sparse results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "seeds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of concept names or queries to activate from.",
            },
            "depth": {
                "type": "integer",
                "description": "Max activation depth in the graph (default 2).",
            },
            "include_weak": {
                "type": "boolean",
                "description": "Include weak/random associations for creative leaps (default true).",
            },
        },
        "required": ["seeds"],
    },
}


def handle_associate(topic: str, params: dict) -> str:
    """
    Run spreading activation from seed concepts through the knowledge graph.

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
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        ks = KnowledgeService(db)

        # Resolve seed names to concept keys
        seed_keys = _resolve_seeds(ks, seeds)

        if not seed_keys:
            return f"[ASSOCIATE] No concepts found matching seeds: {seeds}"

        # Run spreading activation over knowledge table
        activated = _spreading_activation(
            db, ks, seed_keys,
            max_depth=depth,
            decay_factor=config.get("activation_decay", 0.7),
            activation_threshold=config.get("min_activation_threshold", 0.3),
            random_ratio=config.get("random_activation_ratio", 0.15),
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


def _resolve_seeds(ks, seeds: List[str]) -> List[str]:
    """Resolve seed concept names/queries to concept keys in the knowledge table."""
    seed_keys = []
    for seed in seeds:
        concepts = ks.recall(seed, kinds=['concept'], limit=1)
        if concepts:
            seed_keys.append(concepts[0].get("key", ""))
        else:
            logger.debug(f"[ASSOCIATE] No concept found for seed: {seed}")
    return [k for k in seed_keys if k]


def _spreading_activation(
    db,
    ks,
    seed_keys: List[str],
    max_depth: int = 2,
    decay_factor: float = 0.7,
    activation_threshold: float = 0.3,
    random_ratio: float = 0.15,
) -> List[Dict]:
    """
    BFS spreading activation over kind='relationship' entries in the knowledge table.

    Relationship entries are expected to have data like:
      {"source_key": "concept_a", "target_key": "concept_b", "relationship_type": "...", "strength": 0.8}
    """
    import json

    activation_levels = {}  # key -> float
    for key in seed_keys:
        activation_levels[key] = 1.0

    frontier = deque()
    for key in seed_keys:
        frontier.append((key, 0, 1.0))

    visited = set(seed_keys)

    try:
        with db.connection() as conn:
            cursor = conn.cursor()

            while frontier:
                current_key, depth, activation = frontier.popleft()

                if depth >= max_depth:
                    continue

                # Find relationships where this concept is source or target
                cursor.execute("""
                    SELECT data FROM knowledge
                    WHERE kind = 'relationship' AND deleted_at IS NULL
                      AND (
                        json_extract(data, '$.source_key') = ?
                        OR json_extract(data, '$.target_key') = ?
                      )
                """, (current_key, current_key))

                for (raw_data,) in cursor.fetchall():
                    if not raw_data:
                        continue
                    try:
                        rel = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                    except (json.JSONDecodeError, TypeError):
                        continue

                    source_key = rel.get('source_key', '')
                    target_key = rel.get('target_key', '')
                    # Follow the link in the outgoing direction
                    next_key = target_key if source_key == current_key else source_key
                    if not next_key:
                        continue

                    strength = rel.get('strength', 0.5)
                    new_activation = activation * decay_factor

                    # Weak links: random inclusion for creative leaps
                    if strength < 0.5:
                        if random.random() >= random_ratio:
                            continue
                    else:
                        new_activation = new_activation * strength

                    if new_activation < activation_threshold:
                        continue

                    if next_key in activation_levels:
                        activation_levels[next_key] = max(activation_levels[next_key], new_activation)
                    else:
                        activation_levels[next_key] = new_activation

                    if next_key not in visited:
                        visited.add(next_key)
                        frontier.append((next_key, depth + 1, new_activation))

            cursor.close()
    except Exception as e:
        logger.warning(f"[ASSOCIATE] Spreading activation query failed: {e}")

    # Resolve activated keys to full concept entries
    activated_concepts = []
    for key, score in activation_levels.items():
        entry = ks.get('chalie', key)
        if not entry:
            entry = ks.get('user', key)
        if entry:
            entry_copy = dict(entry)
            entry_copy['activation_score'] = score
            activated_concepts.append(entry_copy)

    activated_concepts.sort(key=lambda x: x.get('activation_score', 0), reverse=True)
    return activated_concepts


def _format_activated(activated: List[Dict], seeds: List[str]) -> str:
    """Format activated concepts with scores and relationship context."""
    lines = [f"[ASSOCIATE] {len(activated)} concepts activated from seeds {seeds}:"]

    for concept in activated:
        name = concept.get("key", concept.get("name", "Unknown"))
        score = concept.get("activation_score", 0)
        confidence = concept.get("confidence", 0)
        definition = (concept.get("value", "") or "")[:100]

        strength_label = "strong" if confidence >= 0.5 else "weak"
        lines.append(
            f"  - {name} (activation={score:.2f}, confidence={confidence:.2f}, {strength_label})"
        )
        if definition:
            lines.append(f"    {definition}")

    return "\n".join(lines)
