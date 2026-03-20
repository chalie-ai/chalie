"""
Memorize Skill — Explicit memory storage for user traits.

Stores information as user traits for persistent recall.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "memorize",
    "description": (
        "Store information as persistent user traits for future recall. "
        "Use when you want to record something about the user for long-term reference. "
        "Each trait has a key, value, confidence, and category. "
        "Categories: 'core' (identity facts like name), 'preference' (likes/dislikes), "
        "'behavioral' (observed patterns). The 'facts' format is also accepted for "
        "backward compatibility and maps to traits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "traits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Trait identifier (e.g. 'name', 'favorite_color').",
                        },
                        "value": {
                            "type": "string",
                            "description": "Trait value (e.g. 'Dylan', 'blue').",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence 0.0-1.0 (default 0.7).",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["core", "preference", "behavioral"],
                            "description": "Trait category (default 'preference').",
                        },
                    },
                    "required": ["key", "value"],
                },
                "description": "List of traits to store: [{key, value, confidence?, category?}].",
            },
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Fact key.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Fact value.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence 0.0-1.0 (default 0.7).",
                        },
                    },
                    "required": ["key", "value"],
                },
                "description": "Legacy format — list of facts [{key, value, confidence?}]. Mapped to traits with category 'preference'.",
            },
        },
        "required": [],
    },
}


def handle_memorize(topic: str, params: dict) -> str:
    """
    Store information as user traits.

    Accepts both the new 'traits' format and backward-compatible 'facts'/'gists'
    formats, routing all of them to UserTraitService.

    Args:
        topic: Current conversation topic
        params: {
            traits (optional): [{key, value, confidence, category}]
            facts (optional): [{key, value, confidence}]  — mapped to traits
            gists (optional): ignored (ephemeral context, not persistently stored)
        }

    Returns:
        Confirmation with count stored
    """
    traits = params.get("traits", [])
    facts = params.get("facts", [])

    # Map legacy 'facts' format to traits
    for fact in facts:
        key = fact.get("key")
        value = fact.get("value")
        if key and value is not None:
            traits.append({
                "key": key,
                "value": str(value),
                "confidence": fact.get("confidence", 0.7),
                "category": "preference",
            })

    if not traits:
        return "[MEMORIZE] Error: no traits specified."

    stored = _store_traits(topic, traits)

    if stored:
        return f"[MEMORIZE] Stored {stored} trait(s) for topic '{topic}'."
    return "[MEMORIZE] Nothing stored — check trait format."


def _store_traits(topic: str, traits: list) -> int:
    """Store traits via UserTraitService."""
    try:
        from services.database_service import get_shared_db_service
        from services.user_trait_service import UserTraitService

        db_service = get_shared_db_service()
        service = UserTraitService(db_service)
        stored = 0

        for trait in traits:
            key = trait.get("key")
            value = trait.get("value")
            if not key or value is None:
                continue

            confidence = max(0.0, min(1.0, float(trait.get("confidence", 0.7))))
            category = trait.get("category", "preference")

            # Validate category against allowed values
            allowed_categories = {"core", "preference", "behavioral"}
            if category not in allowed_categories:
                category = "preference"

            service.store_trait(
                trait_key=key,
                trait_value=str(value),
                confidence=confidence,
                category=category,
            )
            stored += 1

        return stored

    except Exception as e:
        logger.error(f"[MEMORIZE] Failed to store traits: {e}")
        return 0
