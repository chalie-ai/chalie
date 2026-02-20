"""
Memorize Skill — Memory storage for gists and facts.

Stores information to short-term (gists) and/or medium-term (facts) memory.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def handle_memorize(topic: str, params: dict) -> str:
    """
    Store information to gists and/or facts memory.

    Args:
        topic: Current conversation topic
        params: {gists (optional), facts (optional)}
            gists: [{content, type, confidence}]
            facts: [{key, value, confidence}]

    Returns:
        Confirmation with count stored
    """
    gists = params.get("gists", [])
    facts = params.get("facts", [])

    if not gists and not facts:
        return "[MEMORIZE] Error: no gists or facts specified."

    stored_gists = 0
    stored_facts = 0

    if gists:
        stored_gists = _store_gists(topic, gists)

    if facts:
        stored_facts = _store_facts(topic, facts)

    parts = []
    if stored_gists:
        parts.append(f"{stored_gists} gist(s)")
    if stored_facts:
        parts.append(f"{stored_facts} fact(s)")

    return f"[MEMORIZE] Stored {' and '.join(parts)} for topic '{topic}'."


def _store_gists(topic: str, gists: list) -> int:
    """Store gists via GistStorageService."""
    try:
        from services.gist_storage_service import GistStorageService

        service = GistStorageService()
        stored = 0

        for gist in gists:
            content = gist.get("content")
            if not content:
                continue

            gist_type = gist.get("type", "general")
            confidence = gist.get("confidence", 7)

            # Validate confidence range (0-10)
            confidence = max(0, min(10, int(confidence)))

            service.store_gists(
                topic=topic,
                gists=[{"content": content, "type": gist_type, "confidence": confidence}],
                prompt="",
                response="",
            )
            stored += 1

        return stored

    except Exception as e:
        logger.error(f"[MEMORIZE] Failed to store gists: {e}")
        return 0


def _store_facts(topic: str, facts: list) -> int:
    """Store facts via FactStoreService."""
    try:
        from services.fact_store_service import FactStoreService

        service = FactStoreService()
        stored = 0

        for fact in facts:
            key = fact.get("key")
            value = fact.get("value")
            if not key or value is None:
                continue

            confidence = fact.get("confidence", 0.7)

            # Validate confidence range (0.0-1.0)
            confidence = max(0.0, min(1.0, float(confidence)))

            service.store_fact(
                topic=topic,
                key=key,
                value=value,
                confidence=confidence,
            )
            stored += 1

        return stored

    except Exception as e:
        logger.error(f"[MEMORIZE] Failed to store facts: {e}")
        return 0
