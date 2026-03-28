"""ContactResolver — passive people index from IMAP and CalDAV data.

Mines email sender headers and calendar attendees into KnowledgeService
as ``kind='relationship'``, ``entity='person'`` entries. Provides
:func:`resolve` for cross-capability identity lookup (e.g. turning a
raw email address into a display name, or vice versa).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _get_knowledge_service():
    """Lazy-import KnowledgeService to avoid circular imports at module load."""
    from services.database_service import get_shared_db_service
    from services.knowledge_service import KnowledgeService
    return KnowledgeService(get_shared_db_service())


def index_person(email: str, name: str | None = None, source: str = "unknown") -> None:
    """Store or reinforce a person identity in the knowledge graph.

    Uses ``kind='relationship'``, ``entity='person'``, ``key=<email>``,
    ``value=<display_name>``, ``decay_class='slow'``.  KnowledgeService
    UPSERTs on ``(entity, key)`` so repeated sightings reinforce confidence.
    """
    display = (name or "").strip()
    if not email or "@" not in str(email) or not display:
        return
    try:
        _get_knowledge_service().store(
            kind="relationship",
            entity="person",
            key=email.strip().lower(),
            value=display,
            data={"source": source},
            decay_class="slow",
            confidence=0.6,
            source=source,
        )
    except Exception as exc:
        logger.debug("[contact_resolver] index_person failed for %s: %s", email, exc)


def resolve(identifier: str, limit: int = 5) -> list[dict]:
    """Look up person entries matching *identifier*.

    Uses KnowledgeService.recall() with RRF across exact key match,
    FTS5 full-text, and vector KNN.
    """
    if not identifier or not str(identifier).strip():
        return []
    try:
        rows = _get_knowledge_service().recall(
            query=identifier.strip(),
            kinds=["relationship"],
            entity="person",
            limit=limit,
        )
        return [{"email": r["key"], "name": r.get("value", "")} for r in rows if r.get("key")]
    except Exception as exc:
        logger.debug("[contact_resolver] resolve(%r) failed: %s", identifier, exc)
        return []
