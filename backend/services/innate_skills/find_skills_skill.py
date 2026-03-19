"""
Find Skills Skill — On-demand innate skill discovery.

Cognitive primitive: always available in ACT mode. Lets Chalie discover
contextual innate skills (schedule, list, focus, document, etc.) by semantic
query instead of having all skill docs pre-injected into the prompt.

Searches `tool_capability_profiles_vec` WHERE `tool_type = 'skill'`, then
filters results to CONTEXTUAL_SKILLS only — primitives (recall, memorize,
associate, find_tools, find_skills) are already injected and need no
discovery. Returns full .md documentation for each match so the skill
can be used immediately without a follow-up lookup.

Zero-tool-name references in infrastructure — fully tool-agnostic.
"""

import logging
import os
import struct
from typing import List, Dict

logger = logging.getLogger(__name__)

LOG_PREFIX = "[FIND_SKILLS]"

# Resolved once at import time: backend/prompts/skills/
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'prompts', 'skills',
)


def handle_find_skills(topic: str, params: dict) -> str:
    """
    Discover contextual innate skills by semantic query.

    Args:
        topic: Current conversation topic (unused, present for handler contract)
        params: {
            query: str (required) — natural language description of what you want to do
            limit: int (optional, default 3, max 5)
        }

    Returns:
        Full .md documentation for matching skills, or a descriptive error string.
        Never raises.
    """
    query = params.get("query", "").strip()
    if not query:
        return f"{LOG_PREFIX} Error: 'query' parameter is required."

    limit = min(int(params.get("limit", 3)), 5)

    # Attempt semantic search via embedding
    try:
        from services.embedding_service import EmbeddingService
        embedding = EmbeddingService().generate_embedding(query)
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        matches = _vec_search(blob, query, limit)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Embedding generation failed: {e}")
        matches = _fallback_keyword_search(query, limit)

    if not matches:
        return (
            f"{LOG_PREFIX} No matching skills found for '{query}'. "
            "The cognitive primitives (recall, memorize, associate, find_tools, find_skills) "
            "are already available."
        )

    return _format_results(query, matches)


# ── Vector search ──────────────────────────────────────────────────────────


def _vec_search(blob: bytes, query: str, limit: int) -> List[Dict]:
    """Query tool_capability_profiles_vec for skill nearest-neighbours."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tcp.tool_name, tcp.tool_type, v.distance
                FROM tool_capability_profiles_vec v
                JOIN tool_capability_profiles tcp ON tcp.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                  AND tcp.tool_type = 'skill'
                ORDER BY v.distance
                """,
                (blob, limit + 10),  # over-fetch to allow filtering
            )
            rows = cursor.fetchall()
            cursor.close()

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Vector search failed: {e}")
        return _fallback_keyword_search(query, limit)

    return _filter_to_contextual(rows, limit)


def _filter_to_contextual(rows: list, limit: int) -> List[Dict]:
    """Keep only CONTEXTUAL_SKILLS — primitives are already injected."""
    try:
        from services.innate_skills.registry import CONTEXTUAL_SKILLS
    except Exception:
        # If registry is unavailable, pass everything through
        CONTEXTUAL_SKILLS = None

    results = []
    for row in rows:
        name = row[0] if not isinstance(row, dict) else row['tool_name']
        if CONTEXTUAL_SKILLS is not None and name not in CONTEXTUAL_SKILLS:
            continue
        results.append({'skill_name': name})
        if len(results) >= limit:
            break

    return results


# ── Keyword fallback ───────────────────────────────────────────────────────


def _fallback_keyword_search(query: str, limit: int) -> List[Dict]:
    """Keyword-based fallback when the embedding service is unavailable."""
    try:
        from services.database_service import get_shared_db_service
        from services.innate_skills.registry import CONTEXTUAL_SKILLS
        db = get_shared_db_service()

        pattern = f"%{query}%"
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tool_name
                FROM tool_capability_profiles
                WHERE tool_type = 'skill'
                  AND (short_summary LIKE ? OR full_profile LIKE ? OR tool_name LIKE ?)
                ORDER BY tool_name
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit + 10),
            )
            rows = cursor.fetchall()
            cursor.close()

        results = []
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row['tool_name']
            if name not in CONTEXTUAL_SKILLS:
                continue
            results.append({'skill_name': name})
            if len(results) >= limit:
                break

        return results

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return []


# ── Result formatting ──────────────────────────────────────────────────────


def _format_results(query: str, matches: List[Dict]) -> str:
    """Load full .md docs for each match and return concatenated output."""
    parts = [f"{LOG_PREFIX} Found {len(matches)} skill(s) matching '{query}':\n"]

    for match in matches:
        name = match['skill_name']
        doc = _load_skill_doc(name)
        if doc is None:
            # Doc file missing — already logged; skip silently
            continue
        parts.append("────────────────────────────────")
        parts.append(f"## {name}")
        parts.append(doc)

    if len(parts) == 1:
        # All doc loads failed
        return (
            f"{LOG_PREFIX} No matching skills found for '{query}'. "
            "The cognitive primitives (recall, memorize, associate, find_tools, find_skills) "
            "are already available."
        )

    return "\n".join(parts)


def _load_skill_doc(skill_name: str) -> str | None:
    """
    Load full documentation from backend/prompts/skills/{skill_name}.md.

    Returns the file contents, or None if the file does not exist.
    Logs a warning on missing files but never raises.
    """
    path = os.path.join(_PROMPTS_DIR, f'{skill_name}.md')
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"{LOG_PREFIX} Skill doc missing for '{skill_name}' at {path}")
        return None
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to read skill doc for '{skill_name}': {e}")
        return None
