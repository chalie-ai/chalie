"""
Autobiography Skill — Retrieve the user's synthesized narrative.

Pure database retrieval (non-LLM, sub-cortical service).
Returns the accumulated understanding of the user as coherent narrative.
"""

import logging
import re
from typing import Optional
from sqlalchemy import text

logger = logging.getLogger(__name__)


def handle_autobiography(topic: str, params: dict) -> str:
    """
    Retrieve the user's synthesized autobiography.

    Optional section parameter to extract one specific section:
    - "identity"
    - "relationship_arc"
    - "values_and_goals"
    - "behavioral_patterns"
    - "active_threads"
    - "long_term_themes"
    - "delta" / "growth" — show growth delta between last two versions

    Args:
        topic: Current conversation topic (unused)
        params: {section (optional): specific section to extract}

    Returns:
        Formatted narrative string or "not synthesized" message
    """
    try:
        from services.database_service import DatabaseService, get_merged_db_config

        db_config = get_merged_db_config()
        db = DatabaseService(db_config)

        user_id = "primary"

        # Handle delta/growth section specially
        section = params.get("section", "").lower()
        if section in ("delta", "growth"):
            return _get_delta_summary(db, user_id)

        with db.get_session() as session:
            result = session.execute(
                text("""
                SELECT narrative FROM autobiography
                WHERE user_id = :user_id
                ORDER BY version DESC
                LIMIT 1
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()

            if not row or not row[0]:
                return "[AUTOBIOGRAPHY] No autobiography synthesized yet."

            narrative = row[0]

            # Extract section if requested
            if section:
                extracted = _extract_section(narrative, section)
                if extracted:
                    return extracted
                else:
                    return f"[AUTOBIOGRAPHY] Section '{section}' not found in narrative."

            return narrative

    except Exception as e:
        logger.error(f"[AUTOBIOGRAPHY SKILL] Error: {e}", exc_info=True)
        return f"[AUTOBIOGRAPHY] Error retrieving narrative: {e}"


def _get_delta_summary(db, user_id: str) -> str:
    """
    Retrieve and format the growth delta between the last two autobiography versions.

    Args:
        db: DatabaseService instance
        user_id: User identifier

    Returns:
        Formatted delta summary string
    """
    import json as _json
    try:
        with db.get_session() as session:
            result = session.execute(
                text("""
                SELECT version, delta_summary
                FROM autobiography
                WHERE user_id = :user_id AND delta_summary IS NOT NULL
                ORDER BY version DESC
                LIMIT 1
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()

            if not row or not row[1]:
                return "[AUTOBIOGRAPHY] No growth delta computed yet. Run another synthesis to generate one."

            version = row[0]
            delta = row[1] if isinstance(row[1], dict) else _json.loads(row[1])

            section_deltas = delta.get('section_deltas', {})
            from_v = delta.get('from_version', '?')
            to_v = delta.get('to_version', version)

            if not section_deltas:
                return f"[AUTOBIOGRAPHY] No changes detected between v{from_v} and v{to_v}."

            lines = [f"## Growth Delta (v{from_v} → v{to_v})"]
            for section, summary in section_deltas.items():
                display = section.replace('_', ' ').title()
                lines.append(f"\n**{display}**: {summary}")

            return "\n".join(lines)

    except Exception as e:
        logger.error(f"[AUTOBIOGRAPHY SKILL] Delta retrieval error: {e}", exc_info=True)
        return f"[AUTOBIOGRAPHY] Error retrieving growth delta: {e}"


def _extract_section(narrative: str, section: str) -> Optional[str]:
    """
    Extract a specific section from the narrative using ## header markers.

    Args:
        narrative: Full narrative text
        section: Section name to extract (identity, relationship_arc, etc.)

    Returns:
        Section content or None if not found
    """
    # Normalize section name: replace underscores with spaces and title case
    section_display = section.replace("_", " ").title()

    # Find the section header
    pattern = rf"##\s+{re.escape(section_display)}\s*\n(.*?)(?=##|\Z)"
    match = re.search(pattern, narrative, re.IGNORECASE | re.DOTALL)

    if match:
        return f"## {section_display}\n{match.group(1).strip()}"

    return None
