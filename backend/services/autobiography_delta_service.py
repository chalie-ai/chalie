"""
Autobiography Delta Service - Detect changes between autobiography versions.

Compares section hashes between consecutive versions to surface growth insights
and feed stability signals back to trait reinforcement.
"""

import hashlib
import logging
import re
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


def _hash_section(text: str) -> str:
    """SHA-256 hash of a section's text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _extract_sections(narrative: str) -> Dict[str, str]:
    """
    Parse a narrative into {section_name: content} dict using ## headers.

    Args:
        narrative: Full narrative text

    Returns:
        Dict mapping section name (lowercase) to section content
    """
    sections = {}
    # Split on ## headers, keeping the header in the match
    parts = re.split(r'^(##\s+.+)$', narrative, flags=re.MULTILINE)

    current_section = None
    current_content = []

    for part in parts:
        if part.startswith('## '):
            if current_section is not None:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = part.lstrip('#').strip().lower().replace(' ', '_')
            current_content = []
        else:
            if current_section is not None:
                current_content.append(part)

    if current_section is not None:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections


class AutobiographyDeltaService:
    """Detect and compute deltas between autobiography versions."""

    def __init__(self, db_service):
        """
        Initialize delta service.

        Args:
            db_service: DatabaseService instance
        """
        self.db = db_service

    def get_changed_sections(self, user_id: str = "primary") -> Optional[Dict[str, Any]]:
        """
        Compare section_hashes between the latest two autobiography versions.

        Args:
            user_id: User identifier

        Returns:
            Dict with from_version, to_version, changed: [...], unchanged: [...]
            or None if fewer than 2 versions exist.
        """
        try:
            from sqlalchemy import text

            with self.db.get_session() as session:
                result = session.execute(
                    text("""
                    SELECT id, version, section_hashes
                    FROM autobiography
                    WHERE user_id = :user_id AND section_hashes IS NOT NULL
                    ORDER BY version DESC
                    LIMIT 2
                    """),
                    {"user_id": user_id}
                )
                rows = result.fetchall()

                if len(rows) < 2:
                    return None

                new_row, old_row = rows[0], rows[1]
                new_hashes = new_row[2] or {}
                old_hashes = old_row[2] or {}

                all_sections = set(list(new_hashes.keys()) + list(old_hashes.keys()))
                changed = []
                unchanged = []

                for section in all_sections:
                    new_hash = new_hashes.get(section)
                    old_hash = old_hashes.get(section)
                    if new_hash != old_hash:
                        changed.append(section)
                    else:
                        unchanged.append(section)

                return {
                    "from_version": old_row[1],
                    "to_version": new_row[1],
                    "from_id": str(old_row[0]),
                    "to_id": str(new_row[0]),
                    "changed": sorted(changed),
                    "unchanged": sorted(unchanged),
                }
        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY DELTA] get_changed_sections failed: {e}")
            return None

    def compute_growth_delta(self, user_id: str = "primary") -> Optional[Dict[str, Any]]:
        """
        For each changed section, produce a one-sentence delta summary via LLM.

        Uses the same lightweight model as autobiography synthesis.

        Args:
            user_id: User identifier

        Returns:
            Dict with from_version, to_version, section_deltas: {section: summary}
            or None on failure.
        """
        try:
            from sqlalchemy import text

            with self.db.get_session() as session:
                # Get latest two versions with narrative
                result = session.execute(
                    text("""
                    SELECT version, narrative, section_hashes
                    FROM autobiography
                    WHERE user_id = :user_id
                    ORDER BY version DESC
                    LIMIT 2
                    """),
                    {"user_id": user_id}
                )
                rows = result.fetchall()

                if len(rows) < 2:
                    return None

                new_version, new_narrative, new_hashes = rows[0]
                old_version, old_narrative, old_hashes = rows[1]

                if not new_narrative or not old_narrative:
                    return None

                new_sections = _extract_sections(new_narrative)
                old_sections = _extract_sections(old_narrative)
                new_hashes = new_hashes or {}
                old_hashes = old_hashes or {}

                # Find changed sections
                changed = [
                    s for s in set(list(new_hashes.keys()) + list(old_hashes.keys()))
                    if new_hashes.get(s) != old_hashes.get(s)
                ]

                if not changed:
                    return {
                        "from_version": old_version,
                        "to_version": new_version,
                        "section_deltas": {},
                    }

                # Generate per-section LLM summaries
                section_deltas = {}
                for section in changed:
                    old_text = old_sections.get(section, "(not present)")
                    new_text = new_sections.get(section, "(not present)")

                    delta = self._summarize_section_delta(section, old_text, new_text)
                    if delta:
                        section_deltas[section] = delta

                return {
                    "from_version": old_version,
                    "to_version": new_version,
                    "section_deltas": section_deltas,
                }
        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY DELTA] compute_growth_delta failed: {e}", exc_info=True)
            return None

    def reinforce_stable_traits(self, user_id: str = "primary") -> int:
        """
        Reinforce traits mentioned in stable (unchanged) sections.

        When Identity or Long Term Themes remain unchanged across 3+ consecutive
        versions, traits mentioned in those sections receive a reinforcement signal.
        Uses asymptotic stabilization: confidence += (1 - confidence) * 0.05

        Args:
            user_id: User identifier

        Returns:
            Number of traits reinforced
        """
        try:
            from sqlalchemy import text

            with self.db.get_session() as session:
                # Count consecutive stable versions
                result = session.execute(
                    text("""
                    SELECT version, section_hashes
                    FROM autobiography
                    WHERE user_id = :user_id AND section_hashes IS NOT NULL
                    ORDER BY version DESC
                    LIMIT 5
                    """),
                    {"user_id": user_id}
                )
                rows = result.fetchall()

                if len(rows) < 3:
                    return 0

                # Identify sections stable across last 3 versions
                stable_sections = set()
                target_sections = {'identity', 'long_term_themes'}

                # Compare consecutive pairs
                consecutive_stable = {}
                for i in range(len(rows) - 1):
                    newer_hashes = rows[i][1] or {}
                    older_hashes = rows[i + 1][1] or {}

                    for section in target_sections:
                        if (newer_hashes.get(section) and
                                newer_hashes.get(section) == older_hashes.get(section)):
                            consecutive_stable[section] = consecutive_stable.get(section, 0) + 1

                for section, count in consecutive_stable.items():
                    if count >= 2:  # Stable across 3 versions (2 pairs)
                        stable_sections.add(section)

                if not stable_sections:
                    return 0

                # Get narrative text for stable sections
                latest_narrative = rows[0]  # version, section_hashes (need narrative)
                result = session.execute(
                    text("""
                    SELECT narrative FROM autobiography
                    WHERE user_id = :user_id
                    ORDER BY version DESC
                    LIMIT 1
                    """),
                    {"user_id": user_id}
                )
                narrative_row = result.fetchone()
                if not narrative_row or not narrative_row[0]:
                    return 0

                sections_text = _extract_sections(narrative_row[0])

                # Build combined text from stable sections
                stable_text = " ".join(
                    sections_text.get(s, "") for s in stable_sections
                )
                if not stable_text.strip():
                    return 0

            # Reinforce traits whose key/value appears in stable text
            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(self.db)

            reinforced = 0
            try:
                with self.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, trait_key, trait_value, confidence
                        FROM user_traits
                        WHERE user_id = %s
                    """, (user_id,))
                    traits = cursor.fetchall()

                    for trait_id, trait_key, trait_value, confidence in traits:
                        # Check if this trait is mentioned in stable sections
                        key_mentioned = trait_key.replace('_', ' ') in stable_text.lower()
                        val_mentioned = str(trait_value).lower() in stable_text.lower()

                        if key_mentioned or val_mentioned:
                            # Asymptotic stabilization — confidence approaches 1.0 but never overshoots
                            new_confidence = confidence + (1.0 - confidence) * 0.05
                            new_confidence = min(1.0, new_confidence)

                            cursor.execute("""
                                UPDATE user_traits
                                SET confidence = %s, last_reinforced_at = NOW(), updated_at = NOW()
                                WHERE id = %s
                            """, (new_confidence, trait_id))
                            reinforced += 1

                    cursor.close()

            except Exception as e:
                logger.warning(f"[AUTOBIOGRAPHY DELTA] Trait reinforcement failed: {e}")

            if reinforced > 0:
                logger.info(
                    f"[AUTOBIOGRAPHY DELTA] Reinforced {reinforced} traits "
                    f"from stable sections: {stable_sections}"
                )
            return reinforced

        except Exception as e:
            logger.error(f"[AUTOBIOGRAPHY DELTA] reinforce_stable_traits failed: {e}")
            return 0

    def _summarize_section_delta(
        self,
        section: str,
        old_text: str,
        new_text: str,
    ) -> Optional[str]:
        """
        Use LLM to produce a one-sentence delta summary for a changed section.

        Args:
            section: Section name
            old_text: Previous section content
            new_text: Updated section content

        Returns:
            One-sentence delta summary or None
        """
        try:
            from services.llm_service import create_llm_service
            from services.config_service import ConfigService

            config = ConfigService.get_agent_config("autobiography")
            llm = create_llm_service(config)

            system_prompt = (
                "You are a growth analyst. Compare two versions of a biographical section "
                "and describe the change in exactly one sentence. Focus on what shifted, grew, "
                "or deepened. Be specific but concise. Output only the sentence, no prefix."
            )

            user_prompt = (
                f"Section: {section.replace('_', ' ').title()}\n\n"
                f"Previous:\n{old_text[:800]}\n\n"
                f"Updated:\n{new_text[:800]}"
            )

            response = llm.send_message(system_prompt, user_prompt)
            if hasattr(response, 'text'):
                return response.text.strip()
            return str(response).strip() if response else None

        except Exception as e:
            logger.warning(f"[AUTOBIOGRAPHY DELTA] Section delta LLM failed for '{section}': {e}")
            return None
