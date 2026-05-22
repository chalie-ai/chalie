"""
SkillAssociationService — Layer 2 of the Self-Refining Skill Library.

Maps the user's active behavioural patterns to curated skill playbooks and
writes personalisation rules into skill_associations in skills.sqlite.

Called by SubconsciousWorker after every PatternMatchProcessor pass.
"""

import json
import logging
import sqlite3
from pathlib import Path

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SKILL_ASSOC]"

_SKILLS_DB = Path(__file__).resolve().parent.parent / "abilities" / "assets" / "skills.sqlite"

_SYSTEM_PROMPT = """You map behavioral patterns to skill playbooks.

Given a list of the user's behavioral patterns and a list of available skills,
identify which patterns are relevant to which skills and produce a personalisation
rule for each match.

A personalisation rule is a single sentence describing how the skill should be
adapted based on the pattern. Only produce rules where the pattern genuinely
informs how the skill should be executed differently.

Respond with a JSON array of objects:
[{"skill_id": <int>, "pattern_name": "<str>", "rule": "<str>"}]

If no patterns match any skills, respond with an empty array: []"""


class SkillAssociationService:
    """Run LLM-driven association passes between behavioural patterns and skills.

    Reads active patterns from data_graph and available skills from skills.sqlite,
    makes one LLM call to determine personalisation rules, then writes the results
    back to skill_associations in skills.sqlite.
    """

    def run_pass(self) -> int:
        """Run a single association pass. Returns the number of rules written."""
        if not _SKILLS_DB.exists():
            logger.info(f"{LOG_PREFIX} skills.sqlite not found — skipping")
            return 0

        patterns = self._load_active_patterns()
        if not patterns:
            logger.info(f"{LOG_PREFIX} no active patterns — skipping")
            return 0

        skills = self._load_skill_index()
        if not skills:
            logger.info(f"{LOG_PREFIX} no skills in DB — skipping")
            return 0

        associations = self._request_associations(patterns, skills)
        if associations is None:
            return 0

        valid_skill_ids = {s[0] for s in skills}
        active_pattern_names = {p[0] for p in patterns}
        written = self._write_associations(associations, valid_skill_ids, active_pattern_names)

        logger.info(f"{LOG_PREFIX} wrote {written} associations")
        return written

    def _load_active_patterns(self) -> list[tuple[str, str]]:
        """Read all active behavioral_pattern rows from data_graph."""
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            return conn.execute(
                "SELECT key, value FROM data_graph "
                "WHERE kind='behavioral_pattern' AND active=1 AND deleted_at IS NULL"
            ).fetchall()

    def _load_skill_index(self) -> list[tuple]:
        """Read skill index (id, title, use_for, tags) from skills.sqlite."""
        conn = sqlite3.connect(str(_SKILLS_DB))
        try:
            return conn.execute(
                "SELECT id, title, use_for, tags FROM skills"
            ).fetchall()
        finally:
            conn.close()

    def _request_associations(
        self,
        patterns: list[tuple[str, str]],
        skills: list[tuple],
    ) -> list[dict] | None:
        """Call LLM to map patterns to skills. Returns parsed associations or None."""
        pattern_text = "\n".join(
            f"- {key}: {_safe_parse_summary(value)}"
            for key, value in patterns
        )
        skill_text = "\n".join(
            f"- id={sid}, title={title}, use_for={use_for}, tags={tags}"
            for sid, title, use_for, tags in skills
        )
        user_prompt = (
            f"## Behavioral Patterns\n{pattern_text}\n\n"
            f"## Available Skills\n{skill_text}"
        )

        from services.providers import Providers
        response = Providers.instance().send(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            job='subconscious',
            tools=[],
        )

        return _parse_associations(response.text)

    def _write_associations(
        self,
        associations: list[dict],
        valid_skill_ids: set[int],
        active_pattern_names: set[str],
    ) -> int:
        """Write valid associations and prune stale rows. Returns count written."""
        now = utc_now().isoformat()
        written = 0

        conn = sqlite3.connect(str(_SKILLS_DB))
        try:
            for assoc in associations:
                sid = assoc.get("skill_id")
                pname = assoc.get("pattern_name")
                rule = assoc.get("rule")
                if not (sid and pname and rule):
                    continue
                if sid not in valid_skill_ids:
                    continue
                if pname not in active_pattern_names:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO skill_associations "
                    "(skill_id, pattern_name, rule, created_at) VALUES (?, ?, ?, ?)",
                    (sid, pname, rule, now),
                )
                written += 1

            # Remove rules for patterns no longer active.
            if active_pattern_names:
                placeholders = ",".join("?" * len(active_pattern_names))
                conn.execute(
                    f"DELETE FROM skill_associations "
                    f"WHERE pattern_name NOT IN ({placeholders})",
                    tuple(active_pattern_names),
                )

            conn.commit()
        finally:
            conn.close()

        return written


def _parse_associations(text: str) -> list[dict] | None:
    """Parse LLM JSON response into a list of association dicts."""
    if not text:
        return None
    try:
        payload = text.strip()
        if '```' in payload:
            payload = payload.split('```')[1]
            if payload.startswith('json'):
                payload = payload[4:]
        result = json.loads(payload.strip())
    except (json.JSONDecodeError, IndexError) as exc:
        logger.warning(f"{LOG_PREFIX} failed to parse LLM response: {exc}")
        return None

    if not isinstance(result, list):
        logger.warning(f"{LOG_PREFIX} LLM returned non-list: {type(result)}")
        return None

    return result


def _safe_parse_summary(value: str) -> str:
    """Extract summary from pattern JSON value, falling back to raw text."""
    try:
        data = json.loads(value)
        return data.get("summary", str(value)[:200])
    except (json.JSONDecodeError, TypeError):
        return str(value)[:200]
