"""
find_skills — surface curated step-by-step skill playbooks.

Queries skills.sqlite using vec+FTS5 RRF fusion and returns matching skill
content together with any personalisation rules derived from the user's
behavioural patterns (written by SkillAssociationService).
"""

import logging
import sqlite3
from pathlib import Path
from typing import ClassVar

from abilities._search import KNN_DEPTH, SearchableAbility
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class FindSkillsAbility(SearchableAbility):
    """Return procedural skill playbooks matching the user's query."""

    def get_name(self) -> str:
        return "find_skills"

    def get_summary(self) -> str:
        return (
            "Find a step-by-step skill playbook for a complex task like research, "
            "planning, or analysis."
        )

    def get_examples(self) -> list[str]:
        return [
            "help me structure a competitor analysis report",
            "what's the best framework for planning meals for the week",
            "I want to write a great performance review for my team",
            "walk me through how to plan a fitness routine",
            "give me a structured approach to managing a complex project",
            "help me create a research framework for evaluating new technologies",
        ]

    def get_search_tooltip(self) -> str:
        return "discover procedural skill playbooks for complex tasks"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Describe the task or topic you need a playbook for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 3, max 5).",
            },
        },
        "required": ["query"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _DB_PATH: ClassVar[Path] = FileMapperService.get_skills_db_path()
    _LOG_PREFIX = "[FIND_SKILLS]"

    def run(self, params: dict) -> dict:
        query = params.get("query", "").strip()
        logger.info(f"{self._LOG_PREFIX} query='{query}' limit={params.get('limit', 3)}")
        if not query:
            return {"text": _skill_tag("find_skills", error="query-required")}

        if not self._DB_PATH.exists():
            logger.warning(f"{self._LOG_PREFIX} skills.sqlite not found at {self._DB_PATH}")
            return {"text": _skill_tag("find_skills", error="skill-db-unavailable")}

        limit = min(params.get("limit", 3), 5)

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query, mp=self.mp)
        except Exception as exc:
            logger.warning(f"{self._LOG_PREFIX} embedding failed: {exc}")
            return self._fallback(query, limit)

        blob = pack_embedding(query_embedding)
        rows = self._query(query, blob, limit)

        if not rows:
            return {
                "text": _skill_tag(
                    "find_skills",
                    f'No skill playbooks found for "{query}".',
                    query=query,
                    found=0,
                )
            }

        content = self._format(rows)
        return {"text": _skill_tag("find_skills", content, query=query, found=len(rows), content_type="playbook")}

    def _query(self, query: str, blob: bytes, limit: int) -> list:
        return self._hybrid_search(
            query, blob, limit,
            vec_sql="""
                SELECT s.id, s.title, v.distance
                FROM skill_search_vec v
                JOIN skill_search_entries e ON e.id = v.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE v.embedding MATCH ? AND k = ?
                AND s.enabled = 1
                ORDER BY v.distance ASC
            """,
            fts_sql="""
                SELECT s.id, s.title, bm25(skill_search_fts) AS score
                FROM skill_search_fts
                JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE skill_search_fts MATCH ?
                AND s.enabled = 1
                ORDER BY score ASC
            """,
            vec_params=(blob, KNN_DEPTH),
            fts_params=(query,),
        )

    def _fetch_detail(self, skill_id: int) -> tuple[str, list[dict]]:
        conn = sqlite3.connect(str(self._DB_PATH))
        try:
            row = conn.execute(
                "SELECT content FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            content = row[0] if row else ""

            rules = conn.execute(
                "SELECT pattern_name, rule FROM skill_associations WHERE skill_id = ?",
                (skill_id,),
            ).fetchall()
        finally:
            conn.close()

        return content, [{"pattern_name": r[0], "rule": r[1]} for r in rules]

    def _format(self, rows: list) -> str:
        parts: list[str] = []
        for row in rows:
            content, rules = self._fetch_detail(row["key"])
            parts.append(f"## {row['label']}\n\n{content}")
            if rules:
                rule_lines = "\n".join(
                    f"- {r['rule']} (pattern: {r['pattern_name']})" for r in rules
                )
                parts.append(f"\n### Personalised for you\n{rule_lines}")

        return "\n\n---\n\n".join(parts)

    def _fallback(self, query: str, limit: int) -> dict:
        rows = self._fts_only_search(
            fts_sql="""
                SELECT s.id, s.title
                FROM skill_search_fts
                JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE skill_search_fts MATCH ?
                AND s.enabled = 1
                GROUP BY s.id
                ORDER BY s.title
                LIMIT ?
            """,
            fts_params=(query, limit),
        )

        if not rows:
            return {
                "text": _skill_tag(
                    "find_skills",
                    f'No skill playbooks found for "{query}".',
                    query=query,
                    found=0,
                )
            }

        matched = [{"key": r[0], "label": r[1], "score": 0.5} for r in rows]
        content = self._format(matched)
        return {"text": _skill_tag("find_skills", content, query=query, found=len(matched), content_type="playbook")}
