"""
find_skills — surface curated step-by-step skill playbooks.

Queries skills.sqlite using vec+FTS5 RRF fusion and returns matching skill
content together with any personalisation rules derived from the user's
behavioural patterns (written by SkillAssociationService).
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List

from abilities._base import Ability
from services.embedding_utils import pack_embedding
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[FIND_SKILLS]"

_RRF_K = 15
_KNN_DEPTH = 30


class FindSkillsAbility(Ability):
    """Return procedural skill playbooks matching the user's query.

    Queries skills.sqlite (vec + FTS5 RRF fusion) and annotates each result
    with personalisation rules from skill_associations when available.
    """

    NAME = "find_skills"
    SEARCH_TOOLTIP = "discover procedural skill playbooks for complex tasks"
    SUMMARY = (
        "Find a step-by-step skill playbook for a complex task like research, "
        "planning, or analysis."
    )
    EXAMPLES = [
        "help me structure a competitor analysis report",
        "what's the best framework for planning meals for the week",
        "I want to write a great performance review for my team",
        "walk me through how to plan a fitness routine",
        "give me a structured approach to managing a complex project",
        "help me create a research framework for evaluating new technologies",
    ]
    INPUT_SCHEMA = {
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
    TIMEOUT = 10

    _SKILLS_DB_PATH: ClassVar[Path] = (
        Path(__file__).resolve().parent / "assets" / "skills.sqlite"
    )

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        """Query skills.sqlite and return matching playbooks with personalisation."""
        query = params.get("query", "").strip()
        logger.info(f"{LOG_PREFIX} query='{query}' limit={params.get('limit', 3)}")
        if not query:
            return {"text": _skill_tag("find_skills", error="query-required")}

        if not self._SKILLS_DB_PATH.exists():
            logger.warning(f"{LOG_PREFIX} skills.sqlite not found at {self._SKILLS_DB_PATH}")
            return {"text": _skill_tag("find_skills", error="skill-db-unavailable")}

        limit = min(params.get("limit", 3), 5)

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query)
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} embedding failed: {exc}")
            return _fallback_keyword_search(query, limit, self._SKILLS_DB_PATH)

        blob = pack_embedding(query_embedding)
        rows = _query_skills_db(query, blob, limit, self._SKILLS_DB_PATH)

        if not rows:
            return {
                "text": _skill_tag(
                    "find_skills",
                    f'No skill playbooks found for "{query}".',
                    query=query,
                    found=0,
                )
            }

        content = _format_skills(rows, self._SKILLS_DB_PATH)
        return {"text": _skill_tag("find_skills", content, query=query, found=len(rows))}


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception as exc:
        logger.debug(f"{LOG_PREFIX} sqlite_vec module load failed, trying vec0: {exc}")
        conn.load_extension("vec0")


def _fetch_search_rows(
    conn: sqlite3.Connection, query: str, blob: bytes
) -> tuple[list, list]:
    """Run vec + FTS queries against skills.sqlite. Returns (vec_rows, fts_rows)."""
    vec_rows = conn.execute(
        """
        SELECT s.id, s.title, v.distance
        FROM skill_search_vec v
        JOIN skill_search_entries e ON e.id = v.rowid
        JOIN skills s ON s.id = e.skill_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance ASC
        """,
        (blob, _KNN_DEPTH),
    ).fetchall()

    try:
        fts_rows = conn.execute(
            """
            SELECT s.id, s.title, bm25(skill_search_fts) AS score
            FROM skill_search_fts
            JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
            JOIN skills s ON s.id = e.skill_id
            WHERE skill_search_fts MATCH ?
            ORDER BY score ASC
            """,
            (query,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning(f"{LOG_PREFIX} FTS5 query failed: {exc}")
        fts_rows = []

    return vec_rows, fts_rows


def _rrf_merge(vec_rows: list, fts_rows: list, k: int) -> List[Dict]:
    """Merge vector + FTS results via reciprocal rank fusion, capped at k."""
    title_by_id: Dict[int, str] = {}
    vec_best: Dict[int, float] = {}
    for sid, title, distance in vec_rows:
        title_by_id[sid] = title
        if sid not in vec_best or distance < vec_best[sid]:
            vec_best[sid] = distance

    fts_best: Dict[int, float] = {}
    for sid, title, score in fts_rows:
        title_by_id.setdefault(sid, title)
        if sid not in fts_best or score < fts_best[sid]:
            fts_best[sid] = score

    rrf_scores: Dict[int, float] = defaultdict(float)
    for rank, (sid, _) in enumerate(sorted(vec_best.items(), key=lambda x: x[1]), start=1):
        rrf_scores[sid] += 1.0 / (_RRF_K + rank)
    for rank, (sid, _) in enumerate(sorted(fts_best.items(), key=lambda x: x[1]), start=1):
        rrf_scores[sid] += 1.0 / (_RRF_K + rank)

    if not rrf_scores:
        return []

    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [
        {"skill_id": sid, "title": title_by_id.get(sid, ""), "score": score}
        for sid, score in merged[:k]
    ]


def _query_skills_db(query: str, blob: bytes, k: int, db_path: Path) -> List[Dict]:
    """Run hybrid search against skills.sqlite. Returns ranked skill dicts."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            _load_vec(conn)
            vec_rows, fts_rows = _fetch_search_rows(conn, query, blob)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} skills.sqlite query failed: {exc}")
        return []

    return _rrf_merge(vec_rows, fts_rows, k)


def _fetch_skill_detail(db_path: Path, skill_id: int) -> tuple[str, list[dict]]:
    """Fetch full content + personalisation rules for one skill."""
    conn = sqlite3.connect(str(db_path))
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


def _format_skills(rows: List[Dict], db_path: Path) -> str:
    """Format matched skills into a human-readable playbook block."""
    parts: list[str] = []

    for row in rows:
        content, rules = _fetch_skill_detail(db_path, row["skill_id"])
        parts.append(f"## {row['title']}\n\n{content}")
        if rules:
            rule_lines = "\n".join(
                f"- {r['rule']} (pattern: {r['pattern_name']})" for r in rules
            )
            parts.append(f"\n### Personalised for you\n{rule_lines}")

    return "\n\n---\n\n".join(parts)


def _fallback_keyword_search(query: str, limit: int, db_path: Path) -> dict:
    """FTS-only search fallback when embedding generation fails."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                """
                SELECT s.id, s.title
                FROM skill_search_fts
                JOIN skill_search_entries e ON e.id = skill_search_fts.rowid
                JOIN skills s ON s.id = e.skill_id
                WHERE skill_search_fts MATCH ?
                GROUP BY s.id
                ORDER BY s.title
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "text": _skill_tag(
                    "find_skills",
                    f'No skill playbooks found for "{query}".',
                    query=query,
                    found=0,
                )
            }

        matched = [{"skill_id": r[0], "title": r[1], "score": 0.5} for r in rows]
        content = _format_skills(matched, db_path)
        return {"text": _skill_tag("find_skills", content, query=query, found=len(matched))}

    except Exception as exc:
        logger.warning(f"{LOG_PREFIX} keyword fallback failed: {exc}")
        return {
            "text": _skill_tag(
                "find_skills", error=f"skill-search-unavailable:{str(exc)[:100]}"
            )
        }
