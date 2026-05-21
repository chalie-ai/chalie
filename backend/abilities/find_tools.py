import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, List, Dict

from abilities._base import Ability
from services.embedding_utils import pack_embedding
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[FIND_TOOLS]"

RRF_K = 15
KNN_DEPTH = 30


class FindToolsAbility(Ability):
    NAME = "find_tools"
    SEARCH_TOOLTIP = "discover available tools"
    SUMMARY = "Use this tool to expose more tools and capabilities."
    EXAMPLES = [
        "I want to check if Apple's Q2 earnings report is out yet",
        "can you look up a flight for me",
        "I need to send an email to my team",
        "find me a tool that can track stock prices",
        "search for a capability that lets you read my calendar",
        "is there a way to check the current weather in another city",
        "can you help me monitor a webpage for changes",
        "look up something from my gmail",
        "is the bedroom AC on",
        "turn off all the downstairs lights",
        "list my smart home devices",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Describe what you need.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 10).",
            },
        },
        "required": ["query"],
    }
    TIMEOUT = 10

    _ABILITIES_DB_PATH: ClassVar[Path] = (
        Path(__file__).resolve().parent / "assets" / "abilities.sqlite"
    )

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        query = params.get("query", "").strip()
        logger.info(f"{LOG_PREFIX} query='{query}' limit={params.get('limit', 5)}")
        if not query:
            return {"text": _skill_tag("find_tools", error="query-required"), "_discovered_tools": []}

        # The calling processor's DISCOVERABLE list is the discovery allowlist.
        # No processor → empty allowlist → no results (find_tools is a no-op
        # outside a turn).
        from services.message_processor import current_processor
        proc = current_processor()
        allow = list(getattr(proc, "DISCOVERABLE", []) or []) if proc is not None else []
        if not allow:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        limit = min(params.get("limit", 5), 10)

        try:
            from services.embedding_service import EmbeddingService
            emb_service = EmbeddingService()
            query_embedding = emb_service.generate_embedding(query)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding generation failed: {e}")
            return _fallback_keyword_search(query, limit, self._ABILITIES_DB_PATH, allow)

        blob = pack_embedding(query_embedding)
        rows = _query_abilities_db(query, blob, limit, self._ABILITIES_DB_PATH, allow)

        if not rows:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        for row in rows:
            logger.info(f"{LOG_PREFIX} RRF: {row['tool_name']} score={row['score']:.4f}")

        raw_text = _format_added_tools(rows)
        discovered_names = [t["tool_name"] for t in rows]

        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered_names)),
            "_discovered_tools": discovered_names,
        }


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} sqlite_vec module load failed, trying vec0 extension: {e}")
        conn.load_extension("vec0")


def _fetch_search_rows(conn: sqlite3.Connection, query: str, blob: bytes, allow: List[str]) -> tuple[list, list]:
    """Run vec + FTS queries against abilities.sqlite, returning (vec_rows, fts_rows).

    FTS failures (special characters in the query) are caught and surfaced as an
    empty list so vec results still reach the caller.
    """
    placeholders = ",".join("?" * len(allow))
    vec_rows = conn.execute(
        f"""
        SELECT a.name, a.summary, v.distance
        FROM ability_search_vec v
        JOIN ability_search_entries e ON e.id = v.rowid
        JOIN abilities a ON a.id = e.ability_id
        WHERE v.embedding MATCH ? AND k = ?
          AND a.name IN ({placeholders})
        ORDER BY v.distance ASC
        """,
        (blob, KNN_DEPTH, *allow),
    ).fetchall()
    try:
        fts_rows = conn.execute(
            f"""
            SELECT a.name, a.summary, bm25(ability_search_fts) AS score
            FROM ability_search_fts
            JOIN ability_search_entries e ON e.id = ability_search_fts.rowid
            JOIN abilities a ON a.id = e.ability_id
            WHERE ability_search_fts MATCH ?
              AND a.name IN ({placeholders})
            ORDER BY score ASC
            """,
            (query, *allow),
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning(f"{LOG_PREFIX} FTS5 query failed (special chars in query?): {e}")
        fts_rows = []
    return vec_rows, fts_rows


def _rrf_merge(vec_rows: list, fts_rows: list, k: int) -> List[Dict]:
    """Merge vector + FTS results via reciprocal rank fusion, capped at ``k`` rows."""
    summary_by_name: Dict[str, str] = {}
    vec_best: Dict[str, float] = {}
    for name, summary, distance in vec_rows:
        summary_by_name[name] = summary
        if name not in vec_best or distance < vec_best[name]:
            vec_best[name] = distance
    fts_best: Dict[str, float] = {}
    for name, summary, score in fts_rows:
        summary_by_name.setdefault(name, summary)
        if name not in fts_best or score < fts_best[name]:
            fts_best[name] = score

    rrf_scores: Dict[str, float] = defaultdict(float)
    for rank, (name, _) in enumerate(sorted(vec_best.items(), key=lambda x: x[1]), start=1):
        rrf_scores[name] += 1.0 / (RRF_K + rank)
    for rank, (name, _) in enumerate(sorted(fts_best.items(), key=lambda x: x[1]), start=1):
        rrf_scores[name] += 1.0 / (RRF_K + rank)

    if not rrf_scores:
        return []
    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [
        {"tool_name": name, "summary": summary_by_name.get(name, ""), "score": rrf_score}
        for name, rrf_score in merged[:k]
    ]


def _query_abilities_db(
    query: str,
    blob: bytes,
    k: int,
    db_path: Path,
    allow: List[str],
) -> List[Dict]:
    if not db_path.exists():
        logger.warning(f"{LOG_PREFIX} abilities.sqlite not found at {db_path}")
        return []
    if not allow:
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            _load_vec(conn)
            vec_rows, fts_rows = _fetch_search_rows(conn, query, blob, allow)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} abilities.sqlite query failed: {e}")
        return []

    return _rrf_merge(vec_rows, fts_rows, k)


def _format_added_tools(tools: List[Dict]) -> str:
    entries = [
        {
            "name": t["tool_name"],
            "relevance": round(min(1.0, t["score"] * 8.0), 2),
        }
        for t in tools
    ]
    return json.dumps({"added_tools": entries})


def _format_no_tools(query: str) -> str:
    return f'INFO: The best tools for "{query}" are already available.'



def _fallback_keyword_search(query: str, limit: int, db_path: Path, allow: List[str]) -> dict:
    if not db_path.exists():
        logger.warning(f"{LOG_PREFIX} abilities.sqlite not found — cannot run keyword fallback")
        return {
            "text": _skill_tag("find_tools", error="tool-search-unavailable:db-missing"),
            "_discovered_tools": [],
        }
    if not allow:
        return {
            "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
            "_discovered_tools": [],
        }
    placeholders = ",".join("?" * len(allow))
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                f"""
                SELECT a.name
                FROM ability_search_fts
                JOIN ability_search_entries e ON e.id = ability_search_fts.rowid
                JOIN abilities a ON a.id = e.ability_id
                WHERE ability_search_fts MATCH ?
                  AND a.name IN ({placeholders})
                GROUP BY a.id
                ORDER BY a.name
                LIMIT ?
                """,
                (query, *allow, limit),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        discovered = [row[0] for row in rows]
        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        raw_text = json.dumps({"added_tools": entries})
        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered)),
            "_discovered_tools": discovered,
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback failed: {e}")
        return {
            "text": _skill_tag("find_tools", error=f"tool-search-unavailable:{str(e)[:100]}"),
            "_discovered_tools": [],
        }
