"""
FindToolsAbility — On-demand tool discovery with dynamic injection.

Cognitive primitive: always available in ACT mode. Lets Chalie search for
external capabilities (tools and interface actions) by semantic query.

After a successful search, the ACT orchestrator injects matching tool
schemas into the native tools list so the LLM can call them directly
in subsequent iterations — no second discovery step needed.

Phase 1: queries abilities.sqlite (new DB) with priority merge — new DB
wins when an entry exists. Also queries tool_capability_profiles_vec.
"""

import logging
import sqlite3
from pathlib import Path
from typing import ClassVar, List, Dict

from abilities._base import Ability
from services.embedding_utils import pack_embedding
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[FIND_TOOLS]"


class FindToolsAbility(Ability):
    NAME = "find_tools"
    SUMMARY = "Discover external tools by semantic search — use when built-in skills cannot handle the task."
    EXAMPLES = [
        "I want to check if Apple's Q2 earnings report is out yet",
        "can you look up a flight for me",
        "I need to send an email to my team",
        "find me a tool that can track stock prices",
        "search for a capability that lets you read my calendar",
        "is there a way to check the current weather in another city",
        "can you help me monitor a webpage for changes",
        "look up something from my gmail",
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
    ALWAYS_AVAILABLE = True
    TIMEOUT = 10

    # abilities.sqlite lives at abilities/assets/abilities.sqlite
    # (one level up from abilities/ for the db, but assets/ is a sibling of this file)
    _ABILITIES_DB_PATH: ClassVar[Path] = (
        Path(__file__).resolve().parent / "assets" / "abilities.sqlite"
    )
    _MIN_RELEVANCE: ClassVar[float] = 0.15

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        query = params.get("query", "").strip()
        logger.info(f"{LOG_PREFIX} query='{query}' limit={params.get('limit', 5)}")
        if not query:
            return {"text": _skill_tag("find_tools", error="query-required"), "_discovered_tools": []}

        limit = min(params.get("limit", 5), 10)

        try:
            from services.embedding_service import EmbeddingService
            emb_service = EmbeddingService()
            query_embedding = emb_service.generate_embedding(query)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding generation failed: {e}")
            return _fallback_keyword_search(query, limit)

        blob = pack_embedding(query_embedding)

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT tcp.tool_name, tcp.tool_type, tcp.short_summary,
                           tcp.full_profile, tcp.domain, tcp.effort,
                           v.distance, tcp.keywords
                    FROM tool_capability_profiles_vec v
                    JOIN tool_capability_profiles tcp ON tcp.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (blob, limit + 5)
                )
                rows = cursor.fetchall()
                cursor.close()

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Vector search failed: {e}")
            return _fallback_keyword_search(query, limit)

        # Phase 1 dual-read: query abilities.sqlite for nearest neighbors and
        # merge with priority — new DB wins when an entry exists.
        new_db_rows = _query_abilities_db(blob, limit + 5, self._ABILITIES_DB_PATH)

        if new_db_rows:
            new_db_names = {r["tool_name"] for r in new_db_rows}
            old_rows_filtered = [r for r in rows if r[0] not in new_db_names]
            rows = new_db_rows + old_rows_filtered

        if not rows:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        for row in rows:
            name = row[0] if not isinstance(row, dict) else row["tool_name"]
            dist = row[6] if not isinstance(row, dict) else row["distance"]
            logger.info(f"{LOG_PREFIX} k-NN: {name} distance={dist:.4f}")

        available_tools = _filter_available(rows, query)

        if not available_tools:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        available_tools = available_tools[:limit]

        raw_text = _format_added_tools(available_tools, self._MIN_RELEVANCE)

        import json
        added = json.loads(raw_text).get("added_tools", [])
        discovered_names = [t["name"] for t in added]

        if not discovered_names:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered_names)),
            "_discovered_tools": discovered_names,
        }


def _query_abilities_db(blob: bytes, k: int, db_path: Path) -> List[Dict]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                conn.load_extension("vec0")

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.name, a.summary, v.distance
                FROM ability_search_vec v
                JOIN ability_search_entries ase ON ase.id = v.rowid
                JOIN abilities a ON a.id = ase.ability_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, k),
            )
            db_rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} abilities.sqlite query failed: {e}")
        return []

    if not db_rows:
        return []

    seen: set = set()
    results: List[Dict] = []
    for name, summary, distance in db_rows:
        if name in seen:
            continue
        seen.add(name)
        score = distance * 10
        results.append({
            "tool_name": name,
            "short_summary": summary,
            "full_profile": summary,
            "domain": "",
            "effort": "low",
            "score": score,
            "distance": distance,
        })

    return results


def _filter_available(rows: list, query: str = "") -> List[Dict]:
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
    except Exception:
        registry = None

    query_lower = query.lower()
    results = []
    for row in rows:
        if isinstance(row, dict) and "score" in row:
            results.append(row)
            continue

        tool_name = row[0] if not isinstance(row, dict) else row["tool_name"]
        tool_type = row[1] if not isinstance(row, dict) else row["tool_type"]
        short_summary = row[2] if not isinstance(row, dict) else row["short_summary"]
        full_profile = row[3] if not isinstance(row, dict) else row["full_profile"]
        domain = row[4] if not isinstance(row, dict) else row["domain"]
        effort = row[5] if not isinstance(row, dict) else row["effort"]
        distance = row[6] if not isinstance(row, dict) else row["distance"]
        keywords = row[7] if not isinstance(row, dict) else row.get("keywords", "")

        if tool_type == "skill":
            continue

        if registry:
            tool_data = registry.tools.get(tool_name)
            if not tool_data:
                continue
            if not registry._is_ready(tool_name, tool_data):
                continue
            if not registry._is_interface_online(tool_data):
                continue

        kw_list = [k.strip().lower() for k in (keywords or "").split(",") if k.strip()]
        query_words = set(query_lower.split())
        kw_match_count = sum(
            1 for kw in kw_list
            if (" " not in kw and kw in query_words) or (" " in kw and kw in query_lower)
        )
        score = (distance * 10) - kw_match_count

        results.append({
            "tool_name": tool_name,
            "short_summary": short_summary,
            "full_profile": full_profile,
            "domain": domain,
            "effort": effort,
            "score": score,
            "distance": distance,
        })

    results.sort(key=lambda t: t["score"])
    return results


def _format_added_tools(tools: List[Dict], min_relevance: float) -> str:
    import json
    entries = []
    for t in tools:
        dist = t.get("distance", 2.0)
        relevance = round(max(0.0, min(1.0, 1.0 - dist / 2.0)), 2)
        if relevance < min_relevance:
            continue
        entries.append({"name": t["tool_name"], "relevance": relevance})
    return json.dumps({"added_tools": entries})


def _format_no_tools(query: str) -> str:
    return f'INFO: The best tools for "{query}" are already available.'


def _fallback_keyword_search(query: str, limit: int) -> dict:
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        query_pattern = f"%{query}%"
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tool_name, tool_type, short_summary, domain, effort
                FROM tool_capability_profiles
                WHERE tool_type = 'tool'
                  AND (short_summary LIKE ? OR full_profile LIKE ? OR tool_name LIKE ?)
                ORDER BY tool_name
                LIMIT ?
                """,
                (query_pattern, query_pattern, query_pattern, limit)
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return {
                "text": _skill_tag("find_tools", _format_no_tools(query), query=query),
                "_discovered_tools": [],
            }

        discovered = []
        for row in rows:
            name = row[0] if not isinstance(row, dict) else row["tool_name"]
            discovered.append(name)

        import json
        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        raw_text = json.dumps({"added_tools": entries})
        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered)),
            "_discovered_tools": discovered,
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Keyword fallback also failed: {e}")
        return {
            "text": _skill_tag("find_tools", error=f"tool-search-unavailable:{str(e)[:100]}"),
            "_discovered_tools": [],
        }
