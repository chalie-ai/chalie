import json
import logging
from pathlib import Path
from typing import ClassVar

from abilities._search import KNN_DEPTH, SearchableAbility
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class FindToolsAbility(SearchableAbility):
    NAME = "find_tools"
    SEARCH_TOOLTIP = "discover available tools"
    POLICY_CATEGORY = "Search & Tools"
    POLICY_LABELS = {"": "Find tools"}
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

    _DB_PATH: ClassVar[Path] = FileMapperService.get_abilities_db_path()
    _LOG_PREFIX = "[FIND_TOOLS]"

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        query = params.get("query", "").strip()
        logger.info(f"{self._LOG_PREFIX} query='{query}' limit={params.get('limit', 5)}")
        if not query:
            return {"text": _skill_tag("find_tools", error="query-required"), "_discovered_tools": []}

        from services.message_processor import current_processor
        proc = current_processor()
        allow = list(getattr(proc, "DISCOVERABLE", []) or []) if proc is not None else []
        if not allow:
            return {
                "text": _skill_tag("find_tools", self._no_results_text(query), query=query),
                "_discovered_tools": [],
            }

        limit = min(params.get("limit", 5), 10)

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query)
        except Exception as e:
            logger.warning(f"{self._LOG_PREFIX} Embedding generation failed: {e}")
            return self._fallback(query, limit, allow)

        blob = pack_embedding(query_embedding)
        rows = self._query(query, blob, limit, allow)

        if not rows:
            return {
                "text": _skill_tag("find_tools", self._no_results_text(query), query=query),
                "_discovered_tools": [],
            }

        for row in rows:
            logger.info(f"{self._LOG_PREFIX} RRF: {row['key']} score={row['score']:.4f}")

        raw_text = self._format(rows)
        discovered_names = [t["key"] for t in rows]

        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered_names)),
            "_discovered_tools": discovered_names,
        }

    def _query(self, query: str, blob: bytes, limit: int, allow: list[str]) -> list:
        if not allow:
            return []
        placeholders = ",".join("?" * len(allow))
        return self._hybrid_search(
            query, blob, limit,
            vec_sql=f"""
                SELECT a.name, a.summary, v.distance
                FROM ability_search_vec v
                JOIN ability_search_entries e ON e.id = v.rowid
                JOIN abilities a ON a.id = e.ability_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND a.name IN ({placeholders})
                ORDER BY v.distance ASC
            """,
            fts_sql=f"""
                SELECT a.name, a.summary, bm25(ability_search_fts) AS score
                FROM ability_search_fts
                JOIN ability_search_entries e ON e.id = ability_search_fts.rowid
                JOIN abilities a ON a.id = e.ability_id
                WHERE ability_search_fts MATCH ?
                  AND a.name IN ({placeholders})
                ORDER BY score ASC
            """,
            vec_params=(blob, KNN_DEPTH, *allow),
            fts_params=(query, *allow),
        )

    def _format(self, rows: list) -> str:
        entries = [
            {"name": t["key"], "relevance": round(min(1.0, t["score"] * 8.0), 2)}
            for t in rows
        ]
        return json.dumps({"added_tools": entries})

    @staticmethod
    def _no_results_text(query: str) -> str:
        return f'INFO: The best tools for "{query}" are already available.'

    def _fallback(self, query: str, limit: int, allow: list[str]) -> dict:
        if not allow:
            return {
                "text": _skill_tag("find_tools", self._no_results_text(query), query=query),
                "_discovered_tools": [],
            }
        placeholders = ",".join("?" * len(allow))
        rows = self._fts_only_search(
            fts_sql=f"""
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
            fts_params=(query, *allow, limit),
        )

        if not rows:
            return {
                "text": _skill_tag("find_tools", self._no_results_text(query), query=query),
                "_discovered_tools": [],
            }

        discovered = [row[0] for row in rows]
        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        raw_text = json.dumps({"added_tools": entries})
        return {
            "text": _skill_tag("find_tools", raw_text, query=query, found=len(discovered)),
            "_discovered_tools": discovered,
        }
