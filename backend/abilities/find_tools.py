import copy
import json
import logging
import sqlite3
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
    SUMMARY = "Use this tool to discover more tools and capabilities. Search for the tools you need."
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

    _DB_PATH: ClassVar[Path] = FileMapperService.get_abilities_db_path()
    # Separate runtime DB for dynamically-synced MCP client tools.
    # Queried in addition to _DB_PATH so build_ability_db rebuilds never
    # destroy _mcp_* rows.  See McpClientService and FileMapperService.
    _MCP_DB_PATH: ClassVar[Path] = FileMapperService.get_mcp_tools_db_path()
    _LOG_PREFIX = "[FIND_TOOLS]"

    def _build_tools_index(self, mp=None) -> str:
        """Return a formatted tools index string for discoverable tools.

        Includes both registered abilities (from AbilityRegistry) and
        enabled+online MCP client tools (from McpClientService).
        """
        from abilities._registry import AbilityRegistry

        proc = mp
        discoverable = list(getattr(proc, "DISCOVERABLE", []) or []) if proc else []
        blocked = set(getattr(proc, "_BLOCKED", set()) or set()) if proc else set()

        index = {}
        for name in discoverable:
            if name in blocked:
                continue
            try:
                ability = AbilityRegistry.get(name)
                index[name] = getattr(ability, "SEARCH_TOOLTIP", "") or ability.SUMMARY
            except KeyError:
                pass

        # MCP tools are listed by their bare server-reported name only
        # (e.g. `list_tickets`), no tooltip.  The MCP protocol's per-tool
        # `description` is matched at search time (FTS over mcp_tools.sqlite),
        # not surfaced in this browse hint.  Gate on the prefixed call name.
        mcp_display = [
            display for call_name, display in self._get_online_mcp_tools_index()
            if call_name not in blocked
        ]

        if not index and not mcp_display:
            return ""
        parts = [f"`{k}` ({v})" for k, v in index.items()]
        parts.extend(f"`{n}`" for n in mcp_display)
        return ", ".join(parts)

    @staticmethod
    def _get_online_mcp_names() -> list[str]:
        """Return names of tools from enabled+online MCP servers.

        Gracefully returns [] when no servers are configured or the service
        is unavailable — never raises so find_tools always completes.
        """
        try:
            from services.mcp_client_service import McpClientService
            return McpClientService().get_online_mcp_tool_names()
        except Exception as exc:
            logger.debug("[FIND_TOOLS] Could not fetch MCP tool names: %s", exc)
            return []

    @staticmethod
    def _get_online_mcp_tools_index() -> list[tuple[str, str]]:
        """Return (call_name, display_name) pairs for enabled+online MCP tools.

        Gracefully returns [] when no servers are configured or the service
        is unavailable — never raises so find_tools always completes.
        """
        try:
            from services.mcp_client_service import McpClientService
            return McpClientService().get_online_mcp_tools_index()
        except Exception as exc:
            logger.debug("[FIND_TOOLS] Could not fetch MCP tool index: %s", exc)
            return []

    def get_input_schema(self, mp=None) -> dict:
        tools_index = self._build_tools_index(mp)
        if not tools_index:
            return self.INPUT_SCHEMA
        schema = copy.deepcopy(self.INPUT_SCHEMA)
        schema["properties"]["query"]["description"] = (
            f"Specify the name of the tool you need. Tools available: {tools_index}"
        )
        return schema

    def run(self, params: dict) -> str:
        query = params.get("query", "").strip()
        logger.info(f"{self._LOG_PREFIX} query='{query}' limit={params.get('limit', 5)}")
        if not query:
            return _skill_tag("find_tools", error="query-required")

        proc = self.MessageProcessor
        allow = list(getattr(proc, "DISCOVERABLE", []) or []) if proc is not None else []
        # Augment the allow-list with enabled+online MCP tool names so the
        # find_tools query gate accepts them.
        mcp_names = self._get_online_mcp_names()
        effective_allow = list(set(allow) | set(mcp_names))

        if not effective_allow:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)

        limit = min(params.get("limit", 5), 10)

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query, mp=self.MessageProcessor)
        except Exception as e:
            logger.warning(f"{self._LOG_PREFIX} Embedding generation failed: {e}")
            return self._fallback(query, limit, effective_allow)

        blob = pack_embedding(query_embedding)
        # Query both abilities.sqlite (registered abilities) and mcp_tools.sqlite.
        rows = self._query(query, blob, limit, allow)
        mcp_rows = self._query_mcp(query, blob, limit, mcp_names)
        rows = self._merge_and_truncate(rows + mcp_rows, limit)

        if not rows:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)

        for row in rows:
            logger.info(f"{self._LOG_PREFIX} RRF: {row['key']} score={row['score']:.4f}")

        raw_text = self._format(rows)
        discovered_names = [t["key"] for t in rows]
        self._append_active(discovered_names)
        return _skill_tag("find_tools", raw_text, query=query, found=len(discovered_names))

    def _append_active(self, names: list[str]) -> None:
        """Append newly-discovered tool NAMES to the live processor's ACTIVE_TOOLS
        so build_tools surfaces them on the next ACT iteration. First-seen wins;
        already-active names are skipped. No-op outside a turn
        (``self.MessageProcessor`` is None — e.g. the action-button path)."""
        if not names:
            return
        proc = self.MessageProcessor
        if proc is None:
            return
        active = proc.active_tools
        for name in names:
            if name not in active:
                active.append(name)

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

    def _query_mcp(self, query: str, blob: bytes, limit: int, mcp_names: list[str]) -> list:
        """FTS-only search against mcp_tools.sqlite for enabled+online tools.

        MCP tools don't have vector embeddings (they are dynamic and not passed
        through the embedding pipeline), so this uses FTS keyword search only.
        Returns rows in the same format as _hybrid_search: {key, label, score}.
        """
        if not mcp_names or not self._MCP_DB_PATH.exists():
            return []
        placeholders = ",".join("?" * len(mcp_names))
        try:
            conn = sqlite3.connect(str(self._MCP_DB_PATH))
            try:
                rows = conn.execute(
                    f"""
                    SELECT mt.tool_name, mt.summary, bm25(mcp_tools_fts) AS score
                    FROM mcp_tools_fts
                    JOIN mcp_tools mt ON mt.id = mcp_tools_fts.rowid
                    WHERE mcp_tools_fts MATCH ?
                      AND mt.tool_name IN ({placeholders})
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (query, *mcp_names, limit),
                ).fetchall()
                # Cap MCP score at 0.12 — ability RRF max is ~0.125 (best of
                # both vector+FTS signals: 2×(1/16)).  This keeps strong MCP
                # matches just below the best ability score so a highly-relevant
                # ability can still edge them out, while weak MCP matches rank
                # low.  bm25() is negative in SQLite; abs() normalizes it.
                return [
                    {"key": r[0], "label": r[1] or "", "score": min(0.12, abs(r[2]))}
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("[FIND_TOOLS] mcp_tools FTS failed: %s", exc)
            return []

    @staticmethod
    def _merge_and_truncate(rows: list[dict], limit: int) -> list[dict]:
        """Deduplicate by key and keep the top `limit` results by score."""
        seen: set[str] = set()
        merged = []
        for row in rows:
            if row["key"] not in seen:
                seen.add(row["key"])
                merged.append(row)
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged[:limit]

    def _format(self, rows: list) -> str:
        entries = [
            {"name": t["key"], "relevance": round(min(1.0, t["score"] * 8.0), 2)}
            for t in rows
        ]
        return json.dumps({"added_tools": entries})

    @staticmethod
    def _no_results_text(query: str) -> str:
        return f'INFO: The best tools for "{query}" are already available.'

    def _fallback(self, query: str, limit: int, allow: list[str]) -> str:
        """FTS-only fallback for abilities.sqlite when embedding fails."""
        if not allow:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)
        # Split allow list into ability names (abilities.sqlite) vs MCP names.
        mcp_names = [n for n in allow if n.startswith("_mcp_")]
        ability_names = [n for n in allow if not n.startswith("_mcp_")]

        discovered: list[str] = []

        if ability_names:
            placeholders = ",".join("?" * len(ability_names))
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
                fts_params=(query, *ability_names, limit),
            )
            discovered.extend(row[0] for row in rows)

        # Also fallback-search the MCP tools DB.
        for row in self._query_mcp(query, b"", limit, mcp_names):
            if row["key"] not in discovered:
                discovered.append(row["key"])

        discovered = discovered[:limit]

        if not discovered:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)

        entries = [{"name": n, "relevance": 0.5} for n in discovered]
        raw_text = json.dumps({"added_tools": entries})
        self._append_active(discovered)
        return _skill_tag("find_tools", raw_text, query=query, found=len(discovered))
