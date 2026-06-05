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
    """Discover and activate tools for the current ACT turn.

    Supports two mutually exclusive selection modes:
    - select: exact case-insensitive match against the effective allow-list.
    - query: hybrid vec+FTS RRF semantic search with a relevance floor.
    When both are supplied, select takes precedence and query is ignored.
    """

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
            "select": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact tool names to activate directly.",
            },
            "query": {
                "type": "string",
                "description": "Describe what you need.",
            },
        },
        "required": [],
    }

    _DB_PATH: ClassVar[Path] = FileMapperService.get_abilities_db_path()
    # Separate runtime DB for dynamically-synced MCP client tools.
    # Queried in addition to _DB_PATH so build_ability_db rebuilds never
    # destroy _mcp_* rows.  See McpClientService and FileMapperService.
    _MCP_DB_PATH: ClassVar[Path] = FileMapperService.get_mcp_tools_db_path()
    _LOG_PREFIX: ClassVar[str] = "[FIND_TOOLS]"

    # Relevance floor: RRF scores below this are single-signal junk.
    # Dual-signal rank-1 in both vec+FTS = 2×(1/(15+1)) = 0.125.
    # Single-signal rank-1 = 1/(15+1) = 0.0625.
    # 0.075 cleanly separates the two populations (empirically verified).
    MIN_RRF_SCORE: ClassVar[float] = 0.075

    # Maximum number of results returned by the query path after floor filtering.
    MAX_QUERY_RESULTS: ClassVar[int] = 5

    @staticmethod
    def _config_scope(proc) -> "tuple[list[str], set[str]]":
        """Return ``(discoverable, blocked)`` for the invoking processor's channel.

        Both are sourced from the per-turn ``ProcessorConfig`` (``config.discoverable``
        / ``config.blocked``) — the single source of truth introduced by the
        typed-subclass refactor (TKT-803/804). This is what keeps raw web tools
        (``browser`` / ``search``) exclusive to the delegate channels and pattern
        writes exclusive to the pattern channels: every channel declares its own
        allow-list and block-list. Returns ``([], set())`` when no config is bound
        (the action-button path / pre-setup), so discovery degrades to empty
        rather than leaking a global list.
        """
        config = getattr(proc, "config", None) if proc is not None else None
        if config is None:
            return [], set()
        discoverable = list(getattr(config, "discoverable", None) or [])
        blocked = set(getattr(config, "blocked", None) or set())
        return discoverable, blocked

    def _build_tools_index(self, mp=None) -> str:
        """Return a formatted tools index string for discoverable tools.

        Includes both registered abilities (from AbilityRegistry) and
        enabled+online MCP client tools (from McpClientService).
        """
        from abilities._registry import AbilityRegistry

        discoverable, blocked = self._config_scope(mp)

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
        # Start from the base schema so the framework `async` property injection
        # (gated on mp.config.SUPPORTS_ASYNC) applies uniformly here too.
        schema = super().get_input_schema(mp)
        tools_index = self._build_tools_index(mp)
        if not tools_index:
            return schema
        schema = copy.deepcopy(schema)
        schema["properties"]["select"]["description"] = (
            f"Exact tool names to activate directly. Available tools: {tools_index}"
        )
        return schema

    def run(self, params: dict) -> str:
        """Dispatch to the select or query path and return a tagged result string."""
        select_names = params.get("select")
        query = params.get("query", "").strip()

        # Require at least one param.
        if not select_names and not query:
            return _skill_tag("find_tools", error="params-required")

        proc = self.MessageProcessor
        discoverable, blocked = self._config_scope(proc)
        # Drop blocked names from BOTH the ability allow-list and the MCP names so
        # the block holds on every path — select, query, and fallback. (The
        # descriptive index in _build_tools_index applies the same filter.)
        allow = [name for name in discoverable if name not in blocked]
        mcp_names = [name for name in self._get_online_mcp_names() if name not in blocked]
        effective_allow = set(allow) | set(mcp_names)

        if not effective_allow:
            return _skill_tag("find_tools", self._no_results_text(query or ""), query=query or None)

        # select wins over query when both are provided.
        if select_names:
            return self._run_select(select_names, effective_allow)

        logger.info("%s query='%s'", self._LOG_PREFIX, query)
        return self._run_query(query, list(allow), list(mcp_names))

    def _run_select(self, requested: list[str], effective_allow: set[str]) -> str:
        """Exact case-insensitive match against effective_allow; append matched names."""
        allow_lower = {name.lower(): name for name in effective_allow}
        matched: list[str] = []
        not_found: list[str] = []

        for name in requested:
            canonical = allow_lower.get(name.lower())
            if canonical is not None:
                matched.append(canonical)
            else:
                not_found.append(name)

        self._append_active(matched)
        parts: list[str] = []
        if matched:
            parts.append(self._format_universal(matched, "Selected and added the following tools"))
        if not_found:
            parts.append(f"Tools not found or unavailable: {', '.join(not_found)}")

        return _skill_tag("find_tools", "\n".join(parts), found=len(matched))

    def _run_query(self, query: str, allow: list[str], mcp_names: list[str]) -> str:
        """Hybrid vec+FTS RRF search with relevance floor and top-N cap."""
        effective_allow = list(set(allow) | set(mcp_names))

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query, mp=self.MessageProcessor)
        except Exception as exc:
            logger.warning("%s Embedding generation failed: %s", self._LOG_PREFIX, exc)
            return self._fallback(query, effective_allow)

        blob = pack_embedding(query_embedding)
        rows = self._query(query, blob, allow)
        mcp_rows = self._query_mcp(query, blob, self.MAX_QUERY_RESULTS, mcp_names)
        rows = self._merge_and_truncate(rows + mcp_rows)

        # Apply relevance floor then cap to MAX_QUERY_RESULTS.
        rows = [r for r in rows if r["score"] >= self.MIN_RRF_SCORE][:self.MAX_QUERY_RESULTS]

        for row in rows:
            logger.info("%s RRF: %s score=%.4f", self._LOG_PREFIX, row["key"], row["score"])

        if not rows:
            return _skill_tag("find_tools", "No tools match the query specified", query=query)

        names = [r["key"] for r in rows]
        self._append_active(names)
        raw_text = self._format_universal(names, f'Query "{query}" matched and added the following tools')
        return _skill_tag("find_tools", raw_text, query=query, found=len(names))

    def _append_active(self, names: list[str]) -> None:
        """Append newly-discovered tool names to the live processor's active_tools.

        Build_tools resolves active_tools to schemas on the next ACT iteration.
        No-op when MessageProcessor is None (action-button path) or names is empty.
        """
        if not names:
            return
        proc = self.MessageProcessor
        if proc is None:
            return
        active = proc.active_tools
        for name in names:
            if name not in active:
                active.append(name)

    def _format_universal(self, names: list[str], lead: str) -> str:
        """Build the universal v2 result string carrying per-tool input_schema.

        Each entry is {"name": <name>, "input_schema": <schema dict>} so the
        literal token "input_schema" always appears in the serialised output.
        Schema lookup failures are logged and skipped — find_tools must always
        complete.
        """
        entries = []
        for name in names:
            schema = self._resolve_schema(name)
            entries.append({"name": name, "input_schema": schema})
        return f"{lead}:\n{json.dumps(entries)}"

    def _resolve_schema(self, name: str) -> dict:
        """Return the input_schema dict for a tool name (ability or MCP).

        Never raises — returns an empty dict on any failure so the caller can
        always produce a result string.
        """
        try:
            if name.startswith("_mcp_"):
                from services.mcp_client_service import McpClientService
                result = McpClientService().get_tool_schema(name)
                if result and "input_schema" in result:
                    return result["input_schema"]
                return {}
            from abilities._registry import AbilityRegistry
            ability = AbilityRegistry.get(name)
            return ability.get_input_schema(self.MessageProcessor)
        except Exception as exc:
            logger.warning("%s Schema lookup failed for %r: %s", self._LOG_PREFIX, name, exc)
            return {}

    def _query(self, query: str, blob: bytes, allow: list[str]) -> list:
        if not allow:
            return []
        placeholders = ",".join("?" * len(allow))
        return self._hybrid_search(
            query, blob, self.MAX_QUERY_RESULTS,
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
    def _merge_and_truncate(rows: list[dict]) -> list[dict]:
        """Deduplicate by key and sort by score descending (no cap — caller applies floor+cap)."""
        seen: set[str] = set()
        merged = []
        for row in rows:
            if row["key"] not in seen:
                seen.add(row["key"])
                merged.append(row)
        merged.sort(key=lambda r: r["score"], reverse=True)
        return merged

    @staticmethod
    def _no_results_text(query: str) -> str:
        return f'INFO: The best tools for "{query}" are already available.'

    def _fallback(self, query: str, allow: list[str]) -> str:
        """FTS-only fallback for abilities.sqlite when embedding fails.

        No relevance floor applied (single-signal by nature); adapts to the
        v2 universal result format.
        """
        if not allow:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)

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
                fts_params=(query, *ability_names, self.MAX_QUERY_RESULTS),
            )
            discovered.extend(row[0] for row in rows)

        for row in self._query_mcp(query, b"", self.MAX_QUERY_RESULTS, mcp_names):
            if row["key"] not in discovered:
                discovered.append(row["key"])

        discovered = discovered[:self.MAX_QUERY_RESULTS]

        if not discovered:
            return _skill_tag("find_tools", self._no_results_text(query), query=query)

        self._append_active(discovered)
        raw_text = self._format_universal(discovered, f'Query "{query}" matched and added the following tools')
        return _skill_tag("find_tools", raw_text, query=query, found=len(discovered))
