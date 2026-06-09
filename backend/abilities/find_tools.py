import copy
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
    """Discover and activate tools for the current ACT turn.

    Supports two mutually exclusive selection modes:
    - select: exact case-insensitive match against the effective allow-list.
    - query: hybrid vec+FTS RRF semantic search with a relevance floor.
    When both are supplied, select takes precedence and query is ignored.
    """

    _PARAMETERS: ClassVar[dict] = {
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

    def get_name(self) -> str:
        return "find_tools"

    def get_summary(self) -> str:
        return "Use this tool to discover more tools and capabilities. Search for the tools you need."

    def get_examples(self) -> list[str]:
        return [
            "I want to check if Apple's Q2 earnings report is out yet",
            "can you look up a flight for me",
            "I need to send an email to my team",
            "find me a tool that can track stock prices",
            "search for a capability that lets you read my calendar",
            "is there a way to check the current weather in another city",
            "can you help me monitor a webpage for changes",
            "look up something from my gmail",
        ]

    def get_search_tooltip(self) -> str:
        return "discover available tools"

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
        typed-subclass refactor. This is what keeps raw web tools
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
                index[name] = ability.get_search_tooltip() or ability.get_summary()
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

    def get_parameters(self) -> dict:
        # Enrich the `select` description with the live discoverable-tools index.
        # Gated on self.mp via _config_scope: at build time (mp=None) the scope is
        # empty, so the base schema is returned unchanged and the search index /
        # SHA map stay deterministic. The framework fields (act_summary / async)
        # are injected uniformly afterwards by the final get_input_schema().
        params = copy.deepcopy(self._PARAMETERS)
        tools_index = self._build_tools_index(self.mp)
        if tools_index:
            params["properties"]["select"]["description"] = (
                f"Exact tool names to activate directly. Available tools: {tools_index}"
            )
        return params

    def run(self, params: dict) -> str:
        """Dispatch to the select or query path and return a tagged result string."""
        select_names = params.get("select")
        query = params.get("query", "").strip()

        # Require at least one param.
        if not select_names and not query:
            return _skill_tag("find_tools", error="params-required")

        proc = self.mp
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
        """Exact case-insensitive match against effective_allow; append matched names.

        Builds a display.lower()→call_name alias map from get_online_mcp_tools_index()
        so callers can use the bare server-reported name (e.g. 'list_tickets') in
        addition to the prefixed call name ('_mcp_taskie_list_tickets').

        Ambiguity rule: if a bare display name maps to more than one distinct call
        name across servers, the alias is dropped — the caller must use the prefixed
        form to avoid silent wrong-server selection.
        """
        allow_lower = {name.lower(): name for name in effective_allow}

        # Build display → call_name alias map for MCP tools in effective_allow.
        # display_to_calls accumulates all call_names for each bare display name so we
        # can detect cross-server collisions before committing to any alias.
        display_to_calls: dict[str, list[str]] = {}
        for call_name, display in self._get_online_mcp_tools_index():
            if call_name not in effective_allow:
                continue
            key = display.lower()
            display_to_calls.setdefault(key, []).append(call_name)

        # Alias is valid only when exactly one call_name owns the bare display name.
        display_alias: dict[str, str] = {
            key: calls[0]
            for key, calls in display_to_calls.items()
            if len(calls) == 1
        }

        matched: list[str] = []
        not_found: list[str] = []

        for name in requested:
            lower = name.lower()
            canonical = allow_lower.get(lower) or display_alias.get(lower)
            if canonical is not None:
                matched.append(canonical)
            else:
                not_found.append(name)

        # Two requested aliases (e.g. the bare display name and its prefixed call
        # name) can resolve to the same canonical tool — collapse so the result
        # JSON and the found= count never double-report a single tool.
        matched = list(dict.fromkeys(matched))

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
            query_embedding = EmbeddingService().generate_embedding(query, mp=self.mp)
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
        No-op when self.mp is None (action-button path) or names is empty.
        """
        if not names:
            return
        proc = self.mp
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
            template = AbilityRegistry.get(name)
            # Bind a fresh per-call instance to mp so the discovered tool's body
            # carries the same framework injection / enrichment build_tools gives
            # it; return the input_schema BODY (what _format_universal embeds).
            return type(template)(mp=self.mp).get_input_schema()["input_schema"]
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
        """Hybrid vec+FTS RRF search against mcp_tools.sqlite for enabled+online tools.

        Delegates entirely to _hybrid_search with the MCP-specific SQL and
        db_path=_MCP_DB_PATH.  When the vec table is absent (sqlite_vec unavailable)
        or the blob is empty/invalid, _hybrid_search degrades gracefully to FTS-only
        via its resilient-vec try/except — no branch needed here.

        Returns rows in the same {key, label, score} format as _query so the caller
        can merge both lists directly.
        """
        if not mcp_names:
            return []
        placeholders = ",".join("?" * len(mcp_names))
        vec_sql = f"""
            SELECT mt.tool_name, mt.summary, v.distance
            FROM mcp_tools_vec v
            JOIN mcp_tool_vectors mv ON mv.rowid = v.rowid
            JOIN mcp_tools mt ON mt.tool_name = mv.tool_name
            WHERE v.embedding MATCH ? AND k = ? AND mt.tool_name IN ({placeholders})
            ORDER BY v.distance ASC
        """
        fts_sql = f"""
            SELECT mt.tool_name, mt.summary, bm25(mcp_tools_fts) AS score
            FROM mcp_tools_fts
            JOIN mcp_tools mt ON mt.id = mcp_tools_fts.rowid
            WHERE mcp_tools_fts MATCH ? AND mt.tool_name IN ({placeholders})
            ORDER BY score ASC
        """
        return self._hybrid_search(
            query, blob, limit,
            vec_sql=vec_sql,
            fts_sql=fts_sql,
            vec_params=(blob, KNN_DEPTH, *mcp_names),
            fts_params=(query, *mcp_names),
            db_path=self._MCP_DB_PATH,
        )

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
