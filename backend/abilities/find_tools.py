import copy
import logging
from pathlib import Path
from typing import ClassVar, cast

from abilities._params import Keys
from abilities._result import ToolResult
from abilities._search import KNN_DEPTH, SearchableAbility
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class FindToolsAbility(SearchableAbility):
    """Discover and activate tools for the current ACT turn.

    Supports two mutually exclusive selection modes:
    - select: exact case-insensitive match against the effective allow-list.
    - query: hybrid vec+FTS RRF semantic search with a relevance floor.
    When both are supplied, select takes precedence and query is ignored.
    """

    DISCOVERABLE: ClassVar[bool] = False  # the discovery entry point itself; pinned, never discovered

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.select: {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact tool names to activate directly.",
            },
            Keys.query: {
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
    def _discoverable_allow() -> set[str]:
        """The single global discovery roster: every DISCOVERABLE ability.

        Discovery scope no longer varies per channel — a tool is reachable here
        iff its ``Ability.DISCOVERABLE`` is True. Channel isolation is achieved
        entirely by (a) the flag and (b) whether the invoking processor carries
        find_tools at all. MCP tools are added separately from the online set."""
        from abilities._registry import AbilityRegistry

        return AbilityRegistry.discoverable_names()

    def _build_tools_index(self) -> str:
        from abilities._registry import AbilityRegistry

        index = {}
        for name in sorted(self._discoverable_allow()):
            try:
                ability = AbilityRegistry.get(name)
                index[name] = ability.get_search_tooltip() or ability.get_summary()
            except KeyError:
                pass

        # MCP tools are listed by their bare server-reported name only
        # (e.g. `list_tickets`), no tooltip.  The MCP protocol's per-tool
        # `description` is matched at search time (FTS over mcp_tools.sqlite),
        # not surfaced in this browse hint.
        mcp_display = [display for _call_name, display in self._get_online_mcp_tools_index()]

        if not index and not mcp_display:
            return ""
        parts = [f"`{k}` ({v})" for k, v in index.items()]
        parts.extend(f"`{n}`" for n in mcp_display)
        return ", ".join(parts)

    @staticmethod
    def _get_online_mcp_names() -> list[str]:
        try:
            from services.mcp_client_service import McpClientService
            return McpClientService().get_online_mcp_tool_names()
        except Exception as exc:
            logger.debug("[FIND_TOOLS] Could not fetch MCP tool names: %s", exc)
            return []

    @staticmethod
    def _get_online_mcp_tools_index() -> list[tuple[str, str]]:
        try:
            from services.mcp_client_service import McpClientService
            return McpClientService().get_online_mcp_tools_index()
        except Exception as exc:
            logger.debug("[FIND_TOOLS] Could not fetch MCP tool index: %s", exc)
            return []

    def get_parameters(self) -> dict[str, object]:
        # Enrich the `select` description with the global discoverable-tools index.
        # The roster is the same on every channel (AbilityRegistry.discoverable_names);
        # find_tools is itself DISCOVERABLE=False, so it never lands in the search
        # index / SHA map and this enrichment never feeds the build. The framework
        # fields (act_summary / async) are injected uniformly afterwards by the
        # final get_input_schema().
        params: dict[str, object] = copy.deepcopy(self._PARAMETERS)
        tools_index = self._build_tools_index()
        if tools_index:
            props = cast("dict[str, dict[str, object]]", params["properties"])
            props[Keys.select]["description"] = (
                f"Exact tool names to activate directly. Available tools: {tools_index}"
            )
        return params

    def run(self, params: dict[str, object]) -> ToolResult:
        """Dispatch to the select or query path and return a ToolResult.

        The result is structured so a weak model can tell what it actually got:
        the success body is ``{"injected": [{"name", "summary"}, …], "not_found":
        […]}`` with ``injected``/``not_found`` counts in the meta, and a request
        that yields nothing usable errors loudly (``unknown-tool``) with a
        ``valid:`` ladder of real selectable names — never a prose-only result
        that hides whether a tool was injected.
        """
        select_names: list[str] | None = cast("list[str] | None", params.get(Keys.select))
        query = cast("str", params.get(Keys.query, "")).strip()

        # Require at least one param.
        if not select_names and not query:
            return ToolResult.err(
                "find_tools requires either 'select' (exact tool names) or 'query' "
                "(a description of what you need).",
                code="missing-params",
                valid=("select", "query"),
            )

        # The discovery roster is global: every DISCOVERABLE ability plus the
        # online MCP tools. There is no per-channel block list — a non-discoverable
        # tool is simply absent from this set, so select reports it unknown and
        # query never ranks it.
        allow = self._discoverable_allow()
        mcp_names = self._get_online_mcp_names()
        effective_allow = allow | set(mcp_names)

        # select wins over query when both are provided.
        if select_names:
            return self._run_select(select_names, effective_allow)

        if not effective_allow:
            return ToolResult.ok({"injected": [], "not_found": []}, injected=0, query=query)

        logger.info("%s query='%s'", self._LOG_PREFIX, query)
        return self._run_query(query, list(allow), list(mcp_names))

    def _run_select(
        self, requested: list[str], effective_allow: set[str]
    ) -> ToolResult:
        """A name absent from the global discoverable roster is NEVER injected; it
        lands in not_found and, when every requested name is unusable, drives a
        loud ``unknown-tool`` error so the model never believes a non-discoverable
        tool was injected.

        Ambiguity rule: if a bare display name maps to more than one distinct call
        name across servers, the alias is dropped — the caller must use the prefixed
        form to avoid silent wrong-server selection."""
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
        # JSON and the injected= count never double-report a single tool.
        matched = list(dict.fromkeys(matched))

        # Nothing usable: loud error so the model self-corrects rather than
        # believing a non-discoverable / unknown tool was injected. The valid
        # ladder is the real selectable names (the global discoverable roster).
        if not matched:
            valid = tuple(sorted(effective_allow))
            return ToolResult.err(
                f"Tools not found or unavailable: {', '.join(not_found)}.",
                code="unknown-tool",
                hint="No tool by that name is selectable here; pick one from the valid list.",
                valid=valid,
            )

        self._append_active(matched)
        body = {
            "injected": [{"name": n, "summary": self._summary_for(n)} for n in matched],
            "not_found": not_found,
        }
        if not_found:
            return ToolResult.ok(body, injected=len(matched), not_found=len(not_found))
        return ToolResult.ok(body, injected=len(matched))

    def _run_query(self, query: str, allow: list[str], mcp_names: list[str]) -> ToolResult:
        effective_allow = list(set(allow) | set(mcp_names))

        try:
            from services.embedding_service import EmbeddingService
            query_embedding = EmbeddingService().generate_embedding(query, mp=self.mp)
        except Exception as exc:
            logger.warning("%s Embedding generation failed: %s", self._LOG_PREFIX, exc)
            return self._fallback(query, effective_allow)

        blob = cast("bytes", pack_embedding(query_embedding))
        rows = self._query(query, blob, allow)
        mcp_rows = self._query_mcp(query, blob, self.MAX_QUERY_RESULTS, mcp_names)
        rows = self._merge_and_truncate(rows + mcp_rows)

        # Apply relevance floor then cap to MAX_QUERY_RESULTS.
        rows = [r for r in rows if cast("float", r["score"]) >= self.MIN_RRF_SCORE][:self.MAX_QUERY_RESULTS]

        for row in rows:
            logger.info("%s RRF: %s score=%.4f", self._LOG_PREFIX, row["key"], row["score"])

        names = [cast("str", r["key"]) for r in rows]
        self._append_active(names)
        return ToolResult.ok(self._query_body(names), injected=len(names), query=query)

    def _append_active(self, names: list[str]) -> None:
        if not names:
            return
        proc = self.mp
        if proc is None:
            return
        active: list[str] = cast("list[str]", getattr(proc, "active_tools", None))
        if active is None:
            return
        for name in names:
            if name not in active:
                active.append(name)

    def _query_body(self, names: list[str]) -> dict[str, object]:
        return {
            "injected": [{"name": n, "summary": self._summary_for(n)} for n in names],
            "not_found": [],
        }

    def _summary_for(self, name: str) -> str:
        """Never raises — returns an empty string on any failure so the caller can
        always produce a body."""
        try:
            if name.startswith("_mcp_"):
                from services.mcp_client_service import McpClientService
                result = McpClientService().get_tool_schema(name)
                if result:
                    return cast("str", result.get("description")) or ""
                return ""
            from abilities._registry import AbilityRegistry
            ability = AbilityRegistry.get(name)
            return ability.get_search_tooltip() or ability.get_summary() or ""
        except Exception as exc:
            logger.warning("%s Summary lookup failed for %r: %s", self._LOG_PREFIX, name, exc)
            return ""

    def _query(self, query: str, blob: bytes, allow: list[str]) -> list[dict[str, object]]:
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

    def _query_mcp(self, query: str, blob: bytes, limit: int, mcp_names: list[str]) -> list[dict[str, object]]:
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
    def _merge_and_truncate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        seen: set[str] = set()
        merged = []
        for row in rows:
            if cast("str", row["key"]) not in seen:
                seen.add(cast("str", row["key"]))
                merged.append(row)
        merged.sort(key=lambda r: cast("float", r["score"]), reverse=True)
        return merged

    def _fallback(self, query: str, allow: list[str]) -> ToolResult:
        if not allow:
            return ToolResult.ok({"injected": [], "not_found": []}, injected=0, query=query)

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
            discovered.extend(cast("str", row[0]) for row in rows)

        for row in self._query_mcp(query, b"", self.MAX_QUERY_RESULTS, mcp_names):
            if cast("str", row["key"]) not in discovered:
                discovered.append(cast("str", row["key"]))

        discovered = discovered[:self.MAX_QUERY_RESULTS]

        self._append_active(discovered)
        return ToolResult.ok(self._query_body(discovered), injected=len(discovered), query=query)
