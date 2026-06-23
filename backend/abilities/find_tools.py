import copy
import logging
from pathlib import Path
from typing import ClassVar, cast

from abilities._params import Keys
from abilities._result import ToolResult
from abilities._search import KNN_DEPTH, SearchableAbility, build_keyword_query
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class FindToolsAbility(SearchableAbility):
    """Discover and activate tools for the current ACT turn.

    Supports two mutually exclusive selection modes:
    - select: exact case-insensitive match against the effective allow-list.
    - query: keyword (trigram FTS5) + vector search run independently; the top
      results of each are deduped and injected. No score fusion, no floor.
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
                "description": (
                    "Keywords describing the capability — NOT a sentence. Prefix a "
                    "term with + to require it, - to exclude it; a bare term is "
                    "optional. Substring matching, e.g. +docs matches `chalie_docs`. "
                    "Example: +calendar +event -delete"
                ),
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

    # Per-signal cap: up to this many keyword + this many vector results are
    # deduped into the injected set (so at most 2×_RESULT_CAP distinct tools).
    _RESULT_CAP: ClassVar[int] = 3

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
        fts_match, embed_text = build_keyword_query(query)

        blob: bytes | None = None
        if embed_text:
            try:
                from services.embedding_service import EmbeddingService
                blob = cast("bytes", pack_embedding(EmbeddingService().generate_embedding(embed_text, mp=self.mp)))
            except Exception as exc:
                logger.warning("%s Embedding generation failed (keyword-only): %s", self._LOG_PREFIX, exc)

        # Keyword and vector search run independently over both the curated
        # ability index and the runtime MCP index; the top of each is deduped.
        vec_rows = self._vec_rows(blob, allow) + self._vec_rows_mcp(blob, mcp_names)
        fts_rows = self._fts_rows(fts_match, allow) + self._fts_rows_mcp(fts_match, mcp_names)
        results = self._combine(vec_rows, fts_rows, self._RESULT_CAP)

        for row in results:
            logger.info("%s injected %s via %s", self._LOG_PREFIX, row["key"], row["source"])

        names = [cast("str", r["key"]) for r in results]
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

    def _vec_rows(self, blob: bytes | None, allow: list[str]) -> "list[tuple[object, object, float]]":
        if blob is None or not allow:
            return []
        placeholders = ",".join("?" * len(allow))
        return self._vec_search(
            blob,
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
        )

    def _fts_rows(self, fts_match: str, allow: list[str]) -> "list[tuple[object, object, float]]":
        if not fts_match or not allow:
            return []
        placeholders = ",".join("?" * len(allow))
        return self._fts_search(
            fts_match,
            f"""
                SELECT a.name, a.summary, bm25(ability_search_fts) AS score
                FROM ability_search_fts
                JOIN ability_search_entries e ON e.id = ability_search_fts.rowid
                JOIN abilities a ON a.id = e.ability_id
                WHERE ability_search_fts MATCH ?
                  AND a.name IN ({placeholders})
                ORDER BY score ASC
            """,
            (fts_match, *allow),
        )

    def _vec_rows_mcp(self, blob: bytes | None, mcp_names: list[str]) -> "list[tuple[object, object, float]]":
        if blob is None or not mcp_names:
            return []
        placeholders = ",".join("?" * len(mcp_names))
        return self._vec_search(
            blob,
            f"""
                SELECT mt.tool_name, mt.summary, v.distance
                FROM mcp_tools_vec v
                JOIN mcp_tool_vectors mv ON mv.rowid = v.rowid
                JOIN mcp_tools mt ON mt.tool_name = mv.tool_name
                WHERE v.embedding MATCH ? AND k = ? AND mt.tool_name IN ({placeholders})
                ORDER BY v.distance ASC
            """,
            (blob, KNN_DEPTH, *mcp_names),
            db_path=self._MCP_DB_PATH,
        )

    def _fts_rows_mcp(self, fts_match: str, mcp_names: list[str]) -> "list[tuple[object, object, float]]":
        if not fts_match or not mcp_names:
            return []
        placeholders = ",".join("?" * len(mcp_names))
        return self._fts_search(
            fts_match,
            f"""
                SELECT mt.tool_name, mt.summary, bm25(mcp_tools_fts) AS score
                FROM mcp_tools_fts
                JOIN mcp_tools mt ON mt.id = mcp_tools_fts.rowid
                WHERE mcp_tools_fts MATCH ? AND mt.tool_name IN ({placeholders})
                ORDER BY score ASC
            """,
            (fts_match, *mcp_names),
            db_path=self._MCP_DB_PATH,
        )
