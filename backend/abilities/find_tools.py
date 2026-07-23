import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from configs.enums.param_key import Keys
from abilities._result import ToolResult
from abilities._search import KNN_DEPTH, SearchableAbility
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from contracts.params.param_bag import ParamBag

logger = logging.getLogger(__name__)


class FindToolsAbility(SearchableAbility):
    """Discover and activate tools for the current ACT turn.

    ``query`` is an array of intents — one tool name or one described action per
    entry — run through the shared precise→broad cascade (see
    :meth:`SearchableAbility._discover`): exact name → bm25 on the tool NAME only
    (segment-gated) → vector on the full prose (``VEC_CEILING``). find_tools adds
    the MCP surface and the bare-name ambiguity guard; everything else is shared.
    """

    PARAMS: ClassVar["type[ParamBag] | None"] = FindToolsParamsBag

    DISCOVERABLE: ClassVar[bool] = False  # the discovery entry point itself; pinned, never discovered

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "array",
                "items": {"type": "string"},
                "description": "One tool name or one described action per entry.",
            },
        },
        "required": [Keys.query],
    }

    def get_name(self) -> str:
        return "find_tools"

    def get_summary(self) -> str:
        base = (
            "Discover and activate the tools you need for this turn. Search by naming "
            "a tool, or by describing the action you want to perform — one intent per "
            "array entry."
        )
        roster = self._build_tools_index()
        return f"{base} Available tools: {roster}" if roster else base

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

    def get_follow_up(self, tr: ToolResult) -> str:
        """Announce the freshly-activated tools by name — live for this turn."""
        injected = tr.body.get("injected") if isinstance(tr.body, dict) else None
        names = [cast("dict[str, str]", n)["name"] for n in injected] if injected else []
        if not names:
            return ""
        joined = ", ".join(f"`{n}`" for n in names)
        verb = "is" if len(names) == 1 else "are"
        return f"{joined} {verb} now available and can be called."

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _DB_PATH: ClassVar[Path] = FileMapperService.get_abilities_db_path()
    # Separate runtime DB for dynamically-synced MCP client tools.
    # Queried in addition to _DB_PATH so build_ability_db rebuilds never
    # destroy _mcp_* rows.  See McpClientService and FileMapperService.
    _MCP_DB_PATH: ClassVar[Path] = FileMapperService.get_mcp_tools_db_path()
    _LOG_PREFIX: ClassVar[str] = "[FIND_TOOLS]"

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

    def run(self, params: FindToolsParamsBag) -> ToolResult:
        """Run each query entry through the shared cascade and inject what wins.

        The success body is ``{"injected": [{"name", "summary"}, …], "not_found":
        [...]}`` with ``injected``/``not_found`` counts in the meta, so a weak model
        can always tell what it actually got. ``not_found`` lists entries the
        cascade could not satisfy plus any bare MCP name that was ambiguous
        (refused, with the prefixed-name guidance, rather than silently resolved)."""
        rows = params.query

        # The discovery roster is global: every DISCOVERABLE ability plus the
        # online MCP tools. A non-discoverable tool is simply absent from this set.
        mcp_names = self._get_online_mcp_names()
        effective_allow = self._discoverable_allow() | set(mcp_names)
        allow_lower = {self._norm(name): name for name in effective_allow}

        # Bare MCP display name → all call_names owning it, so an exact bare name
        # that collides across servers can be refused instead of silently pinned.
        display_to_calls: dict[str, list[str]] = {}
        for call_name, display in self._get_online_mcp_tools_index():
            if call_name in effective_allow:
                display_to_calls.setdefault(self._norm(display), []).append(call_name)

        allow_list, mcp_list = list(effective_allow), list(mcp_names)
        pins, disc, not_found = self._discover(
            rows,
            lambda candidate: self._exact(candidate, allow_lower, display_to_calls),
            lambda terms: self._bm25_name(terms, allow_list, mcp_list),
            lambda terms: self._vector_name(terms, allow_list, mcp_list),
        )
        injected = cast("list[str]", pins + disc)

        for name in injected:
            logger.info("%s injected %s", self._LOG_PREFIX, name)
        self._append_active(injected)

        body = {
            "injected": [{"name": n, "summary": self._summary_for(n)} for n in injected],
            "not_found": not_found,
        }
        if not_found:
            return ToolResult.ok(body, injected=len(injected), not_found=len(not_found))
        return ToolResult.ok(body, injected=len(injected))

    @staticmethod
    def _exact(
        candidate: str,
        allow_lower: dict[str, str],
        display_to_calls: dict[str, list[str]],
    ) -> tuple[str | None, bool]:
        """Resolve the whole normalised entry against tool names.

        Returns ``(canonical_name | None, ambiguous)``. A real ability/MCP
        canonical name wins outright; a bare MCP display name owned by exactly
        one server resolves to it; one owned by more than one server is ambiguous
        (refuse + surface). Chalie names are single underscored tokens, so
        ``"the calendar"`` → ``calendar`` pins but ``"delete my calendar event"``
        does not auto-pin ``calendar`` — it is a fuzzy intent, not a name."""
        canonical = allow_lower.get(candidate)
        if canonical:
            return canonical, False
        calls = display_to_calls.get(candidate, [])
        if len(calls) > 1:
            return None, True
        if len(calls) == 1:
            return calls[0], False
        return None, False

    def _bm25_name(self, terms: list[str], allow: list[str], mcp_names: list[str]) -> list[object]:
        """Rung 2: bm25 over the tool NAME only, gated by name-segment alignment.

        bm25 supplies the ranking; the segment gate (NOT a score floor) decides
        membership, rejecting incidental substrings so prose escalates to vector."""
        fts = self._fts_or(terms)
        if not fts:
            return []
        rows = self._fts_rows(fts, allow) + self._fts_rows_mcp(fts, mcp_names)
        return self._gate(rows, terms)

    def _vector_name(self, terms: list[str], allow: list[str], mcp_names: list[str]) -> list[object]:
        """Rung 3: vector search on the full prose, per-name deduped, kept only
        at/under ``VEC_CEILING``."""
        blob = self._embed(terms)
        if blob is None:
            return []
        rows = self._vec_rows(blob, allow) + self._vec_rows_mcp(blob, mcp_names)
        return self._ceiling(rows)

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
                SELECT a.name, a.name, v.distance
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
        """bm25 over the tool NAME entry only (``kind='name'``); the segment gate
        in ``_gate`` filters the rows this returns by ``row[1]`` (the name)."""
        if not fts_match or not allow:
            return []
        placeholders = ",".join("?" * len(allow))
        return self._fts_search(
            fts_match,
            f"""
                SELECT a.name, a.name, bm25(ability_search_fts) AS score
                FROM ability_search_fts
                JOIN ability_search_entries e ON e.id = ability_search_fts.rowid
                JOIN abilities a ON a.id = e.ability_id
                WHERE ability_search_fts MATCH ? AND e.kind = 'name'
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
                SELECT mt.tool_name, mt.tool_name, v.distance
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
        """MCP has no ``kind='name'`` split, so this matches name+description; the
        segment gate on ``tool_name`` in ``_gate`` keeps only name hits."""
        if not fts_match or not mcp_names:
            return []
        placeholders = ",".join("?" * len(mcp_names))
        return self._fts_search(
            fts_match,
            f"""
                SELECT mt.tool_name, mt.tool_name, bm25(mcp_tools_fts) AS score
                FROM mcp_tools_fts
                JOIN mcp_tools mt ON mt.id = mcp_tools_fts.rowid
                WHERE mcp_tools_fts MATCH ? AND mt.tool_name IN ({placeholders})
                ORDER BY score ASC
            """,
            (fts_match, *mcp_names),
            db_path=self._MCP_DB_PATH,
        )
