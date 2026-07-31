"""Discover and activate tools for the current ACT turn.

``query`` is an array of tool names — one name per entry. Each entry is
normalized (lowercased, non-alphanumeric runs collapsed to a single space,
edges stripped) and looked up in the registry's alias map (canonical names and
declared ``SEARCHABLE_AS`` aliases). A hit activates the tool and returns it in
``injected`` with its tooltip; a miss is returned in ``not_found`` with a
pointer back to the tool list.

No fuzzy search, no external tool surface — this tool is a precise menu.
"""

import logging
import re
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from contracts.params.param_bag import ParamBag

logger = logging.getLogger(__name__)


class FindToolsAbility(Ability):
    """Exact-match tool menu — one name per array entry.

    Describing an action does not work. Each entry must be the canonical tool
    name (or a declared ``SEARCHABLE_AS`` alias) — normalized and looked up
    against the registry's alias map.
    """

    PARAMS: ClassVar[type[ParamBag] | None] = FindToolsParamsBag
    NAME: ClassVar[str] = "find_tools"

    DISCOVERABLE: ClassVar[bool] = False  # the discovery entry point itself; pinned, never discovered

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.query: {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One tool name per entry. Names are matched exactly against "
                    "the canonical name or a declared SEARCHABLE_AS alias."
                ),
            },
        },
        "required": [Keys.query],
    }


    # The two standing lines above the menu. Static: they describe how to call
    # find_tools, which never varies by request — only the menu below them does.
    _INTRO: ClassVar[str] = (
        "Use this tool (`find_tools`) to load more tools into context so that you "
        "can perform more actions.\n"
        "To use the `find_tools` simply supply the list of tool names VERBATIM "
        "from the list below in the `query` parameter."
    )

    # Shown instead of the menu once every discoverable tool is already loaded.
    # An empty "**Available Tools**" heading would read as "no tools exist"; this
    # says the opposite — they are all here, call one directly.
    _NOTHING_LEFT: ClassVar[str] = (
        "Every available tool is already loaded in context — there is nothing "
        "left to load. Call the tool you need directly."
    )

    def get_summary(self) -> str:
        menu = self._build_tools_index()
        if not menu:
            return f"{self._INTRO}\n\n{self._NOTHING_LEFT}"
        return f"{self._INTRO}\n\n**Available Tools**\n{menu}"

    def get_examples(self) -> list[str]:
        return [
            'Check the weather — query ["weather"]',
            'Search the web — query ["web_search"]',
            'Work with files — query ["write_file", "make_dir", "delete"]',
            'Read a page or file — query ["read"]',
            'Email or calendar — query ["pim"]',
            'Set a timer — query ["timer"]',
            'Run a shell command — query ["bash"]',
            'How Chalie works — query ["chalie_docs"]',
        ]

    def get_search_tooltip(self) -> str:
        return "discover available tools"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Announce the freshly-activated tools by name — live for this turn."""
        injected = tr.body.get("injected") if isinstance(tr.body, dict) else None
        rows = injected if isinstance(injected, list) else []
        names = [n["name"] for n in rows]
        if not names:
            return ""
        joined = ", ".join(f"`{n}`" for n in names)
        verb = "is" if len(names) == 1 else "are"
        return f"{joined} {verb} now available and can be called."

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: FindToolsParamsBag) -> ToolResult:
        """Run each query entry through exact-match lookup and inject what wins.

        The success body is ``{"injected": [{"name", "summary"}, …], "not_found":
        [...]}`` with ``injected``/``not_found`` counts in the meta, so a weak
        model can always tell what it actually got. ``not_found`` lists entries
        that did not match any canonical name or alias, with guidance to pick
        from the tool list. When every entry misses (``injected`` empty) this
        returns a loud ``code=no-results`` error instead of a quiet zero-row
        success; a partial hit (some names matched) stays a success."""
        rows = params.query

        from abilities._registry import AbilityRegistry

        aliases = AbilityRegistry.discovery_aliases()
        not_found: list[str] = []
        injected: list[str] = []

        for entry in rows:
            norm = self._normalize(entry)
            canonical = aliases.get(norm)
            if canonical is None:
                not_found.append(
                    f"'{entry}' did not match any tool name. "
                    "Pick an exact tool name from the list in this tool's description."
                )
                continue
            if canonical not in injected:
                injected.append(canonical)

        self._append_active(injected)

        body = {
            "injected": [{"name": n, "summary": self._summary_for(n)} for n in injected],
            "not_found": not_found,
        }
        if not injected:
            return ToolResult.no_results(
                hint="Pick an exact tool name from the list in this tool's description.",
                not_found=len(not_found),
            )
        if not_found:
            return ToolResult.ok(body, injected=len(injected), not_found=len(not_found))
        return ToolResult.ok(body, injected=len(injected))

    @staticmethod
    def _normalize(entry: str) -> str:
        """Lowercase, collapse every non-[a-z0-9] run to a single space, strip edges."""
        return re.sub(r"[^a-z0-9]+", " ", entry.lower()).strip()

    def _summary_for(self, name: str) -> str:
        """Never raises — returns an empty string on any failure so the caller can
        always produce a body."""
        try:
            from abilities._registry import AbilityRegistry
            ability = AbilityRegistry.get(name)
            return ability.get_search_tooltip() or ability.get_summary() or ""
        except Exception as exc:
            logger.warning("FIND_TOOLS Summary lookup failed for %r: %s", name, exc)
            return ""

    def _build_tools_index(self) -> str:
        """The discoverable roster grouped under its category headings, minus
        whatever is already loaded.

        A tool the model can already call is noise in a menu whose only purpose is
        loading tools it cannot — so anything in ``mp.active_tools`` is dropped,
        and a category left with no tools disappears with it rather than rendering
        an empty heading. Categories come out in ``AbilityCategory`` declaration
        order, tools alphabetically inside each.

        Off-spine (``mp is None`` — introspection, index build) nothing is loaded,
        so the full roster renders and the text stays deterministic.
        """
        from abilities._registry import AbilityRegistry

        loaded = set(cast("list[str]", getattr(self.mp, "active_tools", None) or []))

        grouped: dict[AbilityCategory, list[str]] = {}
        for name in sorted(AbilityRegistry.discoverable_names()):
            if name in loaded:
                continue
            try:
                ability = AbilityRegistry.get(name)
            except KeyError:
                continue
            category = ability.CATEGORY
            if category is None:  # unreachable — the registry rejects it at load
                continue
            tooltip = ability.get_search_tooltip() or ability.get_summary()
            grouped.setdefault(category, []).append(f"- `{name}`: {tooltip}")

        return "\n\n".join(
            "\n".join([f"{category.value}:", *grouped[category]])
            for category in AbilityCategory
            if category in grouped
        )
