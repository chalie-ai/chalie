"""Feature tests — find_tools enforces tool visibility via the global DISCOVERABLE flag.

Tool visibility is now a single binary: an ability is reachable through
``find_tools`` iff its ``Ability.DISCOVERABLE`` is True (the global roster is
``AbilityRegistry.discoverable_names()``). There is no per-config ``discoverable``
or ``blocked`` list any more. Channel isolation is achieved entirely by:
  (a) whether a tool is ``DISCOVERABLE`` at all, and
  (b) whether the invoking processor carries ``find_tools`` in its
      ``always_available`` list.

So the raw web tools (``browser`` / ``search`` / ``news``) are
``DISCOVERABLE=False`` — they are absent from the global roster and therefore
never selectable on ANY channel; they reach the model only by being pinned into a
delegate config's ``always_available`` (``WebBrowseConfig`` / ``WebSearchConfig``).
The delegate wrappers (``web_browse`` / ``web_search`` / ``vision``) are
``DISCOVERABLE=True`` and stay selectable on every channel that carries
find_tools — the user channel AND the DMN background channel.

These run against the REAL production stack:
- Real ``FindToolsAbility.run()`` — the same call ToolDispatcher makes.
- Real ``MessageProcessor`` instances carrying the REAL channel configs
  (``UserConfig`` / ``DmnConfig`` / ``WebBrowseConfig``).
- Real ``AbilityRegistry.build_tools()`` resolution against the real registry.
- Real abilities.sqlite + real EmbeddingService for the query path.

No mocks, no stub configs — the configs under test are the ones production runs.
"""

from typing import Mapping, Sequence, cast

import pytest

from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from configs.channels import DmnConfig, UserConfig
from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit


def _mp_for(config: ProcessorConfig) -> MessageProcessor:
    mp = MessageProcessor(config, raw_input="find a tool for me")
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict[str, object]) -> str | Mapping[str, object] | Sequence[object]:
    """Drive the real find_tools dispatch path: bind the invoking mp and run.

    ``run()`` returns a ``ToolResult``; its ``body`` is a structured dict on
    success (``{"injected": [...], "not_found": [...]}``) and the error message
    string on failure (the dispatcher adds the ``[find_tools(...)]`` envelope)."""
    ability = FindToolsAbility()
    ability.mp = mp
    return ability.run(built(FindToolsParamsBag.from_params(params))).body


# ---------------------------------------------------------------------------
# Headline regression — raw browser/search/news are non-discoverable globally,
# so they are never selectable on the user channel.
# ---------------------------------------------------------------------------

class TestUserChannelCannotReachRawWebTools:

    @pytest.mark.parametrize("raw", ["browser", "search", "news"])
    def test_raw_web_tool_is_never_injected_on_user_channel(self, raw: str, db: object) -> None:
        """A query naming a raw web tool (``browser`` / ``search`` / ``news``) must
        NEVER inject that raw tool — it is ``DISCOVERABLE=False``, absent from the
        global roster and from the index the cascade searches (raw web is exclusive
        to the web_browse/web_search delegates). The cascade may helpfully surface a
        discoverable sibling, but the raw name itself must never reach active_tools."""
        mp = _mp_for(UserConfig())

        _find_tools_on(mp, {"query": [raw]})

        assert raw not in mp.active_tools, (
            f"Raw {raw!r} must never be activatable on the user channel — it is "
            f"DISCOVERABLE=False. active_tools={mp.active_tools}"
        )

    def test_query_cannot_surface_browser_on_user_channel(self, db: object) -> None:
        """Even a query that is a perfect semantic match for browser must not
        inject it — browser is DISCOVERABLE=False, so it never enters the global
        roster the query path searches over."""
        mp = _mp_for(UserConfig())
        _find_tools_on(mp, {"query": ["control a headless browser, click buttons and take a screenshot"]})

        assert "browser" not in mp.active_tools, (
            "A non-discoverable tool must never be injected via the query path, "
            f"regardless of relevance. active_tools={mp.active_tools}"
        )

    def test_delegate_tools_remain_available_on_user_channel(self, db: object) -> None:
        """The model must not over-block: the delegate tools (web_browse /
        web_search) are how the user channel reaches the web, and stay
        selectable (they are DISCOVERABLE=True)."""
        mp = _mp_for(UserConfig())
        result = _find_tools_on(mp, {"query": ["web_browse", "web_search"]})

        assert "web_browse" in mp.active_tools, (
            f"web_browse delegate must stay available on user. active_tools={mp.active_tools}"
        )
        assert "web_search" in mp.active_tools, (
            f"web_search delegate must stay available on user. active_tools={mp.active_tools}"
        )
        assert cast(Mapping[str, object], result)["not_found"] == [], (
            f"delegate tools must not be reported unavailable. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Background channel — raw web tools are still non-discoverable, but the DMN
# CAN now discover the delegate wrappers (global discovery; it carries find_tools).
# ---------------------------------------------------------------------------

class TestBackgroundChannelDiscovery:

    def test_dmn_cannot_reach_raw_web_or_pattern_tools(self, db: object) -> None:
        """The raw, non-discoverable tools (browser/search/news/save_pattern/
        save_graph) are absent from the global roster, so a DMN query naming them
        can never inject them either."""
        mp = _mp_for(DmnConfig())
        non_discoverable = ["browser", "search", "news", "save_pattern", "save_graph"]

        _find_tools_on(mp, {"query": non_discoverable})

        leaked = [n for n in non_discoverable if n in mp.active_tools]
        assert not leaked, (
            f"DMN must not reach non-discoverable tools {non_discoverable}; leaked: {leaked}"
        )

    def test_dmn_can_discover_the_delegate_wrappers(self, db: object) -> None:
        """Discovery is global now: the DMN carries find_tools and the delegate
        wrappers are DISCOVERABLE=True, so the background loop can discover and
        spawn web_browse / web_search / vision. Keeping that work out of the
        subconscious is the policy gate's job, NOT a discovery block."""
        mp = _mp_for(DmnConfig())

        result = _find_tools_on(mp, {"query": ["web_browse", "web_search", "vision"]})

        for name in ("web_browse", "web_search", "vision"):
            assert name in mp.active_tools, (
                f"DMN must be able to discover the {name} delegate. "
                f"active_tools={mp.active_tools}"
            )
        assert cast(Mapping[str, object], result)["not_found"] == [], (
            f"delegate wrappers must not be reported unavailable on DMN. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Positive control — browser IS delivered on the web_browse delegate channel.
# ---------------------------------------------------------------------------

class TestWebBrowseDelegateHasBrowser:

    def test_browser_resolves_into_the_tool_schemas_on_web_browse_channel(self, db: object) -> None:
        """WebBrowseConfig pins browser as always-available, so build_tools
        resolves a real 'browser' tool schema for the delegate — the channel
        raw browser is exclusive to. (always_available bypasses discovery
        entirely: a non-discoverable tool is reached ONLY this way.)"""
        mp = _mp_for(WebBrowseConfig(PolicyChannel.CHAT))
        assert "browser" in mp.config.always_available

        schemas = AbilityRegistry.build_tools(mp)
        names = {cast(str, s["name"]) for s in schemas}

        assert "browser" in names, (
            f"web_browse delegate must be handed the real browser tool. names={sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Contract guard — pin the exact set of non-discoverable abilities.
# ---------------------------------------------------------------------------

# The single source of truth for tool visibility: every ability NOT in this set
# is DISCOVERABLE=True and reachable via find_tools on any channel that carries
# it. A silent flip of any ability's DISCOVERABLE flag trips this guard.
_EXPECTED_NON_DISCOVERABLE = frozenset({
    "browser",                 # raw web — web_browse delegate only
    "search",                  # raw web — web_search delegate only
    "news",                    # raw web — web_search delegate only ()
    "save_graph",              # pattern-write — PatternMatchProcessor only
    "save_pattern",            # pattern-write — PatternMatchProcessor only
    "find_tools",              # the discovery entry point itself
    "find_skills",             # framework — always_available only
    "memory",                  # framework — always_available only
    "chat_history_compactor",  # internal — dispatched programmatically
    "document",                # internal — turn-zero attachment transition
    "skill_manager",           # system variant — SkillSuggestionConfig only
    "email",                   # personal info — pim delegate only
    "calendar",                # personal info — pim delegate only
    "contacts",                # personal info — pim delegate only
    "run_script",               # code-agent-delegate-exclusive — CodeAgentConfig only
})


def test_non_discoverable_set_is_exactly_the_expected_set() -> None:
    """Pin the whole visibility model: the set of registered abilities that are
    NOT in the global discovery roster must equal the expected non-discoverable
    set. Flipping any DISCOVERABLE flag (or adding a new non-discoverable
    ability) without updating this guard fails here."""
    all_names = {a.get_name() for a in AbilityRegistry.all()}
    non_discoverable = all_names - AbilityRegistry.discoverable_names()

    assert non_discoverable == _EXPECTED_NON_DISCOVERABLE, (
        "Non-discoverable ability set drifted.\n"
        f"  unexpected (now non-discoverable): {sorted(non_discoverable - _EXPECTED_NON_DISCOVERABLE)}\n"
        f"  missing (now discoverable):        {sorted(_EXPECTED_NON_DISCOVERABLE - non_discoverable)}"
    )
