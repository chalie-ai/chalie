"""Feature tests — find_tools enforces per-channel tool visibility.

The ProcessorConfig typed-subclass refactor made tool visibility
per-channel: every ``ProcessorConfig`` carries ``discoverable`` and ``blocked``.
``find_tools`` MUST gate discovery on the *invoking processor's* config — not on
a global static list — so that raw web tools (``browser`` / ``search``) stay
exclusive to the delegate channels (``WebBrowseConfig`` / ``WebSearchConfig``)
and never leak into the user channel or background loops.

These run against the REAL production stack:
- Real ``FindToolsAbility.run()`` — the same call ToolDispatcher makes.
- Real ``MessageProcessor`` instances carrying the REAL channel configs
  (``UserConfig`` / ``DmnConfig`` / ``WebBrowseConfig``).
- Real ``AbilityRegistry.build_tools()`` resolution against the real registry.
- Real abilities.sqlite + real EmbeddingService for the query path.

No mocks, no stub configs — the configs under test are the ones production runs.
"""

import pytest

from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from configs.channels import DmnConfig, UserConfig
from configs.channels._common import DELEGATE_INTERNAL_TOOLS
from configs.channels.web_browse import WebBrowseConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


def _mp_for(config) -> MessageProcessor:
    """A real MessageProcessor bound to a real channel config, seeded exactly
    as ``_setup`` seeds it (active_tools = config.always_available)."""
    mp = MessageProcessor("find a tool for me")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict) -> str:
    """Drive the real find_tools dispatch path: bind the invoking mp and run.

    ``run()`` returns a ``ToolResult``; its ``body`` is the model-facing
    select/query text (the dispatcher adds the ``[find_tools(...)]`` envelope)."""
    ability = FindToolsAbility()
    ability.mp = mp
    return ability.run(params).body


# ---------------------------------------------------------------------------
# Headline regression — raw browser/search are blocked on the user channel.
# ---------------------------------------------------------------------------

class TestUserChannelBlocksRawWebTools:

    def test_select_browser_is_rejected_on_user_channel(self):
        """``find_tools(select=['browser'])`` on the user channel must NOT add
        browser — UserConfig.blocked contains it (raw browser is exclusive to
        the web_browse delegate). The result reports it as unavailable."""
        mp = _mp_for(UserConfig())
        # Guard: the fixture really does block browser (config contract).
        assert "browser" in mp.config.blocked

        result = _find_tools_on(mp, {"select": ["browser"]})

        assert "browser" not in mp.active_tools, (
            "Raw 'browser' must never be activatable on the user channel — it is "
            f"in UserConfig.blocked. active_tools={mp.active_tools}"
        )
        assert "not found or unavailable" in result.lower(), (
            f"Blocked 'browser' select must be reported unavailable. result={result!r}"
        )

    def test_select_search_is_rejected_on_user_channel(self):
        """Raw 'search' is equally delegate-exclusive — blocked on user."""
        mp = _mp_for(UserConfig())
        result = _find_tools_on(mp, {"select": ["search"]})

        assert "search" not in mp.active_tools, (
            f"Raw 'search' must be blocked on the user channel. active_tools={mp.active_tools}"
        )
        assert "not found or unavailable" in result.lower()

    def test_query_cannot_surface_blocked_browser_on_user_channel(self):
        """Even a query that is a perfect semantic match for browser must not
        inject it on the user channel — blocked names are removed from the
        discovery allow-list before the search runs."""
        mp = _mp_for(UserConfig())
        _find_tools_on(mp, {"query": "control a headless browser, click buttons and take a screenshot"})

        assert "browser" not in mp.active_tools, (
            "A blocked tool must never be injected via the query path, regardless "
            f"of relevance. active_tools={mp.active_tools}"
        )

    def test_delegate_tools_remain_available_on_user_channel(self):
        """The fix must not over-block: the delegate tools (web_browse /
        web_search) are how the user channel reaches the web, and stay
        selectable."""
        mp = _mp_for(UserConfig())
        result = _find_tools_on(mp, {"select": ["web_browse", "web_search"]})

        assert "web_browse" in mp.active_tools, (
            f"web_browse delegate must stay available on user. active_tools={mp.active_tools}"
        )
        assert "web_search" in mp.active_tools, (
            f"web_search delegate must stay available on user. active_tools={mp.active_tools}"
        )
        # find_tools now returns a structured body: a successful select reports
        # nothing under not_found (neither delegate was treated as unavailable).
        assert result["not_found"] == [], (
            f"delegate tools must not be reported unavailable. result={result!r}"
        )


# ---------------------------------------------------------------------------
# Background channel — delegate + raw web + pattern-write tools all blocked.
# ---------------------------------------------------------------------------

class TestBackgroundChannelBlocksDelegates:

    def test_dmn_cannot_select_any_delegate_or_raw_web_or_pattern_tool(self):
        """DMN is a background reflection loop: it must never spawn delegate
        work, reach the web directly, or write patterns."""
        mp = _mp_for(DmnConfig())
        blocked_names = ["browser", "search", "web_browse", "web_search", "save_pattern", "save_graph"]
        # Guard: the config really blocks all of these.
        assert set(blocked_names) <= set(mp.config.blocked)

        result = _find_tools_on(mp, {"select": blocked_names})

        leaked = [n for n in blocked_names if n in mp.active_tools]
        assert not leaked, f"DMN must block all of {blocked_names}; leaked: {leaked}"
        assert "not found or unavailable" in result.lower()


# ---------------------------------------------------------------------------
# Positive control — browser IS delivered on the web_browse delegate channel.
# ---------------------------------------------------------------------------

class TestWebBrowseDelegateHasBrowser:

    def test_browser_resolves_into_the_tool_schemas_on_web_browse_channel(self):
        """WebBrowseConfig pins browser as always-available, so build_tools
        resolves a real 'browser' tool schema for the delegate — the channel
        raw browser is exclusive to."""
        mp = _mp_for(WebBrowseConfig(ProcessorConfig.POLICY_CHANNEL.CHAT))
        assert "browser" in mp.config.always_available

        schemas = AbilityRegistry.build_tools(mp)
        names = {s["name"] for s in schemas}

        assert "browser" in names, (
            f"web_browse delegate must be handed the real browser tool. names={sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Contract guard — the two raw web tools are the ones declared delegate-only.
# ---------------------------------------------------------------------------

def test_delegate_internal_tools_are_browser_and_search():
    """Pin the set this whole isolation rests on so a silent edit to the
    constant trips a test."""
    assert DELEGATE_INTERNAL_TOOLS == frozenset({"browser", "search"})
