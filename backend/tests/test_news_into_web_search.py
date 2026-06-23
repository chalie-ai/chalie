"""Feature tests — `news` is exclusive to the `web_search` delegate.

The web-research tool surface moves `news` to behave exactly like `search`: a
raw tool the user/background channels can never call directly, available ONLY
inside the `web_search` delegate. The same change pins the delegate's thinking
to LOW so a user's global thinking override never leaks a thinking turn into the
focused search loop.

These run against the REAL production stack — no mocks, the configs under test
are the ones production runs:
  * Real ``FindToolsAbility.run()`` — the same call ToolDispatcher makes — over a
    real ``UserConfig`` MessageProcessor, proving the user channel cannot activate
    `news` (mirrors the existing browser/search isolation guarantee).
  * Real ``AbilityRegistry.build_tools()`` over a real ``WebSearchConfig`` MP,
    proving the delegate's per-turn tool surface actually exposes `news`.
  * The REAL ``resolve_thinking_mode`` (the exact function ``providers.send``
    uses) reading the REAL ``WebSearchConfig`` thinking pin, proving a persisted
    `high` override cannot elevate the delegate above LOW.
"""

from typing import cast

import pytest

from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from configs.channels import UserConfig
from configs.channels.web_search import WebSearchConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.providers import resolve_thinking_mode

pytestmark = pytest.mark.unit


def _mp_for(config: ProcessorConfig) -> MessageProcessor:
    mp = MessageProcessor("what's in the news today")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict[str, object]) -> str:
    ability = FindToolsAbility()
    ability.mp = mp
    return cast(str, ability.run(params).body)


# ── news is no longer reachable from the user channel ────────────────────────

class TestNewsBlockedOnUserChannel:

    def test_news_is_never_injected_on_user_channel(self) -> None:
        """A query naming ``news`` on the user channel must NEVER inject the raw
        news tool — it is ``DISCOVERABLE=False``, exactly like browser/search, so it
        is absent from the global discovery roster and the index the cascade
        searches; it can only be reached inside the web_search delegate."""
        mp = _mp_for(UserConfig())

        _find_tools_on(mp, {"query": ["news"]})

        assert "news" not in mp.active_tools, (
            "Raw 'news' must never be activatable on the user channel — it is "
            f"DISCOVERABLE=False. active_tools={mp.active_tools}"
        )


# ── news IS callable inside the web_search delegate ──────────────────────────

class TestNewsAvailableInWebSearchDelegate:

    def test_web_search_delegate_tool_surface_includes_news(self) -> None:
        """The web_search delegate's resolved per-turn tool schemas must include
        `news` — proving the delegate can actually dispatch it, alongside
        search/read/web_download."""
        mp = _mp_for(WebSearchConfig(ProcessorConfig.PolicyChannel.CHAT))
        names = {cast(str, t["name"]) for t in AbilityRegistry.build_tools(mp)}
        assert "news" in names, (
            f"web_search delegate must expose the news tool. tools={sorted(names)}"
        )


# ── the web_search delegate never thinks ─────────────────────────────────────

class TestWebSearchDelegatePinsThinkingLow:

    def test_high_override_cannot_elevate_delegate_above_low(self) -> None:
        """A persisted user `high` thinking override must NOT reach the web_search
        delegate: the config pin wins in the exact precedence function send()
        uses. LOW maps to "no thinking flag" at the provider, so the delegate
        never spends a thinking turn."""
        cfg = WebSearchConfig(ProcessorConfig.PolicyChannel.CHAT)
        config_pin = getattr(cfg, "thinking_mode", None)

        # resolve_thinking_mode(config_pin, override, gate_level) — the real fn.
        assert resolve_thinking_mode(config_pin, "high", "low") == "low", (
            "WebSearchConfig must pin thinking to LOW so a user's high override "
            f"cannot elevate the delegate. config_pin={config_pin!r}"
        )
