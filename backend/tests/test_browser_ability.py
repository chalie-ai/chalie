"""Feature tests: the rebuilt 10-verb browser ability.

Drives the REAL production hot path — ``ToolDispatcher.dispatch()`` on a real
``MessageProcessor`` bound to ``WebBrowseConfig`` (where ``browser`` is
always-available; ``browser`` is in ``PolicyManager.INTERNAL`` so no policy rows
are needed) — with zero mocks. The SSRF guard blocks every local/private address
(security.py) WITHOUT a network call, so anything requiring real navigation
lives in tests/e2e/test_browser_live.py (marker: e2e) and the end-to-end
scenarios; these tests cover the full no-network surface: schema, the SSRF guard
itself, and the no-open-page guard, all asserted on the dispatcher WIRE envelope.

The parameter contract (unknown-action / missing-params / per-verb field
validation) is enforced by ``BrowserParamsBag`` at the dispatch seam and needs
no tests here.
"""

import sqlite3
from typing import cast

import pytest

from abilities.browser import BrowserAbility
from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_VERBS = ["open", "read", "find", "click", "fill", "select", "scroll", "back", "screenshot", "style"]


def _browse_mp() -> MessageProcessor:
    mp = MessageProcessor(WebBrowseConfig(PolicyChannel.CHAT), raw_input="drive a web page")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _dispatch(params: dict[str, object]) -> str:
    return _browse_mp().dispatch_service.dispatch("browser", params)


def test_schema_is_ten_flat_verbs() -> None:
    schema = BrowserAbility(mp=None).get_input_schema()
    params = cast("dict[str, object]", schema["input_schema"])
    assert cast("dict[str, object]", cast("dict[str, object]", params["properties"])["action"])["enum"] == _VERBS
    # 7 model-facing params + the framework act_summary — nothing else.
    assert set(cast("dict[str, object]", params["properties"])) == {
        "action", "url", "target", "value", "query", "section", "direction", "act_summary",
    }
    assert cast("list[str]", params["required"]) == ["action", "act_summary"]


def test_ssrf_guard_blocks_private_urls_before_any_browser_work(db: sqlite3.Connection) -> None:
    rendered = _dispatch({"action": "open", "url": "http://127.0.0.1:9/admin"})
    assert rendered.startswith("[browser(status=error, code=url-blocked"), rendered
    assert "URL blocked" in rendered, rendered


def test_verbs_demand_an_open_page_first(db: sqlite3.Connection) -> None:
    """No session for this key → mechanical guidance, no browser launch."""
    rendered = _dispatch({"action": "click", "target": "Sign in"})
    assert "status=error" in rendered.splitlines()[0], rendered
    assert "code=no-open-page" in rendered.splitlines()[0], rendered
    assert "No page is open" in rendered, rendered
