"""Feature tests: the rebuilt 9-verb browser ability.

Drives the REAL production hot path — ``ToolDispatcher.dispatch()`` on a real
``MessageProcessor`` bound to ``WebBrowseConfig`` (where ``browser`` is
always-available; ``browser`` is in ``PolicyManager.INTERNAL`` so no policy rows
are needed) — with zero mocks. The SSRF guard blocks every local/private address
(security.py) WITHOUT a network call, so anything requiring real navigation
lives in tests/e2e/test_browser_live.py (marker: e2e) and the end-to-end
scenarios; these tests cover the full no-network surface: schema, the SSRF guard
itself, and the no-open-page guard, all asserted on the dispatcher WIRE envelope.

The canonical-result-code contract (kebab codes + hint + the unknown-action /
missing-params pre-gate) is locked in tests/test_ability_browser_tool_result.py.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.browser import BrowserAbility
from configs.channels.web_browse import WebBrowseConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

_VERBS = ["open", "read", "find", "click", "fill", "select", "scroll", "back", "screenshot"]


def _browse_mp() -> MessageProcessor:
    mp = MessageProcessor("drive a web page")
    mp.config = WebBrowseConfig(ProcessorConfig.PolicyChannel.CHAT)
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _dispatch(params: dict) -> str:
    """The exact rendered string the ACT loop consumes for one browser call."""
    return ToolDispatcher(_browse_mp()).dispatch("browser", params)


def test_schema_is_nine_flat_verbs():
    schema = BrowserAbility(mp=None).get_input_schema()
    params = schema["input_schema"]
    assert params["properties"]["action"]["enum"] == _VERBS
    # 7 model-facing params + the framework act_summary — nothing else.
    assert set(params["properties"]) == {
        "action", "url", "target", "value", "query", "section", "direction", "act_summary",
    }
    assert params["required"] == ["action", "act_summary"]


def test_ssrf_guard_blocks_private_urls_before_any_browser_work():
    rendered = _dispatch({"action": "open", "url": "http://127.0.0.1:9/admin"})
    assert rendered.startswith("[browser(status=error, code=url-blocked"), rendered
    assert "URL blocked" in rendered, rendered


def test_verbs_demand_an_open_page_first():
    """No session for this key → mechanical guidance, no browser launch."""
    rendered = _dispatch({"action": "click", "target": "Sign in"})
    assert "status=error" in rendered.splitlines()[0], rendered
    assert "code=no-open-page" in rendered.splitlines()[0], rendered
    assert "No page is open" in rendered, rendered
