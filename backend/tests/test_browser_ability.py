"""Feature tests: the rebuilt 9-verb browser ability (TKT-877).

Drives the REAL ``BrowserAbility.run`` entry point — the same call
``ToolDispatcher._execute`` makes — with zero mocks. The SSRF guard blocks every
local/private address (security.py:21-29), so anything requiring real navigation
lives in tests/e2e/test_browser_live.py (marker: e2e) and the nightly scenarios;
these tests cover the full no-network surface: schema, validation, the SSRF
guard itself, the no-open-page guard, and the uniform error envelope.
"""

import json

import pytest

from abilities.browser import BrowserAbility

pytestmark = pytest.mark.unit

_VERBS = ["open", "read", "find", "click", "fill", "select", "scroll", "back", "screenshot"]


def _envelope(out: dict) -> dict:
    """Every result is {'status', 'result'} with result = the JSON envelope."""
    assert set(out) == {"status", "result"}, out
    env = json.loads(out["result"])
    assert set(env) == {"page", "data", "changed", "error"}, env
    assert set(env["changed"]) == {"navigated", "dialog", "popup", "summary"}, env
    return env


def test_schema_is_nine_flat_verbs():
    schema = BrowserAbility(mp=None).get_input_schema()
    params = schema["input_schema"]
    assert params["properties"]["action"]["enum"] == _VERBS
    # 7 model-facing params + the framework act_summary — nothing else.
    assert set(params["properties"]) == {
        "action", "url", "target", "value", "query", "section", "direction", "act_summary",
    }
    assert params["required"] == ["action", "act_summary"]


def test_unknown_action_is_an_error_envelope():
    env = _envelope(BrowserAbility(mp=None).run({"action": "render"}))
    assert "Unknown action" in env["error"]
    assert "open" in env["error"]  # the error teaches the verb list


@pytest.mark.parametrize("params,needs", [
    ({"action": "open"}, "url"),
    ({"action": "find"}, "query"),
    ({"action": "click"}, "target"),
    ({"action": "fill", "target": "Email"}, "value"),
    ({"action": "select", "value": "Malta"}, "target"),
    ({"action": "scroll"}, "direction"),
])
def test_missing_required_param_is_an_error_envelope(params, needs):
    env = _envelope(BrowserAbility(mp=None).run(params))
    assert needs in env["error"]


def test_ssrf_guard_blocks_private_urls_before_any_browser_work():
    env = _envelope(BrowserAbility(mp=None).run(
        {"action": "open", "url": "http://127.0.0.1:9/admin"}
    ))
    assert "URL blocked" in env["error"]


def test_verbs_demand_an_open_page_first():
    """No session for this key → mechanical guidance, no browser launch."""
    env = _envelope(BrowserAbility(mp=None).run({"action": "click", "target": "Sign in"}))
    assert "No page is open" in env["error"]
    assert "open" in env["error"]
