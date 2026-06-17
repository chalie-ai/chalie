# Requires network + chromium; run with: pytest -m e2e tests/e2e/test_browser_live.py

import json

import pytest

from abilities.browser import BrowserAbility

pytestmark = pytest.mark.e2e


class _Mp:
    """Minimal carrier for the session key — prod reads only `.uid` here."""
    uid = 877_001


def _run(params: dict) -> dict:
    return json.loads(BrowserAbility(mp=_Mp()).run(params)["result"])


def test_full_browse_flow_on_one_persistent_page():
    env = _run({"action": "open", "url": "https://example.com"})
    assert env["error"] is None, env
    assert env["page"]["status"] == 200
    assert "Example Domain" in env["data"]["text"]

    # The SAME page persists — read without re-opening.
    env = _run({"action": "read"})
    assert "Example Domain" in env["data"]["text"]

    env = _run({"action": "find", "query": "More information"})
    assert env["data"]["interactive"], env

    env = _run({"action": "click", "target": "More information"})
    assert env["changed"]["navigated"] is True, env
    assert "iana.org" in env["page"]["url"]
    assert env["data"], "navigation diff must carry the new page's read view"

    env = _run({"action": "back"})
    assert "example.com" in env["page"]["url"]

    from tools.browser.session import close_session
    close_session(_Mp.uid)
