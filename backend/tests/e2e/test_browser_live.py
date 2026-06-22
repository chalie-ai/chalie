# Requires network + chromium; run with: pytest -m e2e tests/e2e/test_browser_live.py

import json
from typing import cast

import pytest

from abilities.browser import BrowserAbility

pytestmark = pytest.mark.e2e


class _Mp:
    """Minimal carrier for the session key — prod reads only `.uid` here."""
    uid = 877_001


def _run(params: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], json.loads(cast(str, cast(dict[str, object], BrowserAbility(mp=_Mp()).run(params))["result"])))


def test_full_browse_flow_on_one_persistent_page() -> None:
    env = _run({"action": "open", "url": "https://example.com"})
    assert env["error"] is None, env
    assert cast(dict[str, object], env["page"])["status"] == 200
    assert "Example Domain" in cast(str, cast(dict[str, object], env["data"])["text"])

    # The SAME page persists — read without re-opening.
    env = _run({"action": "read"})
    assert "Example Domain" in cast(str, cast(dict[str, object], env["data"])["text"])

    env = _run({"action": "find", "query": "More information"})
    assert cast(dict[str, object], env["data"])["interactive"], env

    env = _run({"action": "click", "target": "More information"})
    assert cast(dict[str, object], env["changed"])["navigated"] is True, env
    assert "iana.org" in cast(str, cast(dict[str, object], env["page"])["url"])
    assert env["data"], "navigation diff must carry the new page's read view"

    env = _run({"action": "back"})
    assert "example.com" in cast(str, cast(dict[str, object], env["page"])["url"])

    # A screenshot self-describes: the shared document ingest runs the page
    # through vision and the description rides back inline as data["vision"] —
    # no separate vision-tool round-trip. The old "use the vision tool" note is
    # gone, proving the result shape was rewired.
    env = _run({"action": "screenshot"})
    assert env["error"] is None, env
    data = cast(dict[str, object], env["data"])
    assert data["doc_id"], data
    assert "note" not in data, data
    vision = cast(str, data["vision"])
    assert vision.strip(), "screenshot must carry the page description inline"
    assert "domain" in vision.lower(), vision

    from tools.browser.session import close_session
    close_session(_Mp.uid)
