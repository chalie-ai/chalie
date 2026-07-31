"""The browser layer blocks the ranges its own hand-maintained list used to miss.

History: the SSRF blocked-network ranges lived in two copies — one under the
abilities, one under ``tools/browser``. They drifted: the browser copy silently
lost ``0.0.0.0/8`` and ``100.64.0.0/10``, so a URL the ``read`` ability refused
would sail straight through the headless browser's request interceptor. Both
copies were collapsed into ``services.ssrf``.

What earns a test is the behaviour the drift produced, not the shape of the
refactor that ended it: the browser must actually reject a URL in the recovered
ranges.
"""

import pytest

pytestmark = pytest.mark.unit


def test_browser_rejects_a_url_in_the_recovered_ranges() -> None:
    """``0.0.0.0`` resolves to itself and must be refused at the browser layer.

    This is the exact URL the drifted copy let through.
    """
    from tools.browser import security

    ok, reason = security.validate_url("http://0.0.0.0/")

    assert ok is False, "the browser must refuse a URL in the 0.0.0.0/8 range"
    assert "0.0.0.0" in reason, f"the refusal must name the address. reason={reason!r}"
