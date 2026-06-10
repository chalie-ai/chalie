"""Drift-proof guard: there is exactly ONE SSRF blocklist in the codebase.

History: the SSRF blocked-network ranges used to live in two hand-maintained
copies — ``abilities/_ssrf.py`` and ``tools/browser/security.py``. They drifted:
the browser copy silently lost ``0.0.0.0/8`` and ``100.64.0.0/10``, so a URL the
``read`` ability refused (e.g. ``http://0.0.0.0/``) would sail straight through
the headless browser's request interceptor.

TKT-883 collapses both into ``services.ssrf``. These tests lock that in by
**import identity** — every consumer must reference the SAME ``BLOCKED_NETS``
object, not a list that merely happens to be equal today. Two separately
declared lists could be made equal by hand and then drift apart again; one
shared object cannot. If anyone re-introduces a second copy, the identity
assertion fails loudly.
"""

import pytest

pytestmark = pytest.mark.unit


def test_browser_security_imports_the_single_blocklist():
    """tools/browser/security.py must reference services.ssrf.BLOCKED_NETS itself."""
    from services import ssrf
    from tools.browser import security

    # Same object — `is` not `==`. A second declaration would be a different
    # object even if its contents currently match.
    assert security.BLOCKED_NETS is ssrf.BLOCKED_NETS, (
        "tools/browser/security.py must import BLOCKED_NETS from services.ssrf, "
        "not declare its own copy"
    )


def test_browser_security_resolve_is_the_single_implementation():
    """The browser's resolve-and-check must be services.ssrf.resolve_and_check."""
    from services import ssrf
    from tools.browser import security

    assert security.resolve_and_check is ssrf.resolve_and_check, (
        "tools/browser/security.py must reuse services.ssrf.resolve_and_check, "
        "not declare its own resolver against a private blocklist"
    )


def test_read_and_download_share_the_single_guard():
    """read reaches the guard through web_fetch; web_download imports it directly.

    TKT-899 removed read's direct ``is_private_url`` import — it now fetches
    exclusively through ``services.web_fetch``, whose ``_guard`` holds the
    identity. The invariant is unchanged: no second copy anywhere on read's path.
    """
    from abilities import read, web_download
    from services import ssrf, web_fetch

    assert web_fetch.is_private_url is ssrf.is_private_url
    assert not hasattr(read, "is_private_url"), (
        "read must not re-import the guard; it fetches through web_fetch"
    )
    assert web_download.is_private_url is ssrf.is_private_url


def test_no_residual_abilities_ssrf_module():
    """The old abilities/_ssrf.py copy is gone — no zombie second source."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("abilities._ssrf")


def test_union_ranges_present_so_browser_no_longer_drifts():
    """The merged blocklist carries the ranges the browser copy used to miss.

    ``0.0.0.0/8`` and ``100.64.0.0/10`` were absent from the old browser list.
    Because the browser now shares the single source, these ranges are blocked
    everywhere — proven end-to-end through the browser's own validate_url.
    """
    import ipaddress

    from services import ssrf

    nets = ssrf.BLOCKED_NETS
    assert ipaddress.ip_network("0.0.0.0/8") in nets
    assert ipaddress.ip_network("100.64.0.0/10") in nets

    # "0.0.0.0" resolves to itself; the browser layer must now reject it.
    from tools.browser import security

    ok, reason = security.validate_url("http://0.0.0.0/")
    assert ok is False
    assert "0.0.0.0" in reason
