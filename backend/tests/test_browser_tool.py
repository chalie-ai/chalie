"""Tests for the headless browser tool."""

import pytest


# =============================================================================
#  Security
# =============================================================================

class TestUrlValidation:
    """Test URL validation and SSRF protection."""

    @pytest.mark.unit
    def test_valid_https_url(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("https://example.com")
        assert ok is True
        assert reason == ""

    @pytest.mark.unit
    def test_valid_http_url(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://example.com")
        assert ok is True

    @pytest.mark.unit
    def test_blocks_file_scheme(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("file:///etc/passwd")
        assert ok is False
        assert "Blocked scheme" in reason

    @pytest.mark.unit
    def test_blocks_javascript_scheme(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("javascript:alert(1)")
        assert ok is False
        assert "Blocked scheme" in reason

    @pytest.mark.unit
    def test_blocks_data_scheme(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("data:text/html,<h1>hi</h1>")
        assert ok is False
        assert "Blocked scheme" in reason

    @pytest.mark.unit
    def test_blocks_chrome_scheme(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("chrome://settings")
        assert ok is False

    @pytest.mark.unit
    def test_blocks_localhost(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://127.0.0.1")
        assert ok is False
        assert "Blocked IP" in reason

    @pytest.mark.unit
    def test_blocks_private_10(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://10.0.0.1")
        assert ok is False

    @pytest.mark.unit
    def test_blocks_private_172(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://172.16.0.1")
        assert ok is False

    @pytest.mark.unit
    def test_blocks_private_192(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://192.168.1.1")
        assert ok is False

    @pytest.mark.unit
    def test_blocks_link_local(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://169.254.169.254")  # AWS metadata
        assert ok is False

    @pytest.mark.unit
    def test_blocks_empty_hostname(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://")
        assert ok is False

    @pytest.mark.unit
    def test_blocks_unsupported_scheme(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("ftp://example.com")
        assert ok is False
        assert "Blocked scheme" in reason

    @pytest.mark.unit
    def test_blocks_unresolvable_host(self):
        from tools.browser.security import validate_url
        ok, reason = validate_url("http://this-domain-definitely-does-not-exist-12345.com")
        assert ok is False
        assert "DNS" in reason


class TestDnsCache:
    """Test DNS cache behavior."""

    @pytest.mark.unit
    def test_cache_returns_consistent(self):
        from tools.browser.security import DnsCache
        cache = DnsCache(ttl=60)
        ok1, _ = cache.check("example.com")
        ok2, _ = cache.check("example.com")
        assert ok1 == ok2

    @pytest.mark.unit
    def test_cache_blocks_localhost(self):
        from tools.browser.security import DnsCache
        cache = DnsCache()
        ok, reason = cache.check("localhost")
        assert ok is False


# =============================================================================
#  Execute entry point
# =============================================================================

class TestExecuteEntryPoint:
    """Test the execute() function parameter validation."""

    @pytest.mark.unit
    def test_unknown_action(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "invalid"})
        assert result["error"]
        assert "Unknown action" in result["error"]

    @pytest.mark.unit
    def test_missing_action(self):
        from tools.browser.browser import execute
        result = execute("test", {})
        assert result["error"]

    @pytest.mark.unit
    def test_render_missing_url(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "render"})
        assert "url" in result["error"].lower()

    @pytest.mark.unit
    def test_screenshot_missing_url(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "screenshot"})
        assert "url" in result["error"].lower()

    @pytest.mark.unit
    def test_interact_missing_steps(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "interact", "url": "https://example.com"})
        assert "steps" in result["error"].lower()

    @pytest.mark.unit
    def test_monitor_missing_snapshot_key(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "monitor", "url": "https://example.com"})
        assert "snapshot_key" in result["error"].lower()

    @pytest.mark.unit
    def test_render_blocks_ssrf(self):
        from tools.browser.browser import execute
        result = execute("test", {"action": "render", "url": "http://127.0.0.1:8080"})
        assert "blocked" in result["error"].lower()

    @pytest.mark.unit
    def test_interact_blocks_ssrf(self):
        from tools.browser.browser import execute
        result = execute("test", {
            "action": "interact",
            "url": "http://10.0.0.1",
            "steps": [{"action": "click", "selector": "a"}],
        })
        assert "blocked" in result["error"].lower()


# =============================================================================
#  Helper functions
# =============================================================================

class TestHelpers:
    """Test internal helper functions."""

    @pytest.mark.unit
    def test_parse_wait_default(self):
        from tools.browser.browser import _parse_wait
        assert _parse_wait(None) == "networkidle"
        assert _parse_wait("") == "networkidle"

    @pytest.mark.unit
    def test_parse_wait_known(self):
        from tools.browser.browser import _parse_wait
        assert _parse_wait("domcontentloaded") == "domcontentloaded"
        assert _parse_wait("load") == "load"
        assert _parse_wait("commit") == "commit"

    @pytest.mark.unit
    def test_parse_wait_unknown_defaults(self):
        from tools.browser.browser import _parse_wait
        assert _parse_wait("bogus") == "networkidle"

    @pytest.mark.unit
    def test_parse_max_chars_default(self):
        from tools.browser.browser import _parse_max_chars
        assert _parse_max_chars(None) == 8000

    @pytest.mark.unit
    def test_parse_max_chars_valid(self):
        from tools.browser.browser import _parse_max_chars
        assert _parse_max_chars(5000) == 5000
        assert _parse_max_chars("3000") == 3000

    @pytest.mark.unit
    def test_parse_max_chars_minimum(self):
        from tools.browser.browser import _parse_max_chars
        assert _parse_max_chars(10) == 100  # clamped to minimum

    @pytest.mark.unit
    def test_parse_max_chars_invalid(self):
        from tools.browser.browser import _parse_max_chars
        assert _parse_max_chars("not_a_number") == 8000


# =============================================================================
#  Pool
# =============================================================================

class TestPoolStatus:
    """Test browser pool status and lifecycle."""

    @pytest.mark.unit
    def test_pool_singleton(self):
        from tools.browser.pool import get_pool
        pool1 = get_pool()
        pool2 = get_pool()
        assert pool1 is pool2

    @pytest.mark.unit
    def test_pool_status(self):
        from tools.browser.pool import get_pool
        status = get_pool().status()
        assert "thread_alive" in status
        assert "queue_depth" in status
        assert "ops_total" in status


# =============================================================================
#  Tool Library Registration
# =============================================================================

class TestToolRegistration:
    """Test that browser tool is properly registered."""

    @pytest.mark.unit
    def test_browser_in_handlers(self):
        from services.tool_library_service import TOOL_HANDLERS
        assert "browser" in TOOL_HANDLERS

    @pytest.mark.unit
    def test_browser_in_all_names(self):
        from services.tool_library_service import ALL_TOOL_NAMES
        assert "browser" in ALL_TOOL_NAMES

    @pytest.mark.unit
    def test_browser_handler_callable(self):
        from services.tool_library_service import get_handler
        handler = get_handler("browser")
        assert callable(handler)

    @pytest.mark.unit
    def test_browser_metadata_required_fields(self):
        from services.tool_library_service import TOOL_METADATA
        meta = TOOL_METADATA["browser"]
        for field in ("description", "documentation", "input_schema", "returns", "output", "tips"):
            assert field in meta
        assert "action" in meta["input_schema"]["properties"]
        assert "action" in meta["input_schema"]["required"]


# =============================================================================
#  Integration Tests (require Playwright + Chromium)
# =============================================================================

class TestBrowserIntegration:
    """Integration tests that launch a real browser."""

    @pytest.mark.integration
    def test_render_example_com(self):
        from tools.browser.browser import execute
        result = execute("test", {
            "action": "render",
            "url": "https://example.com",
        })
        assert result["error"] == ""
        assert "Example Domain" in result["title"]
        assert len(result["text"]) > 50
        assert result["_meta"]["js_rendered"] is True
        assert result["_meta"]["status_code"] == 200

    @pytest.mark.integration
    def test_screenshot_example_com(self):
        from tools.browser.browser import execute
        result = execute("test", {
            "action": "screenshot",
            "url": "https://example.com",
        })
        assert result["error"] == ""
        assert len(result["screenshot_b64"]) > 1000
        assert result["_meta"]["image_bytes"] > 0

    @pytest.mark.integration
    def test_render_with_selector(self):
        from tools.browser.browser import execute
        result = execute("test", {
            "action": "render",
            "url": "https://example.com",
            "selector": "p",
        })
        assert result["error"] == ""
        assert len(result["text"]) > 10

    @pytest.mark.integration
    def test_interact_click_link(self):
        from tools.browser.browser import execute
        result = execute("test", {
            "action": "interact",
            "url": "https://example.com",
            "steps": [
                {"action": "click", "selector": "a"},
            ],
        })
        assert result["error"] == ""
        assert result["steps_completed"] == 1
        assert result["url"] != "https://example.com/"
