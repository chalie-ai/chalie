"""
Browser Tool — Headless browser for the modern web.

Actions:
  render     — Load URL with full JS rendering, extract clean text
  screenshot — Capture visual state, optionally OCR
  interact   — Fill forms, click buttons, navigate multi-step flows
  monitor    — Track page changes over time (for persistent tasks)

Follows Chalie tool contract: execute(topic, params, config, telemetry) -> dict
All Playwright operations are funneled through the dedicated browser thread
(pool.py) to avoid cross-thread event loop issues.
"""

import base64
import difflib
import hashlib
import logging
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Navigation wait strategies
_WAIT_MAP = {
    "networkidle": "networkidle",
    "domcontentloaded": "domcontentloaded",
    "load": "load",
    "commit": "commit",
}

_DEFAULT_VIEWPORT_W = 1280
_DEFAULT_VIEWPORT_H = 720
_DEFAULT_MAX_CHARS = 8000
_NAV_TIMEOUT = 30000        # 30s per navigation
_MAX_SCREENSHOT_HEIGHT = 16384  # Prevent enormous full-page screenshots


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """Browser tool entry point.

    Args:
        topic: Current conversation topic.
        params: Tool parameters (action, url, steps, etc.).
        config: Tool-specific configuration (unused currently).
        telemetry: Client telemetry (unused currently).

    Returns:
        Result dict with action-specific fields + error + _meta.
    """
    action = (params.get("action") or "").lower().strip()

    if action == "render":
        return _action_render(params)
    elif action == "screenshot":
        return _action_screenshot(params)
    elif action == "interact":
        return _action_interact(params)
    elif action == "monitor":
        return _action_monitor(params)
    else:
        return {"error": f"Unknown action: {action!r}. Use: render, screenshot, interact, monitor", "text": ""}


# =============================================================================
#  RENDER
# =============================================================================

def _action_render(params: dict) -> dict:
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "url parameter required", "text": ""}

    from .security import validate_url
    ok, reason = validate_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}", "text": ""}

    wait_for = _parse_wait(params.get("wait_for"))
    selector = params.get("selector")
    max_chars = _parse_max_chars(params.get("max_chars"))
    extract_mode = (params.get("extract") or "text").lower()

    def _work(browser):
        from .security import DnsCache, setup_page_security
        from .extraction import extract_text, extract_html, extract_links

        vw = params.get("viewport_width", _DEFAULT_VIEWPORT_W)
        vh = params.get("viewport_height", _DEFAULT_VIEWPORT_H)
        dns_cache = DnsCache()

        context = browser.new_context(
            viewport={"width": vw, "height": vh},
            user_agent=_user_agent(),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = context.new_page()
        setup_page_security(page, dns_cache)
        page.set_default_navigation_timeout(_NAV_TIMEOUT)

        t0 = time.time()
        try:
            resp = page.goto(url, wait_until=wait_for)
            _wait_extra(page, params)
            load_ms = int((time.time() - t0) * 1000)

            title = page.title()
            final_url = page.url

            # Extract links BEFORE text — extract_text mutates the DOM
            # (removes noise elements), which would strip links from nav/header
            links = extract_links(page, final_url)

            if extract_mode == "html":
                content = extract_html(page, selector, max_chars)
            else:
                content = extract_text(page, selector, max_chars)

            selector_matched = True
            if selector:
                selector_matched = page.query_selector(selector) is not None

            return {
                "text": content,
                "title": title,
                "url": final_url,
                "links": links,
                "error": "",
                "_meta": {
                    "action": "render",
                    "load_ms": load_ms,
                    "js_rendered": True,
                    "status_code": resp.status if resp else None,
                    "selector_matched": selector_matched,
                },
            }
        finally:
            page.close()
            context.close()

    return _submit(_work)


# =============================================================================
#  SCREENSHOT
# =============================================================================

def _action_screenshot(params: dict) -> dict:
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "url parameter required", "text": ""}

    from .security import validate_url
    ok, reason = validate_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}", "text": ""}

    wait_for = _parse_wait(params.get("wait_for"))
    full_page = bool(params.get("full_page", False))
    do_ocr = bool(params.get("ocr", False))
    selector = params.get("selector")
    vw = params.get("viewport_width", _DEFAULT_VIEWPORT_W)
    vh = params.get("viewport_height", _DEFAULT_VIEWPORT_H)

    def _work(browser):
        from .security import DnsCache, setup_page_security

        dns_cache = DnsCache()
        context = browser.new_context(
            viewport={"width": vw, "height": vh},
            user_agent=_user_agent(),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = context.new_page()
        setup_page_security(page, dns_cache)
        page.set_default_navigation_timeout(_NAV_TIMEOUT)

        t0 = time.time()
        try:
            page.goto(url, wait_until=wait_for)
            _wait_extra(page, params)
            load_ms = int((time.time() - t0) * 1000)

            title = page.title()
            final_url = page.url

            # Take screenshot
            ss_opts = {"type": "png", "full_page": full_page}
            if selector:
                el = page.query_selector(selector)
                if el:
                    img_bytes = el.screenshot(type="png")
                else:
                    img_bytes = page.screenshot(**ss_opts)
            else:
                img_bytes = page.screenshot(**ss_opts)

            screenshot_b64 = base64.b64encode(img_bytes).decode()

            # Optional OCR
            ocr_text = ""
            ocr_ms = 0
            if do_ocr:
                ocr_t0 = time.time()
                ocr_text = _run_ocr(img_bytes)
                ocr_ms = int((time.time() - ocr_t0) * 1000)

            return {
                "screenshot_b64": screenshot_b64,
                "ocr_text": ocr_text,
                "title": title,
                "url": final_url,
                "error": "",
                "_meta": {
                    "action": "screenshot",
                    "load_ms": load_ms,
                    "image_bytes": len(img_bytes),
                    "viewport": f"{vw}x{vh}",
                    "full_page": full_page,
                    "ocr_ms": ocr_ms if do_ocr else None,
                },
            }
        finally:
            page.close()
            context.close()

    return _submit(_work)


# =============================================================================
#  INTERACT
# =============================================================================

def _action_interact(params: dict) -> dict:
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "url parameter required", "text": ""}

    steps = params.get("steps")
    if not steps or not isinstance(steps, list):
        return {"error": "steps parameter required (list of interaction steps)", "text": ""}

    from .security import validate_url
    ok, reason = validate_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}", "text": ""}

    wait_for = _parse_wait(params.get("wait_for"))
    extract_after = (params.get("extract_after") or "text").lower()
    selector = params.get("selector")
    screenshot_after = bool(params.get("screenshot_after", False))
    max_chars = _parse_max_chars(params.get("max_chars"))
    save_session = bool(params.get("save_session", False))
    cred_label = params.get("credential_label")

    def _work(browser):
        from .security import DnsCache, setup_page_security
        from .extraction import extract_text, extract_html
        from .interaction import execute_steps

        vw = params.get("viewport_width", _DEFAULT_VIEWPORT_W)
        vh = params.get("viewport_height", _DEFAULT_VIEWPORT_H)
        dns_cache = DnsCache()

        context = browser.new_context(
            viewport={"width": vw, "height": vh},
            user_agent=_user_agent(),
            java_script_enabled=True,
            ignore_https_errors=True,
        )

        # Inject stored cookies if credential_label provided
        if cred_label:
            _inject_credentials(context, url, cred_label)

        page = context.new_page()
        setup_page_security(page, dns_cache)
        page.set_default_navigation_timeout(_NAV_TIMEOUT)

        t0 = time.time()
        try:
            page.goto(url, wait_until=wait_for)
            _wait_extra(page, params)

            # Execute interaction steps
            step_results = execute_steps(page, steps)
            total_ms = int((time.time() - t0) * 1000)

            steps_completed = sum(1 for s in step_results if s["ok"])
            last_failed = next((s for s in step_results if not s["ok"]), None)

            title = page.title()
            final_url = page.url

            # Extract content after steps
            if extract_after == "html":
                text = extract_html(page, selector, max_chars)
            else:
                text = extract_text(page, selector, max_chars)

            # Optional screenshot after steps
            screenshot_b64 = ""
            if screenshot_after:
                img_bytes = page.screenshot(type="png")
                screenshot_b64 = base64.b64encode(img_bytes).decode()

            # Optional session capture
            if save_session and steps_completed == len(steps):
                _capture_session(context, url, cred_label or "auto")

            result = {
                "text": text,
                "title": title,
                "url": final_url,
                "steps_completed": steps_completed,
                "steps_total": len(steps),
                "error": "",
                "_meta": {
                    "action": "interact",
                    "total_ms": total_ms,
                    "steps": step_results,
                },
            }
            if screenshot_b64:
                result["screenshot_b64"] = screenshot_b64
            if last_failed:
                result["step_error"] = last_failed["error"]

            return result
        finally:
            page.close()
            context.close()

    return _submit(_work)


# =============================================================================
#  MONITOR
# =============================================================================

def _action_monitor(params: dict) -> dict:
    url = (params.get("url") or "").strip()
    if not url:
        return {"error": "url parameter required", "text": ""}

    snapshot_key = (params.get("snapshot_key") or "").strip()
    if not snapshot_key:
        return {"error": "snapshot_key parameter required", "text": ""}

    from .security import validate_url
    ok, reason = validate_url(url)
    if not ok:
        return {"error": f"URL blocked: {reason}", "text": ""}

    wait_for = _parse_wait(params.get("wait_for"))
    selector = params.get("selector")
    max_chars = _parse_max_chars(params.get("max_chars"))

    def _work(browser):
        from .security import DnsCache, setup_page_security
        from .extraction import extract_text

        vw = params.get("viewport_width", _DEFAULT_VIEWPORT_W)
        vh = params.get("viewport_height", _DEFAULT_VIEWPORT_H)
        dns_cache = DnsCache()

        context = browser.new_context(
            viewport={"width": vw, "height": vh},
            user_agent=_user_agent(),
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        page = context.new_page()
        setup_page_security(page, dns_cache)
        page.set_default_navigation_timeout(_NAV_TIMEOUT)

        t0 = time.time()
        try:
            page.goto(url, wait_until=wait_for)
            _wait_extra(page, params)
            load_ms = int((time.time() - t0) * 1000)

            current_text = extract_text(page, selector, max_chars)
            content_hash = hashlib.sha256(current_text.encode()).hexdigest()

            # Load previous snapshot
            snapshot_store = _get_snapshot_store()
            # Account ID 1 — single-user system
            previous = snapshot_store.get(1, snapshot_key) if snapshot_store else None

            first_check = previous is None
            changed = False
            diff_lines = ""
            previous_text = ""
            change_ratio = 0.0

            if previous:
                previous_text = previous["content_text"]
                changed = previous["content_hash"] != content_hash
                if changed:
                    # Compute diff
                    diff = list(difflib.unified_diff(
                        previous_text.splitlines(keepends=True),
                        current_text.splitlines(keepends=True),
                        fromfile="previous",
                        tofile="current",
                        lineterm="",
                        n=1,
                    ))
                    diff_lines = "\n".join(diff[:100])  # Cap at 100 diff lines
                    # Change ratio
                    if previous_text:
                        sm = difflib.SequenceMatcher(None, previous_text, current_text)
                        change_ratio = round(1.0 - sm.ratio(), 3)

            # Save current snapshot
            if snapshot_store:
                snapshot_store.save(1, snapshot_key, url, content_hash, current_text)

            return {
                "changed": changed,
                "first_check": first_check,
                "diff": diff_lines,
                "current_text": current_text,
                "previous_text": previous_text if changed else "",
                "change_ratio": change_ratio,
                "title": page.title(),
                "url": page.url,
                "error": "",
                "_meta": {
                    "action": "monitor",
                    "load_ms": load_ms,
                    "snapshot_key": snapshot_key,
                    "content_hash": content_hash,
                    "chars_current": len(current_text),
                    "chars_previous": len(previous_text) if previous else 0,
                },
            }
        finally:
            page.close()
            context.close()

    return _submit(_work)


# =============================================================================
#  HELPERS
# =============================================================================

def _submit(work_fn) -> dict:
    """Submit work to the browser pool.  Wraps exceptions into error dicts."""
    try:
        from .pool import get_pool
        pool = get_pool()
        return pool.execute(work_fn, timeout=90)
    except TimeoutError:
        return {"error": "Browser operation timed out (90s)", "text": ""}
    except RuntimeError as e:
        if "chromium" in str(e).lower() or "executable" in str(e).lower():
            return {"error": f"Browser unavailable: {e}", "text": ""}
        return {"error": str(e), "text": ""}
    except Exception as e:
        logger.error("[BROWSER] Unexpected error: %s", e, exc_info=True)
        return {"error": f"Browser error: {str(e)[:300]}", "text": ""}


def _parse_wait(wait_for) -> str:
    """Parse wait_for parameter into Playwright wait_until value."""
    if not wait_for:
        return "networkidle"
    wait_for = str(wait_for).lower().strip()
    return _WAIT_MAP.get(wait_for, "networkidle")


def _wait_extra(page, params):
    """Handle custom wait strategies beyond Playwright's built-in wait_until.

    Supports:
      - selector:<css>   — wait for a specific element to appear
      - timeout:<ms>     — additional fixed wait after page load
    """
    wait_for = (params.get("wait_for") or "")
    if wait_for.startswith("selector:"):
        css = wait_for[len("selector:"):]
        if css.strip():
            try:
                page.wait_for_selector(css.strip(), state="visible", timeout=10000)
            except Exception:
                pass
    elif wait_for.startswith("timeout:"):
        try:
            ms = int(wait_for[len("timeout:"):])
            page.wait_for_timeout(min(ms, 15000))
        except (ValueError, TypeError):
            pass


def _parse_max_chars(raw) -> int:
    """Parse max_chars with sane defaults."""
    if raw is None:
        return _DEFAULT_MAX_CHARS
    try:
        return max(100, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CHARS


def _user_agent() -> str:
    """Return a realistic browser user-agent string."""
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )


def _run_ocr(img_bytes: bytes) -> str:
    """Run RapidOCR on screenshot bytes.  Returns extracted text."""
    try:
        import io
        from PIL import Image
        from services.ocr_service import _extract_text
        image = Image.open(io.BytesIO(img_bytes))
        return _extract_text(image)
    except Exception as e:
        logger.warning("[BROWSER] OCR failed: %s", e)
        return ""


def _get_snapshot_store():
    """Return SnapshotStore instance, or None if DB unavailable."""
    try:
        from .credentials import SnapshotStore
        from services.database_service import get_shared_db_service
        return SnapshotStore(get_shared_db_service())
    except Exception:
        return None


def _inject_credentials(context, url: str, label: str):
    """Inject stored cookies for the URL's domain into the browser context."""
    try:
        from .credentials import CredentialVault
        from services.database_service import get_shared_db_service
        domain = urlparse(url).hostname
        if domain:
            vault = CredentialVault(get_shared_db_service())
            vault.inject_cookies(context, 1, domain)
    except Exception as e:
        logger.debug("[BROWSER] Credential injection skipped: %s", e)


def _capture_session(context, url: str, label: str):
    """Capture session cookies from browser context into the vault."""
    try:
        from .credentials import CredentialVault
        from services.database_service import get_shared_db_service
        domain = urlparse(url).hostname
        if domain:
            vault = CredentialVault(get_shared_db_service())
            vault.capture_cookies(context, 1, domain, label)
    except Exception as e:
        logger.debug("[BROWSER] Session capture skipped: %s", e)
