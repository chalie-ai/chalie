"""
Browser Interaction — Step-based page interaction engine.

Executes a sequence of declarative steps (click, fill, select, check, wait,
scroll, press) on a Playwright page.  Stops on first failure; returns
partial results so the calling LLM can adjust and retry.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Default timeout per step (ms) — individual steps can override
_DEFAULT_STEP_TIMEOUT = 5000


def execute_steps(page, steps: list[dict]) -> list[dict]:
    """Execute interaction steps on a page.

    Each step is a dict with:
        action (str, required): click|fill|select|check|wait|scroll|press
        selector (str):         CSS selector for the target element
        value (str):            Text/option for fill/select/press
        timeout (int):          Max ms to wait for element (default 5000)

    Returns:
        List of step result dicts: {action, selector, ok, ms, error}
        Execution stops on first failure.
    """
    results = []

    for i, step in enumerate(steps):
        action = (step.get("action") or "").lower()
        selector = step.get("selector", "")
        value = step.get("value", "")
        timeout = step.get("timeout", _DEFAULT_STEP_TIMEOUT)

        if not action:
            results.append({
                "action": action, "selector": selector,
                "ok": False, "ms": 0, "error": f"Step {i}: empty action",
            })
            break

        t0 = time.time()
        try:
            _execute_one(page, action, selector, value, timeout)
            elapsed_ms = int((time.time() - t0) * 1000)
            results.append({
                "action": action, "selector": selector,
                "ok": True, "ms": elapsed_ms, "error": "",
            })
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            error_msg = f"Step {i} ({action} '{selector}'): {str(e)[:200]}"
            logger.debug("[BROWSER INTERACT] %s", error_msg)
            results.append({
                "action": action, "selector": selector,
                "ok": False, "ms": elapsed_ms, "error": error_msg,
            })
            break  # Stop on first failure

    return results


# ── Action handler helpers ──────────────────────────────────────────────────

def _handle_check(page, selector: str, timeout: int) -> None:
    """Wait for a checkbox element and check it if not already checked."""
    el = page.wait_for_selector(selector, timeout=timeout)
    if el and not el.is_checked():
        el.check()


def _handle_scroll(page, selector: str, timeout: int) -> None:
    """Scroll an element into view."""
    el = page.wait_for_selector(selector, timeout=timeout)
    if el:
        el.scroll_into_view_if_needed()


def _handle_press(page, selector: str, value: str, timeout: int) -> None:
    """Press a key on a selector, or on the keyboard when no selector given."""
    if selector:
        page.press(selector, value, timeout=timeout)
    else:
        page.keyboard.press(value)


def _execute_one(page, action: str, selector: str, value: str, timeout: int) -> None:
    """Execute a single interaction step.  Raises on failure."""
    _ACTION_HANDLERS = {
        "click":  lambda: page.click(selector, timeout=timeout),
        "fill":   lambda: page.fill(selector, value, timeout=timeout),
        "select": lambda: page.select_option(selector, value, timeout=timeout),
        "check":  lambda: _handle_check(page, selector, timeout),
        "wait":   lambda: page.wait_for_selector(selector, state="visible", timeout=timeout),
        "scroll": lambda: _handle_scroll(page, selector, timeout),
        "press":  lambda: _handle_press(page, selector, value, timeout),
        "hover":  lambda: page.hover(selector, timeout=timeout),
        "type":   lambda: page.type(selector, value, timeout=timeout),
    }
    handler = _ACTION_HANDLERS.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    handler()
