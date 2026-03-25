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


def _execute_one(page, action: str, selector: str, value: str, timeout: int):
    """Execute a single interaction step.  Raises on failure."""
    if action == "click":
        page.click(selector, timeout=timeout)

    elif action == "fill":
        page.fill(selector, value, timeout=timeout)

    elif action == "select":
        page.select_option(selector, value, timeout=timeout)

    elif action == "check":
        el = page.wait_for_selector(selector, timeout=timeout)
        if el:
            if not el.is_checked():
                el.check()

    elif action == "wait":
        page.wait_for_selector(selector, state="visible", timeout=timeout)

    elif action == "scroll":
        el = page.wait_for_selector(selector, timeout=timeout)
        if el:
            el.scroll_into_view_if_needed()

    elif action == "press":
        if selector:
            page.press(selector, value, timeout=timeout)
        else:
            page.keyboard.press(value)

    elif action == "hover":
        page.hover(selector, timeout=timeout)

    elif action == "type":
        # Type character-by-character (for sites that watch keydown events)
        page.type(selector, value, timeout=timeout)

    else:
        raise ValueError(f"Unknown action: {action}")
