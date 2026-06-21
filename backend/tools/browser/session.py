# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PageSession — one persistent browser tab per delegate run.

The model never sees a session id: the invoking delegate's transcript uid IS
the key, so consecutive `browser` calls in one web_browse run land on the same
live page. All Playwright objects live on the BrowserPool thread; the ONLY
entry points for other threads are run_verb() (submits to the pool),
close_session(), and the ledger readers.

Every verb returns the same JSON-able envelope:

    {"page": {"url", "title", "status"},
     "data": <verb-specific or null>,
     "changed": {"navigated", "dialog", "popup", "summary"},
     "error": null | "<message>"}

The `changed` block is a MECHANICAL diff (url change, captured dialog/popup,
body-text delta) — the model is told what happened instead of re-deriving it.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from tools.browser.security import DnsCache, setup_page_security

if TYPE_CHECKING:
    from typing import Protocol

    class _Locator(Protocol):
        def count(self) -> int: ...
        def nth(self, index: int) -> _Locator: ...
        def inner_text(self) -> str: ...
        def click(self, *, timeout: int) -> None: ...
        def fill(self, value: str, *, timeout: int) -> None: ...
        def select_option(self, *, label: str | None = None, value: str | None = None, timeout: int = 0) -> object: ...

    class _Page(Protocol):
        url: str
        def goto(self, url: str, *, wait_until: str) -> object: ...
        def evaluate(self, script: str, arg: object = None) -> object: ...
        def inner_text(self, selector: str) -> str: ...
        def wait_for_load_state(self, state: str, *, timeout: int) -> None: ...
        def is_closed(self) -> bool: ...
        def title(self) -> str: ...
        def go_back(self, *, wait_until: str) -> None: ...
        def screenshot(self, *, type: str) -> bytes: ...
        def set_default_navigation_timeout(self, timeout: int) -> None: ...
        def set_default_timeout(self, timeout: int) -> None: ...
        def on(self, event: str, handler: object) -> None: ...
        def get_by_role(self, role: str, *, name: str) -> _Locator: ...
        def get_by_label(self, text: str) -> _Locator: ...
        def get_by_placeholder(self, text: str) -> _Locator: ...
        def get_by_text(self, text: str) -> _Locator: ...
        def route(self, pattern: str, handler: object) -> None: ...

    class _Context(Protocol):
        def new_page(self) -> _Page: ...
        def close(self) -> None: ...
        browser: _Browser

    class _Browser(Protocol):
        def is_connected(self) -> bool: ...
        def new_context(self, *, java_script_enabled: bool, ignore_https_errors: bool) -> _Context: ...

    class _Dialog(Protocol):
        type: str
        message: str
        def dismiss(self) -> None: ...

    class _Popup(Protocol):
        url: str
        def close(self) -> None: ...

    class _Response(Protocol):
        status: int

logger = logging.getLogger(__name__)

READ_CAP = 6_000          # chars of page text per envelope — beyond this, outline + find
_FIND_LIMIT = 10
_CONTEXT_CHARS = 100      # chars either side of a find() match
_CANDIDATE_LIMIT = 5
_SUMMARY_CHARS = 240
_OUTLINE_LIMIT = 40
_NAV_TIMEOUT_MS = 30_000
_ACTION_TIMEOUT_MS = 5_000
_SETTLE_TIMEOUT_MS = 3_000

# Both dicts are mutated ONLY on the pool thread, except: _LEDGERS is appended
# by record_screenshot() and popped by close_session() on the dispatcher
# thread, and read by the delegate prompt builder — single dict ops, GIL-safe.
_SESSIONS: dict[int, "PageSession"] = {}
_LEDGERS: dict[int, list[tuple[str, str]]] = {}

_WHITESPACE_RE = re.compile(r"\n{3,}")


# ── Cross-thread entry points ────────────────────────────────────────────────

def run_verb(key: int, verb: str, kwargs: dict[str, object]) -> dict[str, object]:
    """Execute one verb on *key*'s session via the pool thread. Returns the envelope."""
    if verb != "open" and key not in _SESSIONS:
        return error_envelope("No page is open. Use action='open' with a url first.")
    from tools.browser.pool import get_pool  # noqa: PLC0415
    return cast("dict[str, object]", get_pool().execute(_dispatch, key, verb, kwargs, timeout=90))


def close_session(key: int) -> None:
    """End-of-run cleanup: drop the screenshot ledger and close the tab (if any)."""
    _LEDGERS.pop(key, None)
    if key in _SESSIONS:
        from tools.browser.pool import get_pool  # noqa: PLC0415
        get_pool().submit(_close_on_thread, key)


def record_screenshot(key: int, [document_id]: str, url: str) -> None:
    """Append one captured screenshot to *key*'s ledger (compaction-immune)."""
    _LEDGERS.setdefault(key, []).append(([document_id], url))


def screenshot_ledger(key: int) -> list[tuple[str, str]]:
    """All ([document_id], url) screenshots captured in *key*'s run, oldest first."""
    return list(_LEDGERS.get(key, ()))


def error_envelope(message: str) -> dict[str, object]:
    """A full envelope carrying only an error — same shape as every success."""
    return {"page": {}, "data": None, "changed": _no_change(), "error": message}


def _no_change() -> dict[str, object]:
    return {"navigated": False, "dialog": None, "popup": None, "summary": ""}


# ── Pool-thread internals ────────────────────────────────────────────────────

def _dispatch(browser: object, key: int, verb: str, kwargs: dict[str, object]) -> dict[str, object]:
    session = _SESSIONS.get(key)
    if session and not session.alive():
        session.close()
        _SESSIONS.pop(key, None)
        session = None
    if verb == "open":
        if session is None:
            session = PageSession(browser)
            _SESSIONS[key] = session
        return session.open(**cast("dict[str, str]", kwargs))
    if session is None:
        return error_envelope("No page is open. Use action='open' with a url first.")
    return cast("dict[str, object]", getattr(session, verb)(**kwargs))


def _close_on_thread(browser: object, key: int) -> None:  # noqa: ARG001 — pool always passes browser
    session = _SESSIONS.pop(key, None)
    if session:
        session.close()


def _trim(value: object, cap: int = READ_CAP) -> object:
    """Recursively cap every string in the envelope — the hard token-bomb guard."""
    if isinstance(value, str) and len(value) > cap:
        return value[:cap] + "…[truncated]"
    if isinstance(value, list):
        return [_trim(v, cap) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, cap) for k, v in value.items()}
    return value


class PageSession:
    """One browser context + page. Constructed and driven ONLY on the pool thread."""

    _CLICK_ROLES = ("button", "link", "tab", "menuitem", "checkbox", "radio")
    _FILL_ROLES = ("textbox", "searchbox", "combobox", "spinbutton")

    def __init__(self, browser: object) -> None:
        self._context = cast("_Browser", browser).new_context(
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        self.page = self._context.new_page()
        setup_page_security(self.page, DnsCache())
        self.page.set_default_navigation_timeout(_NAV_TIMEOUT_MS)
        self.page.set_default_timeout(_ACTION_TIMEOUT_MS)
        self._dialog: str | None = None
        self._popup: str | None = None
        self.page.on("dialog", self._on_dialog)
        self.page.on("popup", self._on_popup)

    def alive(self) -> bool:
        try:
            return not self.page.is_closed() and self._context.browser.is_connected()
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._context.close()
        except Exception:
            pass

    # ── Verbs (each returns an envelope; kwargs match abilities/browser.py) ──

    def open(self, url: str) -> dict[str, object]:
        before = self._observe()
        resp = self.page.goto(url, wait_until="load")
        self._settle()
        return self._envelope(
            data=self._read_view(),
            changed=self._changed(before),
            status=cast("_Response", resp).status if resp else None,
        )

    def read(self, section: str | None = None) -> dict[str, object]:
        return self._envelope(data=self._read_view(section))

    def find(self, query: str) -> dict[str, object]:
        found = cast("dict[str, object]", self.page.evaluate(_FIND_JS, {"q": query, "limit": _FIND_LIMIT, "ctx": _CONTEXT_CHARS}))
        if not found["matches"] and not found["interactive"]:
            found["note"] = f"No matches for {query!r} on this page."
        return self._envelope(data=found)

    def click(self, target: str) -> dict[str, object]:
        locator, err = self._locate(target, self._CLICK_ROLES, text_fallback=True)
        if err:
            return err
        before = self._observe()
        cast("_Locator", locator).click(timeout=_ACTION_TIMEOUT_MS)
        self._settle()
        changed = self._changed(before)
        data = self._read_view() if changed["navigated"] else None
        return self._envelope(data=data, changed=changed)

    def fill(self, target: str, value: str) -> dict[str, object]:
        locator, err = self._locate(target, self._FILL_ROLES, text_fallback=False)
        if err:
            return err
        before = self._observe()
        cast("_Locator", locator).fill(value, timeout=_ACTION_TIMEOUT_MS)
        return self._envelope(changed=self._changed(before))

    def select(self, target: str, value: str) -> dict[str, object]:
        locator, err = self._locate(target, ("combobox", "listbox"), text_fallback=False)
        if err:
            return err
        before = self._observe()
        try:
            cast("_Locator", locator).select_option(label=value, timeout=_ACTION_TIMEOUT_MS)
        except Exception:
            cast("_Locator", locator).select_option(value=value, timeout=_ACTION_TIMEOUT_MS)
        self._settle()
        return self._envelope(changed=self._changed(before))

    def scroll(self, direction: str) -> dict[str, object]:
        sign = -1 if direction == "up" else 1
        position = cast("dict[str, object]", self.page.evaluate(
            "(s) => { window.scrollBy(0, s * window.innerHeight * 0.8);"
            " return {y: Math.round(window.scrollY),"
            " max: Math.round(document.body.scrollHeight - window.innerHeight)}; }",
            sign,
        ))
        position["at_bottom"] = cast(int, position["y"]) >= cast(int, position["max"])
        return self._envelope(data=position)

    def back(self) -> dict[str, object]:
        before = self._observe()
        self.page.go_back(wait_until="load")
        self._settle()
        return self._envelope(data=self._read_view(), changed=self._changed(before))

    def grab(self) -> dict[str, object]:
        """Viewport PNG + page info — NOT an envelope; the ability layer ingests it."""
        return {
            "png": self.page.screenshot(type="png"),
            "page": {"url": self.page.url, "title": self.page.title(), "status": None},
        }

    # ── Mechanical observation / diff ────────────────────────────────────────

    def _observe(self) -> dict[str, object]:
        try:
            return {"url": self.page.url, "chars": len(self._body_text())}
        except Exception:
            return {"url": "", "chars": 0}

    def _changed(self, before: dict[str, object]) -> dict[str, object]:
        after = self._observe()
        navigated = bool(before["url"]) and after["url"] != before["url"]
        if navigated:
            summary = f"Navigated to {after['url']} — {self.page.title()}"
        elif self._dialog:
            summary = f"A dialog appeared: {self._dialog}"
        elif self._popup:
            summary = f"A popup opened ({self._popup}) and was closed."
        else:
            delta = cast(int, after["chars"]) - cast(int, before["chars"])
            summary = (
                "Page content unchanged."
                if abs(delta) < 50
                else f"Page content changed ({delta:+d} characters)."
            )
        changed: dict[str, object] = {
            "navigated": navigated,
            "dialog": self._dialog,
            "popup": self._popup,
            "summary": summary[:_SUMMARY_CHARS],
        }
        self._dialog = None
        self._popup = None
        return changed

    def _on_dialog(self, dialog: object) -> None:
        self._dialog = f"{cast('_Dialog', dialog).type}: {cast('_Dialog', dialog).message}"[:_SUMMARY_CHARS]
        try:
            cast("_Dialog", dialog).dismiss()
        except Exception:
            pass

    def _on_popup(self, popup: object) -> None:
        try:
            self._popup = cast("_Popup", popup).url
            cast("_Popup", popup).close()
        except Exception:
            pass

    # ── Reading ───────────────────────────────────────────────────────────────

    def _read_view(self, section: str | None = None) -> dict[str, object]:
        if section:
            text = self.page.evaluate(_SECTION_JS, section)
            if text is None:
                return {
                    "outline": self._outline(),
                    "note": f"No heading matching {section!r}. Pick one from the outline.",
                }
            return {"text": _WHITESPACE_RE.sub("\n\n", cast(str, text).strip())[:READ_CAP]}
        text = self._body_text()
        if len(text) > READ_CAP:
            return {
                "too_long": True,
                "chars": len(text),
                "outline": self._outline(),
                "note": (
                    "Page text is too large to return at once. Use action='find' "
                    "with a query to locate what you need, or action='read' with "
                    "section=<a heading from the outline>."
                ),
            }
        return {"text": text}

    def _body_text(self) -> str:
        return _WHITESPACE_RE.sub("\n\n", self.page.inner_text("body").strip())

    def _outline(self) -> list[dict[str, object]]:
        return cast("list[dict[str, object]]", self.page.evaluate(_OUTLINE_JS, _OUTLINE_LIMIT))

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=_SETTLE_TIMEOUT_MS)
        except Exception:
            pass  # busy pages never go idle — the bounded wait is the point

    # ── Visible-text targeting (no CSS, ever) ─────────────────────────────────

    def _locate(
        self, target: str, roles: tuple[str, ...], *, text_fallback: bool
    ) -> tuple[object, None] | tuple[None, dict[str, object]]:
        """Resolve visible text to EXACTLY one element.

        Returns (locator, None) on success, (None, error-envelope) otherwise.
        Strategy order: role+accessible-name → label → placeholder → visible text.
        """
        strategies: list[object] = [self.page.get_by_role(role, name=target) for role in roles]
        strategies += [self.page.get_by_label(target), self.page.get_by_placeholder(target)]
        if text_fallback:
            strategies.append(self.page.get_by_text(target))
        for locator in strategies:
            count = cast("_Locator", locator).count()
            if count == 1:
                return locator, None
            if count > 1:
                names: list[object] = []
                for i in range(min(count, _CANDIDATE_LIMIT)):
                    try:
                        names.append(cast("_Locator", locator).nth(i).inner_text()[:80].strip() or f"<unnamed #{i}>")
                    except Exception:
                        names.append(f"<unreadable #{i}>")
                return None, self._envelope(
                    data={"candidates": names},
                    error=(
                        f"{target!r} matches {count} elements — say which one, "
                        f"e.g. by its fuller visible text."
                    ),
                )
        return None, self._envelope(
            error=(
                f"No element matching {target!r}. Use action='find' with "
                f"query={target!r} to see what is on the page."
            )
        )

    # ── Envelope assembly ─────────────────────────────────────────────────────

    def _envelope(self, data: object = None, changed: dict[str, object] | None = None,
                  status: int | None = None, error: str | None = None) -> dict[str, object]:
        try:
            page: object = {"url": self.page.url, "title": self.page.title(), "status": status}
        except Exception:
            page = {}
        return cast("dict[str, object]", _trim({
            "page": page,
            "data": data,
            "changed": changed or _no_change(),
            "error": error,
        }))


_OUTLINE_JS = """(limit) => {
  return [...document.querySelectorAll('h1,h2,h3')].slice(0, limit).map(h => ({
    level: +h.tagName[1],
    text: h.innerText.trim().slice(0, 120),
  })).filter(h => h.text);
}"""

_SECTION_JS = """(name) => {
  const hs = [...document.querySelectorAll('h1,h2,h3,h4')];
  const i = hs.findIndex(h => h.innerText.trim().toLowerCase().includes(name.toLowerCase()));
  if (i === -1) return null;
  const start = hs[i], stopLevel = +start.tagName[1];
  const out = [start.innerText];
  let node = start;
  while ((node = node.nextElementSibling)) {
    if (/^H[1-6]$/.test(node.tagName) && +node.tagName[1] <= stopLevel) break;
    out.push(node.innerText || '');
  }
  return out.join('\\n');
}"""

_FIND_JS = """({q, limit, ctx}) => {
  const ql = q.toLowerCase();
  const body = document.body.innerText;
  const lower = body.toLowerCase();
  const matches = [];
  let idx = 0;
  while (matches.length < limit && (idx = lower.indexOf(ql, idx)) !== -1) {
    matches.push(body.slice(Math.max(0, idx - ctx), idx + q.length + ctx)
      .replace(/\\s+/g, ' ').trim());
    idx += q.length;
  }
  const interactive = [];
  for (const el of document.querySelectorAll('a,button,input,select,textarea,[role="button"]')) {
    const label = (el.innerText || el.value || el.placeholder ||
                   el.getAttribute('aria-label') || '').trim();
    if (label && label.toLowerCase().includes(ql)) {
      interactive.push({tag: el.tagName.toLowerCase(), text: label.slice(0, 80)});
      if (interactive.length >= limit) break;
    }
  }
  return {matches, interactive};
}"""
