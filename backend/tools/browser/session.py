# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PageSession — one browser tab: a delegate run's page, or one the user opened.

The model never sees a session id: the invoking delegate's transcript uid IS
the key, so consecutive `browser` calls in one web_browse run land on the same
live page. The user's own tabs are keyed by target id in the tab registry
(``tools/browser/tabs.py``), which also lists the run tabs for the live view.
Tabs and their CDP sessions live on the BrowserPool thread; the ONLY entry
points for other threads are the functions under "Cross-thread entry points".

Every verb returns the same JSON-able envelope:

    {"page": {"url", "title", "status"},
     "data": <verb-specific or null>,
     "changed": {"navigated", "dialog", "popup", "summary"},
     "error": null | "<message>"}

The `changed` block is a MECHANICAL diff (url change, captured dialog/popup,
body-text delta) — the model is told what happened instead of re-deriving it.
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import re
import time
from collections.abc import Callable
from functools import partial
from typing import cast

from tools.browser import tabs
from tools.browser.cdp import CLOSED, COMMAND_TIMEOUT, Browser, BrowserError, Event, get, get_dict, get_objects
from tools.browser.tabs import Opener, Tab

logger = logging.getLogger(__name__)

READ_CAP = 6_000          # chars of page text per envelope — beyond this, outline + find
_FIND_LIMIT = 10
_CONTEXT_CHARS = 100      # chars either side of a find() match
_CANDIDATE_LIMIT = 5
_SUMMARY_CHARS = 240
_OUTLINE_LIMIT = 40
_NAV_TIMEOUT = 30.0
_ACTION_TIMEOUT = 5.0
_SETTLE_TIMEOUT = 3.0
_RETRY_PAUSES = (0.0, 0.02, 0.1, 0.1, 0.5)   # between action attempts; the last repeats
_GROUP = "chalie-target"  # Runtime object group holding the element handles of the current action

_POOL_TIMEOUT = 30.0      # s — a tab operation from the live view waits on the pool at most this long

_SESSIONS: dict[int, "PageSession"] = {}

_WHITESPACE_RE = re.compile(r"\n{3,}")

# Re-resolves the target: (how many elements match, handles of the first few).
_Finder = Callable[[], tuple[int, list[str]]]


# ── Cross-thread entry points ────────────────────────────────────────────────

def run_verb(key: int, verb: str, kwargs: dict[str, object]) -> dict[str, object]:
    """Execute one verb on *key*'s session via the pool thread. Returns the envelope."""
    if verb != "open" and key not in _SESSIONS:
        return error_envelope("No page is open. Use action='open' with a url first.")
    from tools.browser.pool import get_pool  # noqa: PLC0415
    return cast("dict[str, object]", get_pool().execute(_dispatch, key, verb, kwargs, timeout=90))


def close_session(key: int) -> None:
    """End of *key*'s run: close its tab, if it has one. A tab that went with its
    browser is only forgotten, so a run's end never launches a browser."""
    session = _SESSIONS.get(key)
    if session is None:
        return
    if not session.tab.live:
        _SESSIONS.pop(key, None)
        return
    from tools.browser.pool import get_pool  # noqa: PLC0415
    get_pool().submit(_close_on_thread, key)


def run_tab(key: int) -> Tab | None:
    """*key*'s live tab, or ``None`` when its run has no page open."""
    session = _SESSIONS.get(key)
    return session.tab if session is not None and session.tab.live else None


def open_tab(url: str | None) -> dict[str, object]:
    """Open a user tab, on *url* when given; the new tab as the list shows it."""
    from tools.browser.pool import get_pool  # noqa: PLC0415
    return cast("dict[str, object]", get_pool().execute(_open_tab, url, timeout=_POOL_TIMEOUT))


def close_tab(tab_id: str) -> None:
    """Close the tab *tab_id*. Raises LookupError when no such tab is open."""
    _on_tab(_close_tab, tab_id)


def navigate_tab(tab_id: str, url: str) -> None:
    """Point the tab *tab_id* at *url*, without waiting for the page to load. Raises LookupError."""
    _on_tab(_navigate_tab, tab_id, url)


def history_tab(tab_id: str, action: str) -> None:
    """Go ``back``, ``forward`` or ``reload`` in the tab *tab_id*, without waiting for the load. Raises LookupError."""
    _on_tab(_history_tab, tab_id, action)


def _on_tab(fn: Callable[..., None], tab_id: str, *args: str) -> None:
    """Run *fn* on the pool for an open tab — never launching a browser for a tab that is not there."""
    _live_tab(tab_id)
    from tools.browser.pool import get_pool  # noqa: PLC0415
    get_pool().execute(fn, tab_id, *args, timeout=_POOL_TIMEOUT)


def error_envelope(message: str) -> dict[str, object]:
    """A full envelope carrying only an error — same shape as every success."""
    return {"page": {}, "data": None, "changed": _no_change(), "error": message}


def _no_change() -> dict[str, object]:
    return {"navigated": False, "dialog": None, "popup": None, "summary": ""}


# ── Pool-thread internals ────────────────────────────────────────────────────

def _dispatch(browser: Browser, key: int, verb: str, kwargs: dict[str, object]) -> dict[str, object]:
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


def _close_on_thread(browser: Browser, key: int) -> None:  # noqa: ARG001 — pool always passes browser
    session = _SESSIONS.pop(key, None)
    if session:
        session.close()


def _open_tab(browser: Browser, url: str | None) -> dict[str, object]:
    tab = PageSession(browser, opener="user").tab
    if url:
        browser.post("Page.navigate", {"url": url}, session_id=tab.session_id)
    return tab.to_dict()


def _adopt(browser: Browser, target_id: str) -> None:
    """Keep the popup *target_id* as a user tab of its own."""
    if tabs.get(target_id) is not None:
        return
    try:
        PageSession(browser, target=target_id, opener="user")
    except BrowserError as e:
        logger.warning("[BROWSER] could not keep popup %s as a tab: %s", target_id, e)


def _live_tab(tab_id: str) -> Tab:
    tab = tabs.get(tab_id)
    if tab is None:
        raise LookupError(f"No open tab {tab_id!r}")
    return tab


def _close_tab(browser: Browser, tab_id: str) -> None:  # noqa: ARG001 — pool always passes browser
    _live_tab(tab_id).page.close()


def _navigate_tab(browser: Browser, tab_id: str, url: str) -> None:
    browser.post("Page.navigate", {"url": url}, session_id=_live_tab(tab_id).session_id)


def _history_tab(browser: Browser, tab_id: str, action: str) -> None:
    session_id = _live_tab(tab_id).session_id
    if action == "reload":
        browser.post("Page.reload", session_id=session_id)
        return
    history = browser.send("Page.getNavigationHistory", session_id=session_id)
    index = get(history, "currentIndex", int) + (-1 if action == "back" else 1)
    entries = get_objects(history, "entries")
    if 0 <= index < len(entries):
        browser.post("Page.navigateToHistoryEntry", {"entryId": get(entries[index], "id", int)}, session_id=session_id)


def _trim(value: object, cap: int = READ_CAP) -> object:
    """Recursively cap every string in the envelope — the hard token-bomb guard."""
    if isinstance(value, str) and len(value) > cap:
        return value[:cap] + "…[truncated]"
    if isinstance(value, list):
        return [_trim(v, cap) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, cap) for k, v in value.items()}
    return value


def _normalize(text: str) -> str:
    """Whitespace-collapsed, case-folded text — what accessible names are matched on."""
    return " ".join(text.replace("​", "").replace("­", "").split()).lower()


def _ax_name(node: dict[str, object]) -> str:
    name = node.get("name")
    value = name.get("value") if isinstance(name, dict) else None
    return value if isinstance(value, str) else ""


class PageSession:
    """One tab over CDP, in the browser's one persistent profile. Constructed and driven ONLY on the pool thread.

    *target* adopts a page that already exists (a popup); otherwise a blank one opens.
    """

    _CLICK_ROLES = ("button", "link", "tab", "menuitem", "checkbox", "radio")
    _FILL_ROLES = ("textbox", "searchbox", "combobox", "spinbutton")

    def __init__(self, browser: Browser, target: str | None = None, opener: Opener = "chalie") -> None:
        self._browser = browser
        self._url = "about:blank"
        self._loader = ""                  # the main-frame document loading or loaded now
        self._lifecycle: set[str] = set()  # lifecycle events _loader has reached
        self._commits = 0                  # main-frame commits, same-document ones included
        self._requested = False            # a main-frame navigation is scheduled, not yet committed
        self._dialog: str | None = None
        self._popups: dict[str, str] = {}  # popup target id -> its latest url
        self._target = target or get(browser.send("Target.createTarget", {"url": "about:blank"}), "targetId", str)
        try:
            # Only a run's tab reads its events back; the user's tabs are kept current by _listen alone.
            self._session = browser.attach(self._target, queue=opener == "chalie")
        except BrowserError:
            self._close_target()
            raise
        self.tab = Tab(self, browser, self._target, self._session, opener)
        browser.watch(self._session, self._listen)
        try:
            self._send("Page.enable")
            self._send("Page.setLifecycleEventsEnabled", {"enabled": True})
            self._send("Emulation.setDeviceMetricsOverride", {"width": self.tab.width, "height": self.tab.height,
                                                              "deviceScaleFactor": self.tab.scale, "mobile": False})
            self._send("Emulation.setFocusEmulationEnabled", {"enabled": True})
            info = get_dict(browser.send("Target.getTargetInfo", {"targetId": self._target}), "targetInfo")
            self.tab.describe(str(info.get("url", "")), str(info.get("title", "")))
        except BrowserError:
            self.close()
            raise
        tabs.register(self.tab)

    def alive(self) -> bool:
        self._pump()
        return self.tab.live

    def close(self) -> None:
        gone = self.tab.closed
        self.tab.close()
        self._browser.release(self._session)
        if not gone:
            if self.tab.opener == "chalie":
                self._close_popups()
            self._close_target()

    def _close_popups(self) -> None:
        """Close the pages this run's tab opened, directly or through a popup of its own,
        except those kept as tabs — they and what they open are the user's.

        Runs while this tab is still open: the browser forgets a page's opener once it closes."""
        if not self._browser.connected:
            return
        try:
            infos = get_objects(self._browser.send("Target.getTargets"), "targetInfos")
        except BrowserError as e:
            logger.warning("[BROWSER] could not list the run's popups: %s", e)
            return
        children: dict[object, list[str]] = {}
        for info in infos:
            if info.get("type") == "page":
                children.setdefault(info.get("openerId"), []).append(str(info.get("targetId")))
        pending = children.pop(self._target, [])
        while pending:
            popup = pending.pop()
            if tabs.get(popup) is None:
                pending += children.pop(popup, [])
                try:
                    self._browser.send("Target.closeTarget", {"targetId": popup})
                except BrowserError:
                    pass  # it closed itself before we got to it

    def _close_target(self) -> None:
        if self._browser.connected:
            try:
                self._browser.send("Target.closeTarget", {"targetId": self._target})
            except BrowserError as e:
                logger.warning("[BROWSER] could not close the tab: %s", e)

    def _listen(self, event: Event) -> None:
        """Reader thread: keep the tab current, and keep a popup as a tab of its own when it
        opens from a user's tab or from a run's tab the user holds."""
        self.tab.observe(event)
        if self.tab.closed:
            self._browser.release(self._session)
            return
        info = event.params.get("targetInfo")
        if (event.method == "Target.targetCreated" and isinstance(info, dict) and info.get("type") == "page"
                and info.get("openerId") == self._target
                and (self.tab.opener == "user" or self.tab.handoff is not None)):
            from tools.browser.pool import get_pool  # noqa: PLC0415
            get_pool().submit(_adopt, str(info.get("targetId")))

    # ── Verbs (each returns an envelope; kwargs match abilities/browser.py) ──

    def open(self, url: str) -> dict[str, object]:
        before = self._observe()
        prior, start, deadline = self._loader, self._commits, time.monotonic() + _NAV_TIMEOUT
        nav = self._send("Page.navigate", {"url": url}, timeout=_NAV_TIMEOUT)
        if nav.get("isDownload"):
            raise BrowserError("Download is starting")
        if nav.get("errorText"):
            raise BrowserError(f"{nav['errorText']} at {url}")
        loader = nav.get("loaderId")
        if loader:
            # The new document's load — or the load of whichever document replaced it
            # first, as a script redirect does before the page it leaves ever loads.
            loaded = self._wait(lambda: self._loader != prior and "load" in self._lifecycle, deadline)
        else:  # same-document navigation: the commit is all there is
            loaded = self._wait(lambda: self._commits > start, deadline)
        if not loaded:
            raise BrowserError(f"open {url}: Timeout {round(_NAV_TIMEOUT * 1000)}ms exceeded.")
        status = self._evaluate(_STATUS_JS) if loader else None
        self._settle()
        return self._envelope(
            data=self._read_view(),
            changed=self._changed(before),
            status=status if isinstance(status, int) and status > 0 else None,
        )

    def read(self, section: str | None = None) -> dict[str, object]:
        return self._envelope(data=self._read_view(section))

    def find(self, query: str) -> dict[str, object]:
        found = cast("dict[str, object]", self._evaluate(_FIND_JS, {"q": query, "limit": _FIND_LIMIT, "ctx": _CONTEXT_CHARS}))
        if not found["matches"] and not found["interactive"]:
            found["note"] = f"No matches for {query!r} on this page."
        return self._envelope(data=found)

    def click(self, target: str) -> dict[str, object]:
        find = self._locate(target, self._CLICK_ROLES, text_fallback=True)
        if isinstance(find, dict):
            return find
        before = self._observe()
        deadline = time.monotonic() + _ACTION_TIMEOUT
        point = self._act(f"click {target!r}", find, _CLICK_JS)
        self._requested = False
        for kind, button, buttons, count in (("mouseMoved", "none", 0, 0), ("mousePressed", "left", 1, 1),
                                             ("mouseReleased", "left", 0, 1)):
            self._send("Input.dispatchMouseEvent", {"type": kind, "x": point["x"], "y": point["y"],
                                                    "button": button, "buttons": buttons, "clickCount": count})
        self._send("Page.enable")  # a round trip: any navigation the click scheduled is queued by now
        if not self._wait(lambda: not self._requested, deadline):
            raise BrowserError(f"click {target!r}: Timeout {round(_ACTION_TIMEOUT * 1000)}ms exceeded"
                               " — waiting for scheduled navigations to finish")
        self._settle()
        changed = self._changed(before)
        data = self._read_view() if changed["navigated"] else None
        return self._envelope(data=data, changed=changed)

    def fill(self, target: str, value: str) -> dict[str, object]:
        find = self._locate(target, self._FILL_ROLES, text_fallback=False)
        if isinstance(find, dict):
            return find
        before = self._observe()
        typed = self._act(f"fill {target!r}", find, _FILL_JS, value)
        if isinstance(typed["text"], str):  # null: the value was set directly
            self._send("Input.insertText", {"text": typed["text"]})
        return self._envelope(changed=self._changed(before))

    def select(self, target: str, value: str) -> dict[str, object]:
        find = self._locate(target, ("combobox", "listbox"), text_fallback=False)
        if isinstance(find, dict):
            return find
        before = self._observe()
        self._act(f"select {target!r}", find, _SELECT_JS, value)
        self._settle()
        return self._envelope(changed=self._changed(before))

    def scroll(self, direction: str) -> dict[str, object]:
        sign = -1 if direction == "up" else 1
        position = cast("dict[str, object]", self._evaluate(
            "(s) => { window.scrollBy(0, s * window.innerHeight * 0.8);"
            " return {y: Math.round(window.scrollY),"
            " max: Math.round(document.body.scrollHeight - window.innerHeight)}; }",
            sign,
        ))
        position["at_bottom"] = cast(int, position["y"]) >= cast(int, position["max"])
        return self._envelope(data=position)

    def back(self) -> dict[str, object]:
        before = self._observe()
        start = self._commits
        history = self._send("Page.getNavigationHistory")
        index = get(history, "currentIndex", int)
        if index > 0:
            entry = get_objects(history, "entries")[index - 1]
            self._send("Page.navigateToHistoryEntry", {"entryId": get(entry, "id", int)})
            if not self._wait(lambda: self._commits > start and "load" in self._lifecycle,
                              time.monotonic() + _NAV_TIMEOUT):
                raise BrowserError(f"back: Timeout {round(_NAV_TIMEOUT * 1000)}ms exceeded.")
        self._settle()
        return self._envelope(data=self._read_view(), changed=self._changed(before))

    def style(self, target: str) -> dict[str, object]:
        find = self._locate(target, self._CLICK_ROLES, text_fallback=True)
        if isinstance(find, dict):
            return find
        return self._envelope(data=self._act(f"style {target!r}", find, _COMPUTED_STYLE_JS))

    def grab(self) -> dict[str, object]:
        """Viewport PNG at one image px per CSS px, whatever the tab's scale, + page info —
        NOT an envelope; the ability layer ingests it."""
        view = get_dict(self._send("Page.getLayoutMetrics"), "cssVisualViewport")
        clip = {"x": view["pageX"], "y": view["pageY"], "width": view["clientWidth"], "height": view["clientHeight"],
                "scale": 1 / self.tab.scale}
        png = base64.b64decode(get(self._send("Page.captureScreenshot", {"format": "png", "clip": clip}), "data", str))
        self._pump()
        return {"png": png, "page": {"url": self._url, "title": self._title(), "status": None}}

    # ── Page events ──────────────────────────────────────────────────────────

    def _pump(self, timeout: float = 0.0) -> None:
        """Fold the tab's queued events into its state, first waiting up to *timeout* s for one."""
        for method, params, _ in self._browser.events(self._session, timeout):
            main = params.get("frameId") == self._target
            if method == "Page.lifecycleEvent" and main:
                if params.get("name") == "init":
                    self._loader, self._lifecycle = str(params.get("loaderId")), set()
                elif params.get("loaderId") == self._loader:  # never a late event of the last document
                    self._lifecycle.add(str(params.get("name")))
            elif method == "Page.frameNavigated":
                frame = get_dict(params, "frame")
                if frame.get("id") == self._target:
                    if frame.get("loaderId") != self._loader:
                        self._loader, self._lifecycle = str(frame.get("loaderId")), set()
                    self._commit(f"{frame.get('url', '')}{frame.get('urlFragment', '')}")
            elif method == "Page.navigatedWithinDocument" and main:
                self._commit(str(params.get("url")))
            elif method == "Page.frameRequestedNavigation" and main and params.get("disposition") == "currentTab":
                self._requested = True
            elif method in ("Page.frameStoppedLoading", "Page.downloadWillBegin") and main:
                self._requested = False
            elif method == "Page.javascriptDialogOpening":  # cdp.py has already dismissed it
                self._dialog = f"{params.get('type')}: {params.get('message')}"[:_SUMMARY_CHARS]
            elif method in ("Target.targetCreated", "Target.targetInfoChanged"):
                info = get_dict(params, "targetInfo")
                popup = str(info.get("targetId"))
                # Only creation adds: a late change event must not resurrect a popup already closed.
                if info.get("type") == "page" and (method == "Target.targetCreated" or popup in self._popups):
                    self._popups[popup] = str(info.get("url"))

    def _commit(self, url: str) -> None:
        self._url, self._commits, self._requested = url, self._commits + 1, False

    def _wait(self, done: Callable[[], bool], deadline: float) -> bool:
        """Pump events until *done*; False once the monotonic *deadline* passes first."""
        self._pump()
        while not done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not self.tab.live:
                raise BrowserError(CLOSED)
            self._pump(remaining)
        return True

    def _settle(self) -> None:
        """Bounded wait for network idle — busy pages never go idle, and the bound is the point."""
        self._wait(lambda: not self._requested and "networkIdle" in self._lifecycle
                   and "" not in self._popups.values(), time.monotonic() + _SETTLE_TIMEOUT)

    # ── Mechanical observation / diff ────────────────────────────────────────

    def _observe(self) -> dict[str, object]:
        self._pump()
        try:
            return {"url": self._url, "chars": len(self._body_text())}
        except BrowserError:
            return {"url": "", "chars": 0}

    def _changed(self, before: dict[str, object]) -> dict[str, object]:
        after = self._observe()
        # A popup kept as a tab of its own (one the user opened during a hand-off) stays open.
        urls = [url for popup_id, url in self._popups.items() if tabs.get(popup_id) is None]
        if urls:
            self._close_popups()
        self._popups.clear()
        popup = (urls[-1] or "about:blank") if urls else None
        navigated = bool(before["url"]) and after["url"] != before["url"]
        if navigated:
            summary = f"Navigated to {after['url']} — {self._title()}"
        elif self._dialog:
            summary = f"A dialog appeared: {self._dialog}"
        elif popup:
            summary = f"A popup opened ({popup}) and was closed."
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
            "popup": popup,
            "summary": summary[:_SUMMARY_CHARS],
        }
        self._dialog = None
        return changed

    # ── Reading ───────────────────────────────────────────────────────────────

    def _read_view(self, section: str | None = None) -> dict[str, object]:
        if section:
            text = self._evaluate(_SECTION_JS, section)
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
        return _WHITESPACE_RE.sub("\n\n", str(self._evaluate("() => document.body ? document.body.innerText : ''")).strip())

    def _title(self) -> str:
        return str(self._evaluate("() => document.title"))

    def _outline(self) -> list[dict[str, object]]:
        return cast("list[dict[str, object]]", self._evaluate(_OUTLINE_JS, _OUTLINE_LIMIT))

    # ── Visible-text targeting (no CSS, ever) ─────────────────────────────────

    def _locate(self, target: str, roles: tuple[str, ...], *, text_fallback: bool) -> _Finder | dict[str, object]:
        """Resolve visible text to EXACTLY one element.

        Returns a finder that re-resolves it for every action attempt, or an
        error envelope. Strategy order: role+accessible-name → label →
        placeholder → visible text, each a case-insensitive substring match.
        """
        self._send("Runtime.releaseObjectGroup", {"objectGroup": _GROUP})
        kinds = ("label", "placeholder", "text") if text_fallback else ("label", "placeholder")
        strategies: list[_Finder] = [partial(self._by_role, role, target) for role in roles]
        strategies += [partial(self._by_query, kind, target) for kind in kinds]
        for find in strategies:
            count, handles = find()
            if count == 1:
                return find
            if count > 1:
                names: list[object] = []
                for i, handle in enumerate(handles):
                    text = self._call(handle, "(el) => el instanceof HTMLElement ? el.innerText : null")
                    names.append(text[:80].strip() or f"<unnamed #{i}>" if isinstance(text, str) else f"<unreadable #{i}>")
                return self._envelope(
                    data={"candidates": names},
                    error=(
                        f"{target!r} matches {count} elements — say which one, "
                        f"e.g. by its fuller visible text."
                    ),
                )
        return self._envelope(
            error=(
                f"No element matching {target!r}. Use action='find' with "
                f"query={target!r} to see what is on the page."
            )
        )

    def _by_role(self, role: str, target: str) -> tuple[int, list[str]]:
        """Unhidden elements of *role* whose accessible name contains *target*."""
        tree = self._send("Accessibility.queryAXTree", {"objectId": self._handle("document"), "role": role})
        want = _normalize(target)
        hits = list(dict.fromkeys(  # one element can surface as several AX nodes
            node["backendDOMNodeId"] for node in get_objects(tree, "nodes")
            if not node.get("ignored") and isinstance(node.get("backendDOMNodeId"), int)
            and want in _normalize(_ax_name(node))
        ))
        handles = [
            get(get_dict(self._send("DOM.resolveNode", {"backendNodeId": node, "objectGroup": _GROUP}), "object"),
                "objectId", str)
            for node in hits[:_CANDIDATE_LIMIT]
        ]
        return len(hits), handles

    def _by_query(self, kind: str, target: str) -> tuple[int, list[str]]:
        """Elements whose label, placeholder or own visible text contains *target*."""
        found = self._handle(f"({_QUERY_JS})({json.dumps(kind)}, {json.dumps(target)})")
        listing = self._send("Runtime.getProperties", {"objectId": found, "ownProperties": True})
        items = {get(item, "name", str): item for item in get_objects(listing, "result")}
        count = sum(name.isdigit() for name in items)
        handles = [get(get_dict(items[str(i)], "value"), "objectId", str) for i in range(min(count, _CANDIDATE_LIMIT))]
        return count, handles

    def _act(self, what: str, find: _Finder, js: str, *args: object) -> dict[str, object]:
        """Run the action script *js* on the target until it reports success.

        The target is re-resolved and its checks re-run on every attempt until
        the action timeout. *js* gets the element, *args* and the attempt
        number; it returns an object once done, or a string naming what is not
        yet true.
        """
        deadline = time.monotonic() + _ACTION_TIMEOUT
        pauses = itertools.chain(_RETRY_PAUSES, itertools.repeat(_RETRY_PAUSES[-1]))
        for attempt in itertools.count():
            count, handles = find()
            if count > 1:
                raise BrowserError(f"{what}: strict mode violation — the target now matches {count} elements")
            outcome = self._call(handles[0], js, *args, attempt) if count else "waiting for the element"
            if isinstance(outcome, dict):
                return {str(k): v for k, v in outcome.items()}
            pause = next(pauses)
            if time.monotonic() + pause > deadline:
                raise BrowserError(f"{what}: Timeout {round(_ACTION_TIMEOUT * 1000)}ms exceeded — {outcome}")
            time.sleep(pause)
        raise AssertionError("unreachable")  # itertools.count() never ends

    # ── Runtime calls ─────────────────────────────────────────────────────────

    def _send(self, method: str, params: dict[str, object] | None = None,
              timeout: float = COMMAND_TIMEOUT) -> dict[str, object]:
        return self._browser.send(method, params, session_id=self._session, timeout=timeout)

    def _runtime(self, method: str, params: dict[str, object]) -> dict[str, object]:
        """A Runtime call's result object; a script that threw raises BrowserError instead."""
        reply = self._send(method, {"awaitPromise": True, **params})
        details = reply.get("exceptionDetails")
        if isinstance(details, dict):
            exception = details.get("exception")
            text = exception.get("description") if isinstance(exception, dict) else None
            raise BrowserError(str(text or details.get("text")).split("\n")[0])
        return get_dict(reply, "result")

    def _evaluate(self, fn: str, *args: object) -> object:
        """Call the page function *fn* with JSON *args*; its JSON-able result."""
        call = f"({fn})({', '.join(json.dumps(a) for a in args)})"
        return self._runtime("Runtime.evaluate", {"expression": call, "returnByValue": True}).get("value")

    def _call(self, handle: str, fn: str, *args: object) -> object:
        """Call *fn* with the element *handle* and JSON *args*; its JSON-able result."""
        return self._runtime("Runtime.callFunctionOn", {
            "objectId": handle, "functionDeclaration": fn, "returnByValue": True,
            "arguments": [{"objectId": handle}, *({"value": a} for a in args)],
        }).get("value")

    def _handle(self, expression: str) -> str:
        """A handle, in the action's object group, to what *expression* evaluates to."""
        return get(self._runtime("Runtime.evaluate", {"expression": expression, "objectGroup": _GROUP}),
                   "objectId", str)

    # ── Envelope assembly ─────────────────────────────────────────────────────

    def _envelope(self, data: object = None, changed: dict[str, object] | None = None,
                  status: int | None = None, error: str | None = None) -> dict[str, object]:
        self._pump()
        try:
            page: object = {"url": self._url, "title": self._title(), "status": status}
        except BrowserError:
            page = {}
        return cast("dict[str, object]", _trim({
            "page": page,
            "data": data,
            "changed": changed or _no_change(),
            "error": error,
        }))


_STATUS_JS = "() => performance.getEntriesByType('navigation')[0]?.responseStatus"

# Label / placeholder / visible-text lookup, matched the way the locators it
# replaces matched: whitespace-normalized, case-insensitive substring; for text,
# the innermost element whose text matches.
_QUERY_JS = r"""(kind, query) => {
  const norm = (s) => s.replace(/[​­]/g, '').trim().replace(/\s+/g, ' ').toLowerCase();
  const q = norm(query);
  const all = [];
  const walk = (root) => {
    for (const el of root.querySelectorAll('*')) { all.push(el); if (el.shadowRoot) walk(el.shadowRoot); }
  };
  walk(document);
  const skip = (el) => ['SCRIPT', 'NOSCRIPT', 'STYLE'].includes(el.nodeName) || !!document.head?.contains(el);
  const texts = new Map(), hits = new Map();
  const text = (node) => {
    if (texts.has(node)) return texts.get(node);
    let full = '';
    if (node.nodeType === Node.ELEMENT_NODE && skip(node)) full = '';
    else if (node instanceof HTMLInputElement && ['submit', 'button', 'reset'].includes(node.type)) full = node.value;
    else {
      for (let c = node.firstChild; c; c = c.nextSibling) {
        if (c.nodeType === Node.TEXT_NODE) full += c.nodeValue || '';
        else if (c.nodeType === Node.ELEMENT_NODE) full += text(c);
      }
      if (node.shadowRoot) full += text(node.shadowRoot);
    }
    texts.set(node, full);
    return full;
  };
  const has = (node) => {
    if (!hits.has(node)) hits.set(node, norm(text(node)).includes(q));
    return hits.get(node);
  };
  const labels = (el) => {
    const ids = (el.getAttribute('aria-labelledby') || '').split(' ').filter(Boolean);
    const refs = [...new Set(ids.map((id) => el.getRootNode().querySelector('#' + CSS.escape(id))).filter(Boolean))];
    if (refs.length) return refs.map(text);
    const aria = el.getAttribute('aria-label');
    if (aria !== null && aria.trim()) return [aria];
    const control = ['BUTTON', 'METER', 'OUTPUT', 'PROGRESS', 'SELECT', 'TEXTAREA'].includes(el.nodeName)
      || (el.nodeName === 'INPUT' && el.type !== 'hidden');
    return control ? [...(el.labels || [])].map(text) : [];
  };
  if (kind === 'label') return all.filter((el) => labels(el).some((t) => norm(t).includes(q)));
  if (kind === 'placeholder') {
    return all.filter((el) => (el.getAttribute('placeholder') ?? null)?.toLowerCase().includes(query.toLowerCase()));
  }
  const out = [];
  let miss = null;
  for (const el of all) {
    if (miss?.contains(el)) continue;
    if (skip(el) || !has(el)) { miss = el; continue; }
    if (![...el.children].some(has) && !(el.shadowRoot && has(el.shadowRoot))) out.push(el);
  }
  return out;
}"""

# Shared actionability checks; each action script starts with them.
_CHECKS_JS = r"""
  if (!el.isConnected) return 'element is not attached to the DOM';
  const rect = el.getBoundingClientRect();
  if (!el.checkVisibility({visibilityProperty: true}) || !rect.width || !rect.height) return 'element is not visible';
  if (el.matches(':disabled') || el.closest('[aria-disabled="true"]')) return 'element is not enabled';
"""

# Scroll the element in (a different alignment each attempt, so a sticky header
# cannot cover it every time), wait until it holds still for a frame, and hit-test
# the centre of its first box — the point the real mouse click lands on.
_CLICK_JS = r"""async (el, attempt) => {""" + _CHECKS_JS + r"""
  const align = [null, 'end', 'center', 'start'][attempt % 4];
  if (align) el.scrollIntoView({block: align, inline: align, behavior: 'instant'});
  else el.scrollIntoViewIfNeeded();
  const frame = () => new Promise((resolve) => requestAnimationFrame(resolve));
  const where = () => { const r = el.getBoundingClientRect(); return [r.x, r.y, r.width, r.height].join(); };
  await frame();
  const settled = where();
  await frame();
  if (where() !== settled) return 'element is not stable';
  const box = el.getClientRects()[0];  // none once the page has re-rendered the element away
  if (!box) return el.isConnected ? 'element is not visible' : 'element is not attached to the DOM';
  const left = Math.max(box.left, 0), top = Math.max(box.top, 0);
  const right = Math.min(box.right, innerWidth), bottom = Math.min(box.bottom, innerHeight);
  if (right - left < 1 || bottom - top < 1) return 'element is outside of the viewport';
  const x = (left + right) / 2, y = (top + bottom) / 2;
  let hit = document.elementFromPoint(x, y);
  while (hit?.shadowRoot) {
    const inner = hit.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === hit) break;
    hit = inner;
  }
  for (let node = hit; node; node = node.parentElement || node.getRootNode().host) if (node === el) return {x, y};
  return (hit ? hit.outerHTML.split('>')[0].slice(0, 100) + '>' : 'another element') + ' intercepts pointer events';
}"""

# Select the field's content so the text typed next replaces it; the input types
# that cannot be typed into get their value set directly instead.
_FILL_JS = r"""(el, value) => {""" + _CHECKS_JS + r"""
  if (el.readOnly || el.getAttribute('aria-readonly') === 'true') return 'element is not editable';
  if (el instanceof HTMLInputElement) {
    const type = el.type.toLowerCase();
    const direct = ['color', 'date', 'time', 'datetime-local', 'month', 'range', 'week'].includes(type);
    if (!direct && !['', 'email', 'number', 'password', 'search', 'tel', 'text', 'url'].includes(type))
      throw new Error(`Input of type "${type}" cannot be filled`);
    if (type === 'number' && isNaN(Number(value.trim()))) throw new Error('Cannot type text into input[type=number]');
    if (direct) {
      const set = type === 'color' ? value.toLowerCase().trim() : value.trim();
      el.focus();
      el.value = set;
      if (el.value !== set) throw new Error('Malformed value');
      el.dispatchEvent(new Event('input', {bubbles: true, composed: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return {text: null};
    }
    el.select();
  } else if (el instanceof HTMLTextAreaElement) {
    el.selectionStart = 0;
    el.selectionEnd = el.value.length;
  } else if (el.isContentEditable) {
    const range = document.createRange();
    range.selectNodeContents(el);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  } else {
    throw new Error('Element is not an <input>, <textarea> or [contenteditable] element');
  }
  el.focus();
  return {text: value};
}"""

# Pick the option by its label, else by its value.
_SELECT_JS = r"""(el, value) => {""" + _CHECKS_JS + r"""
  if (!(el instanceof HTMLSelectElement)) throw new Error('Element is not a <select> element');
  const norm = (s) => s.replace(/[​­]/g, '').trim().replace(/\s+/g, ' ');
  const options = [...el.options];
  const option = options.find((o) => o.label === value || norm(o.label) === norm(value))
    || options.find((o) => o.value === value);
  if (!option) return `no option ${JSON.stringify(value)} among ${JSON.stringify(options.map((o) => o.label)).slice(0, 200)}`;
  if (option.matches(':disabled')) return 'the option is not enabled';
  el.value = undefined;
  option.selected = true;
  el.dispatchEvent(new Event('input', {bubbles: true, composed: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return {};
}"""

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

_COMPUTED_STYLE_JS = """(el) => {
  const cs = getComputedStyle(el);
  const out = {};
  for (const k of ['color', 'background-color', 'border-color',
                   'font-size', 'font-family', 'font-weight'])
    out[k] = cs.getPropertyValue(k);
  return out;
}"""
