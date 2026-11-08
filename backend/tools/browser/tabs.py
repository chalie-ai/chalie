# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The browser's open tabs — what the live view lists, streams and steers.

A tab is one page target with an attached session: a browsing run's page
(opener ``chalie``) or a page the user opened, including a popup kept as its
own tab (opener ``user``). The pool thread registers and closes tabs; the CDP
reader thread keeps each tab's url, title, newest screencast frame and
liveness current through :meth:`Tab.observe`; any thread reads them.

Every change to the list — a tab opening or closing, a navigation, a new
title, a hand-off opening or closing — sends one ``browser_tabs_changed``
frame on the main socket, from a worker thread so a slow socket never stalls
the reader or the pool; the same thread looks up a page's title once its
document loads.

A hand-off parks a browsing run on its tab until the user says they are done
(:func:`hand_off`, :func:`finish_handoff`), the way a permission ask parks a
tool call: the run's thread waits, the browser stays up.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from models.ws_message import WsMessage
from services.time_utils import utc_now
from services.websocket import Websocket
from tools.browser.cdp import Browser, BrowserError, Event, get_dict

if TYPE_CHECKING:
    from tools.browser.session import PageSession

logger = logging.getLogger(__name__)

Opener = Literal["chalie", "user"]

VIEWPORT_WIDTH = 1280      # CSS px a new tab renders at, until a viewer sizes it
VIEWPORT_HEIGHT = 720

_GONE = frozenset({"Target.targetDestroyed", "Target.targetCrashed", "Target.detachedFromTarget"})
_LOADED = frozenset({"Page.domContentEventFired", "Page.loadEventFired"})
_PARK_POLL = 0.25          # seconds between stop checks while a hand-off waits
_INFO_TIMEOUT = 5.0        # seconds — the url/title lookup after a document loads


@dataclass(eq=False)
class Handoff:
    """A run handing its tab to the user; ``done`` is set when the user finishes."""

    reason: str
    origin: dict[str, object]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    asked_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    done: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "reason": self.reason, "origin": self.origin, "asked_at": self.asked_at}


class Tab:
    """One open page. ``changed`` is notified on every new frame, meta change or close."""

    def __init__(self, page: PageSession, browser: Browser, target_id: str, session_id: str,
                 opener: Opener) -> None:
        self.page = page
        self.browser = browser
        self.id = target_id
        self.session_id = session_id
        self.opener: Opener = opener
        self.url = ""
        self.title = ""
        self.handoff: Handoff | None = None
        self.viewers = 0
        self.frame = ""            # newest screencast frame, base64 JPEG
        self.frames = 0            # bumped per frame
        self.metas = 0             # bumped per url, title or size change
        self.width, self.height = VIEWPORT_WIDTH, VIEWPORT_HEIGHT  # the page's window, CSS px
        self.scale = 1.0           # device px per CSS px
        self.closed = False
        self.changed = threading.Condition()

    @property
    def live(self) -> bool:
        return not self.closed and self.browser.connected

    def describe(self, url: str, title: str) -> None:
        """Record the page's url and title, announcing a change."""
        with self.changed:
            if (url, title) == (self.url, self.title):
                return
            self.url, self.title = url, title
            self.metas += 1
            self.changed.notify_all()
        announce()

    def resize(self, width: int, height: int, scale: float) -> bool:
        """Make the page's window *width* x *height* CSS px at *scale* device px each, as resizing
        a browser window does: the page lays itself out again. The last size set wins; returns
        whether the size changed."""
        with self.changed:
            if (width, height, scale) == (self.width, self.height, self.scale):
                return False
            # Sent under the lock, so the browser applies sizes in the order they are recorded.
            self.browser.post("Emulation.setDeviceMetricsOverride",
                              {"width": width, "height": height, "deviceScaleFactor": scale, "mobile": False},
                              session_id=self.session_id)
            self.width, self.height, self.scale = width, height, scale
            self.metas += 1
            self.changed.notify_all()
        return True

    def close(self) -> None:
        """Mark the tab gone and drop it from the list; idempotent."""
        with self.changed:
            if self.closed:
                return
            self.closed = True
            self.changed.notify_all()
        _unregister(self)

    def observe(self, event: Event) -> None:
        """Reader thread: ack and keep the newest frame, track url and title, notice the end."""
        params = event.params
        if event.method == "Page.screencastFrame":
            self.browser.post("Page.screencastFrameAck", {"sessionId": params.get("sessionId")},
                              session_id=self.session_id)
            data = params.get("data")
            if isinstance(data, str):
                with self.changed:
                    self.frame = data
                    self.frames += 1
                    self.changed.notify_all()
            return
        info = params.get("targetInfo")
        if event.method == "Target.targetInfoChanged" and isinstance(info, dict) and info.get("targetId") == self.id:
            url, title = info.get("url"), info.get("title")
            self.describe(url if isinstance(url, str) else "", title if isinstance(title, str) else "")
        elif event.method in _LOADED:
            _worker.submit(self._refresh)
        elif event.method in _GONE and (params.get("targetId") == self.id or params.get("sessionId") == self.session_id):
            self.close()

    def _refresh(self) -> None:
        """Re-read the url and title: the browser reports a new url as a target change, but not
        the title the page sets once it parses."""
        if not self.live:
            return
        try:
            info = get_dict(self.browser.send("Target.getTargetInfo", {"targetId": self.id},
                                              timeout=_INFO_TIMEOUT), "targetInfo")
        except BrowserError as e:
            logger.debug("[BROWSER] no target info for tab %s: %s", self.id, e)
            return
        self.describe(str(info.get("url", "")), str(info.get("title", "")))

    def to_dict(self) -> dict[str, object]:
        handoff = self.handoff
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "opener": self.opener,
            "handoff": handoff.to_dict() if handoff is not None and not handoff.done.is_set() else None,
        }


# ── Registry ─────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_tabs: dict[str, Tab] = {}


def register(tab: Tab) -> None:
    """List *tab*, unless it already closed while it was being set up."""
    with _lock:
        if tab.closed:
            return
        _tabs[tab.id] = tab
    announce()


def _unregister(tab: Tab) -> None:
    with _lock:
        if _tabs.get(tab.id) is not tab:
            return
        del _tabs[tab.id]
    announce()


def drop(browser: Browser) -> None:
    """Forget every tab of a browser that has closed: its pages went with it."""
    with _lock:
        gone = [tab for tab in _tabs.values() if tab.browser is browser]
    for tab in gone:
        tab.close()


def get(tab_id: str) -> Tab | None:
    """The live tab *tab_id*, or ``None``."""
    with _lock:
        tab = _tabs.get(tab_id)
    return tab if tab is not None and tab.live else None


def listing() -> list[dict[str, object]]:
    """Every live tab, in the order they opened. Never starts the browser."""
    with _lock:
        tabs = list(_tabs.values())
    return [tab.to_dict() for tab in tabs if tab.live]


def busy() -> bool:
    """Whether the browser must stay up: a tab is watched, the user's, or handed off."""
    with _lock:
        tabs = list(_tabs.values())
    return any(tab.live and (tab.viewers or tab.opener == "user" or tab.handoff is not None) for tab in tabs)


def _find_handoff(handoff_id: str) -> Handoff | None:
    with _lock:
        tabs = list(_tabs.values())
    for tab in tabs:
        handoff = tab.handoff
        if handoff is not None and handoff.id == handoff_id and not handoff.done.is_set():
            return handoff
    return None


# ── Hand-off ─────────────────────────────────────────────────────────────────

def hand_off(tab: Tab, reason: str, origin: dict[str, object], should_stop: Callable[[], bool]) -> bool:
    """Park the caller until the user finishes the hand-off on *tab*.

    True once the user says they are done; False when the run is stopped or
    the tab closes first. The tab lists the hand-off before any frame goes
    out, so a client re-reading the list on the frame finds it.
    """
    handoff = Handoff(reason=reason, origin=origin)
    tab.handoff = handoff
    try:
        Websocket.broadcast(WsMessage(type="browser_handoff", tab=tab.id, handoff=handoff.id, reason=reason,
                                      origin=origin, asked_at=handoff.asked_at))
        announce()
        while not handoff.done.wait(_PARK_POLL):
            if should_stop() or not tab.live:
                return False
        return True
    finally:
        tab.handoff = None
        announce()


def finish_handoff(handoff_id: str) -> bool:
    """The user is done with hand-off *handoff_id*; False when it is unknown or already over."""
    handoff = _find_handoff(handoff_id)
    if handoff is None:
        return False
    handoff.done.set()
    return True


# ── Change frames ────────────────────────────────────────────────────────────

_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-tabs")
_announce_lock = threading.Lock()
_announce_queued = False


def announce() -> None:
    """Queue one ``browser_tabs_changed`` frame; changes made before it goes out share it."""
    global _announce_queued
    with _announce_lock:
        if _announce_queued:
            return
        _announce_queued = True
    _worker.submit(_send_changed)


def _send_changed() -> None:
    global _announce_queued
    with _announce_lock:
        _announce_queued = False
    try:
        Websocket.broadcast(WsMessage(type="browser_tabs_changed"))
    except Exception:
        logger.exception("[BROWSER] browser_tabs_changed broadcast failed")
