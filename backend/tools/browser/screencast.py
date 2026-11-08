# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The live view of one tab: a screencast that runs only while someone watches.

Each socket on ``/ws/browser/<tab_id>`` is one :class:`Viewer`. The first
viewer of a tab starts its screencast and the last one to leave stops it; a
viewer's ``size`` sets the largest frame the browser sends. Frames, input and
the history lookup go straight to the browser connection, never through the
pool, so a viewer never waits on a browsing run and a run never waits on a
viewer.

The browser sends a screencast frame only when the page repaints, so a new
viewer of a still page is seeded with a screenshot of it.

:func:`stream` is the socket's one writer: every frame, meta update and the
final ``closed`` go out from its thread, so sends never interleave.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import TYPE_CHECKING

from simple_websocket import ConnectionClosed

from tools.browser.cdp import BrowserError, get, get_objects
from tools.browser.tabs import VIEWPORT_HEIGHT, VIEWPORT_WIDTH, Tab

if TYPE_CHECKING:
    from simple_websocket import Server

logger = logging.getLogger(__name__)

_QUALITY = 60              # JPEG quality of streamed frames
_MIN_SIZE = 16             # px — the smallest frame a viewer may ask for
_HISTORY_TIMEOUT = 5.0     # s — the back/forward lookup behind each meta update
_IDLE_WAIT = 1.0           # s — how often a quiet stream rechecks the browser is still up
_PING_AFTER = 30.0         # s — a stream silent this long pings, so no proxy times the socket out


class Viewer:
    """One socket watching *tab*. Driven by that socket's handler thread."""

    def __init__(self, tab: Tab) -> None:
        self.tab = tab
        self.gone = False
        self._size = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)

    def open(self) -> None:
        """Join the tab's viewers, starting its screencast as the first, and seed the current picture."""
        # Start and stop go out under the lock that decides first in and last out, so a viewer
        # joining as the last one leaves can never have its start overtaken by that stop.
        with self.tab.changed:
            self.tab.viewers += 1
            if self.tab.viewers == 1:
                self._start()
        try:
            shot = self.tab.browser.send("Page.captureScreenshot", {"format": "jpeg", "quality": _QUALITY},
                                         session_id=self.tab.session_id)
        except BrowserError as e:
            logger.warning("[BROWSER] no first frame for tab %s: %s", self.tab.id, e)
            return
        data = shot.get("data")
        if isinstance(data, str):
            with self.tab.changed:
                self.tab.frame = data
                self.tab.frames += 1
                self.tab.changed.notify_all()

    def resize(self, width: int, height: int) -> None:
        """Send frames no larger than *width* x *height*; restarts the screencast only on a new size.

        The browser ignores a start while its screencast runs, so the new size needs a stop first."""
        size = (max(_MIN_SIZE, min(width, VIEWPORT_WIDTH)), max(_MIN_SIZE, min(height, VIEWPORT_HEIGHT)))
        if size == self._size:
            return
        self._size = size
        self.tab.browser.post("Page.stopScreencast", session_id=self.tab.session_id)
        self._start()

    def input(self, method: str, params: dict[str, object]) -> None:
        """Forward one input event to the page without waiting for it."""
        self.tab.browser.post(method, params, session_id=self.tab.session_id)

    def close(self) -> None:
        """Leave; the last viewer out stops the screencast. Ends this viewer's :func:`stream`."""
        with self.tab.changed:
            self.gone = True
            self.tab.viewers -= 1
            if self.tab.viewers == 0 and self.tab.live:
                self.tab.browser.post("Page.stopScreencast", session_id=self.tab.session_id)
            self.tab.changed.notify_all()

    def _start(self) -> None:
        width, height = self._size
        self.tab.browser.post("Page.startScreencast",
                              {"format": "jpeg", "quality": _QUALITY, "maxWidth": width, "maxHeight": height},
                              session_id=self.tab.session_id)


def stream(ws: Server, viewer: Viewer) -> None:
    """Send *viewer*'s tab to *ws* — meta on connect and on change, then each newest frame —
    until the viewer leaves or the tab closes, which sends ``closed`` and closes the socket."""
    tab = viewer.tab
    sent_frames = sent_metas = -1
    meta: dict[str, object] | None = None
    last_sent = time.monotonic()
    try:
        while True:
            with tab.changed:
                tab.changed.wait_for(lambda: viewer.gone or not tab.live or tab.frames != sent_frames
                                     or tab.metas != sent_metas, _IDLE_WAIT)
                if viewer.gone:
                    return
                live, frames, frame, metas = tab.live, tab.frames, tab.frame, tab.metas
            if not live:
                ws.send(json.dumps({"type": "closed"}))
                ws.close()
                return
            if metas != sent_metas:
                sent_metas = metas
                current = _meta(tab)
                if current != meta:
                    meta = current
                    ws.send(json.dumps(meta))
                    last_sent = time.monotonic()
            if frames != sent_frames:
                sent_frames = frames
                if frame:
                    ws.send(base64.b64decode(frame))
                    last_sent = time.monotonic()
            if time.monotonic() - last_sent >= _PING_AFTER:
                ws.send(json.dumps({"type": "ping"}))
                last_sent = time.monotonic()
    except (ConnectionClosed, OSError) as e:
        logger.debug("[BROWSER] stream of tab %s ended: %s", tab.id, e)


def _meta(tab: Tab) -> dict[str, object]:
    back = forward = False
    try:
        history = tab.browser.send("Page.getNavigationHistory", session_id=tab.session_id, timeout=_HISTORY_TIMEOUT)
        index, count = get(history, "currentIndex", int), len(get_objects(history, "entries"))
        back, forward = index > 0, index < count - 1
    except BrowserError as e:
        logger.debug("[BROWSER] no history for tab %s: %s", tab.id, e)
    return {"type": "meta", "width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT, "url": tab.url, "title": tab.title,
            "can_go_back": back, "can_go_forward": forward}
