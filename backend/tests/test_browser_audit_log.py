"""Feature tests: the browser's log reads back as what happened, and never says what was typed.

Every thing the browser does — a verb a run calls, a page committing, a hand-off
starting and ending, the panel opening, steering or closing a tab, a person
watching through the live page — is one INFO line that names the tab and the page
it was on. These tests drive a real browsing run and the real panel through the
real app on a real port, against the real headless shell on a throwaway profile
and a loopback site, and read the lines the app actually logged: each one in full,
built from the real tab id, hand-off id and site address. Text a person types
through the panel reaches the page and never the log, at any level.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    Site,
    StreamSocket,
    as_frame,
    cookie_header,
    listed,
    live_server,
    main_socket,
    open_user_tab,
    receive_until,
    run_opens,
    stream_socket,
    wait_for,
)
from tools.browser import session

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 1, "forked": False}
_WAIT = 15.0  # seconds a line gets to be logged before the test fails
_SECRET = "hunter2"  # what the person types; it must never be logged
_PREFIX = "[BROWSER] "


class _Log(logging.Handler):
    """Every record logged anywhere, at every level; a waiting test wakes on each one."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self._heard = threading.Condition()

    def emit(self, record: logging.LogRecord) -> None:
        with self._heard:
            self.records.append(record)
            self._heard.notify_all()

    def lines(self) -> list[str]:
        """The browser's lines so far, in the order they were logged."""
        with self._heard:
            messages = [r.getMessage() for r in self.records]
        return [m for m in messages if m.startswith(_PREFIX)]

    def wait(self, *lines: str) -> None:
        """Wait until every one of *lines* has been logged; fails the test naming what was."""
        wanted = [_PREFIX + line for line in lines]
        with self._heard:
            heard = self._heard.wait_for(
                lambda: all(w in self.lines() for w in wanted), _WAIT
            )
        assert heard, f"never logged {[w for w in wanted if w not in self.lines()]}; logged {self.lines()}"

    def first(self, start: str) -> str:
        """The first line that starts with *start*, once one is logged."""
        want = _PREFIX + start
        with self._heard:
            self._heard.wait_for(
                lambda: any(m.startswith(want) for m in self.lines()), _WAIT
            )
        found = [m for m in self.lines() if m.startswith(want)]
        assert found, f"never logged a line starting {want!r}; logged {self.lines()}"
        return found[0][len(_PREFIX):]


@contextmanager
def _logged() -> Iterator[_Log]:
    """Hear every record the app logs while the block runs, at every level.

    Installed after the app is built, so the app's own logging setup cannot lower
    the level under it; the root logger's level is put back on the way out."""
    log = _Log()
    root = logging.getLogger()
    level = root.level
    root.addHandler(log)
    root.setLevel(logging.DEBUG)
    try:
        yield log
    finally:
        root.removeHandler(log)
        root.setLevel(level)


def _browsing_run(origin: dict[str, object]) -> MessageProcessor:
    run = MessageProcessor(
        WebBrowseConfig(PolicyChannel.CHAT),
        raw_input="browse",
        metadata={"origin": origin},
    )
    run.active_tools = list(run.config.always_available or [])
    return run


def _browse(run: MessageProcessor, **params: str) -> str:
    """Dispatch one browser action; its result header line."""
    return run.dispatch_service.dispatch("browser", dict[str, object](params)).splitlines()[0]


def _send(ws: StreamSocket, frame: dict[str, object]) -> None:
    ws.send(json.dumps(frame))


def _type_enter_and_click(ws: StreamSocket) -> None:
    """One text frame, one Enter (down and up) and one click (down and up), as the panel sends them."""
    _send(ws, {"type": "text", "text": _SECRET})
    for event in ("keyDown", "keyUp"):
        _send(ws, {
            "type": "key", "event": event, "key": "Enter", "code": "Enter",
            "text": "\r" if event == "keyDown" else None, "keyCode": 13, "modifiers": 0,
        })
    for event in ("mousePressed", "mouseReleased"):
        _send(ws, {
            "type": "mouse", "event": event, "x": 100, "y": 100,
            "button": "left", "clickCount": 1, "modifiers": 0,
        })


def _address_bar(client: FlaskClient, tab_id: str, verb: str, body: dict[str, str]) -> None:
    response = client.post(f"/api/browser/{verb}/{tab_id}", json=body)
    assert response.status_code == 204, response.data


def _never_logs_the_secret(log: _Log) -> None:
    said = [r.getMessage() for r in log.records if _SECRET in r.getMessage()]
    carried = [r.getMessage() for r in log.records if _SECRET in repr(r.args)]
    assert not said and not carried, (said, carried)


def test_a_runs_page_handed_to_the_person_and_back_is_logged_line_by_line_without_what_they_typed(
    app: Flask, user: tuple[FlaskClient, str], site: Site
) -> None:
    client, token = user
    run = _browsing_run(_ORIGIN)
    a, login = f"{site.url}/a", f"{site.url}/login"

    with _logged() as log:
        assert "code=no-open-page" in _browse(run, action="read")
        log.wait("verb read tab=- url=- outcome=no-open-page")

        assert "status=success" in _browse(run, action="open", url=a)
        tab = session.run_tab(run.uid or 0)
        assert tab is not None, "the run's page is not open"
        t = tab.id
        log.wait(
            f"verb open tab={t} url={a} outcome=ok",
            f"navigation tab={t} from=about:blank to={a} by=verb open",
        )

        results: list[str] = []
        waiting = threading.Thread(
            target=lambda: results.append(_browse(run, action="handoff", reason="Please sign in")),
            name="handoff-run",
            daemon=True,
        )
        with live_server(app) as host, main_socket(host, token) as ui:
            waiting.start()
            try:
                receive_until(ui, lambda m: (
                    as_frame(m, "browser_tabs_changed") is not None
                    and any(row["handoff"] is not None for row in listed(client))
                ))
                [row] = listed(client)
                h = str(cast("dict[str, object]", row["handoff"])["id"])
                log.wait(f"hand-off started tab={t} url={a} id={h} reason=Please sign in")

                with stream_socket(host, t, cookie_header(token)) as ws:
                    _send(ws, {"type": "size", "width": 900, "height": 700, "scale": 1})
                    receive_until(ws, lambda m: as_frame(m, "meta") is not None)
                    log.wait(f"viewer connected tab={t} url={a}")

                    _type_enter_and_click(ws)

                    _address_bar(client, t, "navigate", {"url": login})
                    assert wait_for(tab, lambda: tab.title == "Signed in"), tab.title
                    log.wait(
                        f"panel navigate tab={t} url={login} outcome=ok",
                        f"navigation tab={t} from={a} to={login} by=address bar",
                    )

                    _address_bar(client, t, "history", {"action": "reload"})
                    log.wait(
                        f"panel history reload tab={t} url={login} outcome=ok",
                        f"navigation tab={t} from={login} to={login} by=history reload",
                    )

                    done = client.post(f"/api/browser/handoffs/{h}")
                    assert done.status_code == 204, done.data
                    waiting.join(timeout=60)
                    assert not waiting.is_alive(), "the run did not carry on after Done"
                    assert results and "status=success" in results[0], results
                    log.wait(
                        f"panel hand-off done id={h} tab={t} outcome=ok",
                        f"hand-off ended tab={t} url={login} id={h} how=done",
                        f"verb handoff tab={t} url={login} outcome=ok",
                    )
                log.wait(f"viewer disconnected tab={t} url={login} clicks=1 keys=1 texts=1")
            finally:
                if waiting.is_alive():  # free the run before the browser goes away
                    session.close_session(run.uid or 0)
                    waiting.join(timeout=30)

        session.close_session(run.uid or 0)
        log.wait(f"tab closed tab={t} url={login} opener=chalie by=run end")

    _never_logs_the_secret(log)


def test_the_persons_own_tab_is_logged_from_opening_through_its_own_reload_to_closing(
    user: tuple[FlaskClient, str], site: Site
) -> None:
    client, _ = user
    once, a = f"{site.url}/once", f"{site.url}/a"

    with _logged() as log:
        tab = open_user_tab(client, once, "Once")
        t = tab.id
        log.wait(
            f"panel opened tab={t} url={once}",
            f"navigation tab={t} from=about:blank to={once} by=panel open",
        )
        # The page reloading itself, with nobody acting, is put down to the page and the
        # reason the browser gives for it.
        reloaded = log.first(f"navigation tab={t} from={once} to={once} by=")
        assert reloaded == f"navigation tab={t} from={once} to={once} by=page (reload)", reloaded

        assert client.delete(f"/api/browser/tabs/{t}").status_code == 204
        log.wait(
            f"panel close tab={t} url={once} outcome=ok",
            f"tab closed tab={t} url={once} opener=user by=panel",
        )

        _, theirs = run_opens(a)
        assert client.delete(f"/api/browser/tabs/{theirs.id}").status_code == 409
        log.wait(f"panel close tab={theirs.id} url={a} outcome=conflict")

        assert client.delete("/api/browser/tabs/no-such-tab").status_code == 404
        log.wait("panel close tab=no-such-tab url=- outcome=not-found")
