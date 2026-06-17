# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig
from tests._tool_result_harness import MP, allow_policy, body, parse_body, seed_transcript

pytestmark = pytest.mark.unit


# ── A real local Home Assistant stub (zero mocks; a genuine HTTP server) ─────────

# The fixed entity universe the stub serves on /api/states. Real HA shapes:
# each state row carries entity_id, state, and an attributes dict.
_STATES = [
    {
        "entity_id": "light.office",
        "state": "off",
        "attributes": {"friendly_name": "Office Light"},
    },
    {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {"friendly_name": "Living Room Light"},
    },
    {
        "entity_id": "climate.bedroom",
        "state": "heat",
        "attributes": {"friendly_name": "Bedroom Thermostat", "temperature": 21},
    },
    {
        "entity_id": "automation.good_morning",
        "state": "on",
        "attributes": {"friendly_name": "Good Morning"},
    },
]

_STATES_BY_ID = {s["entity_id"]: s for s in _STATES}


def _make_handler(posts: list):
    """Build a real local Home Assistant stub that serves HA endpoints and records
    every service POST so tests can assert calls reached (or didn't reach) the wire."""

    class _HAHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # silence the stub's stderr access log
            pass

        def _send_json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server contract)
            path = self.path
            if path == "/api/" or path == "/api":
                self._send_json(200, {"message": "API running."})
                return
            if path == "/api/states":
                self._send_json(200, _STATES)
                return
            if path.startswith("/api/states/"):
                eid = path[len("/api/states/"):]
                state = _STATES_BY_ID.get(eid)
                if state is None:
                    self._send_json(404, {"message": "Entity not found."})
                    return
                self._send_json(200, state)
                return
            if path == "/api/config/area_registry/list":
                # The handler tolerates a 404 here (area filter is best-effort).
                self._send_json(404, {"message": "Not found."})
                return
            self._send_json(404, {"message": "Not found."})

        def do_POST(self):  # noqa: N802 (http.server contract)
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = {}
            posts.append({"path": self.path, "body": body})
            # HA answers a service call with the list of changed states.
            self._send_json(200, [])

    return _HAHandler


@pytest.fixture()
def ha_server():
    """A real local HA REST stub in a background thread. Yields
    ``{"url", "posts"}`` — *posts* is the live list of every service POST the stub
    received, so a test can assert a control call did or did not fire."""
    posts: list = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(posts))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="ha-stub")
    thread.start()
    host, port = server.server_address
    try:
        yield {"url": f"http://{host}:{port}", "posts": posts}
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _connect_home(url: str):
    """Point the REAL ``HomeCapability`` at *url* and connect it for real.

    Mirrors exactly what ``HomeCapability.connect()`` does once credentials are
    loaded: set the in-memory connection fields, run the genuine
    ``ha_rest_handler.probe`` against the stub, and reflect the probe result in
    ``_connected``. ``is_connected()`` then runs unmodified, reading the flag a
    REAL probe set — no monkeypatch, no fake class. The vault is bypassed by
    seeding the in-memory fields the real ``connect()`` would have populated from
    it, so the test stays offline-from-the-vault and deterministic.

    Returns the live capability instance from the production singleton cache so
    the ability (which calls ``load_capabilities().get('home')``) sees the same
    connected object.
    """
    from capabilities import load_capabilities
    from capabilities.home_capability import ha_rest_handler as rest

    cap = load_capabilities().get("home")
    assert cap is not None, "the home capability must be discovered"
    cap._url = url
    cap._token = "test-token"
    cap._verify_ssl = False
    # Genuine REST probe against the real stub — this is the real connect() body.
    rest.probe(cap._url, cap._token, cap._verify_ssl)
    cap._connected = True
    return cap


@pytest.fixture(autouse=True)
def _reset_home_connection():
    """Each test starts with the home capability disconnected — the singleton
    cache persists across tests, so a connected fixture from a prior test must not
    leak into a not-connected assertion."""
    yield
    from capabilities import load_capabilities

    cap = load_capabilities().get("home")
    if cap is not None:
        cap._connected = False
        cap._url = ""
        cap._token = ""


def _allow(db, action: str, channel: str = "subconscious") -> None:
    """Seed an ``allow`` policy row so the real gate reaches run().

    home.control / home.trigger_automation / home.subscribe_events ship as
    deny/ask; flipping the REAL policy row to ``allow`` (what a user does when they
    pick "always allow") lets the production gate pass so run() — and its entity
    guardrail — actually executes. The production policy table drives the
    production gate; no monkeypatch."""
    allow_policy(db, f"home.{action}", channel)


@pytest.fixture
def dmn_mp(db):
    """A real non-user-broadcast mp on the subconscious channel: the home read
    actions are seeded ``allow`` there, and write actions are flipped to ``allow``
    per-test via ``_allow`` so the real gate reaches run() without a WS ask-hang."""
    return MP(seed_transcript(db, "subconscious", "turn off the office light"), DmnConfig())


def _body_text(rendered: str, tool: str = "home") -> str:
    return body(rendered, tool)


def _parse_body(rendered: str, tool: str = "home") -> object:
    return parse_body(rendered, tool)


# ── 4. connected list_devices → structured dict body with devices + count ───────


def test_list_devices_returns_structured_rows(db, dmn_mp, ha_server):
    """``list_devices`` returns a STRUCTURED JSON dict body (devices rows + count),
    not a double-encoded JSON string. RED-first: the old ability json.dumps'd the
    dict, so ``json.loads`` of the body yielded a ``str``, not a dict."""
    _connect_home(ha_server["url"])

    out = ToolDispatcher(dmn_mp).dispatch("home", {"action": "list_devices"})

    assert "[home(status=success" in out
    body = _parse_body(out)
    assert isinstance(body, dict), f"body must be a structured dict, got {body!r}"
    ids = {d["entity_id"] for d in body["devices"]}
    assert "light.office" in ids
    assert "automation.good_morning" in ids
    assert body["count"] == len(_STATES)


# ── 5. connected get_state on a real entity → state dict body ───────────────────


def test_get_state_on_real_entity_returns_state_dict(db, dmn_mp, ha_server):
    """``get_state`` for an existing entity returns its state dict (entity_id +
    state + attributes) as a structured body."""
    _connect_home(ha_server["url"])

    out = ToolDispatcher(dmn_mp).dispatch(
        "home", {"action": "get_state", "entity_id": "climate.bedroom"}
    )

    assert "[home(status=success" in out
    body = _parse_body(out)
    assert body["entity_id"] == "climate.bedroom"
    assert body["state"] == "heat"
    assert body["attributes"]["temperature"] == 21


# ── 6. connected control on an EXISTING entity → success + the POST fired ────────


def test_control_existing_entity_succeeds_and_posts(db, dmn_mp, ha_server):
    """``control`` on an existing entity reaches the REST service POST and returns a
    structured ``{"status": "ok", …}`` body. The stub recorded exactly one service
    POST against the entity's domain/service."""
    _connect_home(ha_server["url"])
    _allow(db, "control")

    out = ToolDispatcher(dmn_mp).dispatch(
        "home",
        {"action": "control", "entity_id": "light.office", "service": "turn_off"},
    )

    assert "[home(status=success" in out
    body = _parse_body(out)
    assert body["status"] == "ok"
    assert body["entity_id"] == "light.office"
    # The control call really hit the wire.
    posts = ha_server["posts"]
    assert len(posts) == 1
    assert posts[0]["path"] == "/api/services/light/turn_off"
    assert posts[0]["body"]["entity_id"] == "light.office"


# ── 7. connected control on a NONEXISTENT entity → unknown-entity, NO POST ───────


def test_control_nonexistent_entity_blocks_with_candidates_no_post(db, dmn_mp, ha_server):
    """The silent-false-success kill shot: ``control`` on an entity that does NOT
    exist errors ``code=unknown-entity`` with closest-match candidates BEFORE any
    REST write — and the stub received NO service POST. RED-first: the old ability
    fired the POST and rendered ``status=success`` even though HA changed nothing.
    """
    _connect_home(ha_server["url"])
    _allow(db, "control")

    out = ToolDispatcher(dmn_mp).dispatch(
        "home",
        {"action": "control", "entity_id": "light.ofice", "service": "turn_off"},
    )

    assert "[home(status=error, code=unknown-entity" in out
    assert "status=success" not in out
    assert "code=error]" not in out
    # Names the closest real entity so a weak model can self-correct.
    assert "light.office" in out
    # Nothing reached the wire — the no-op write was prevented.
    assert ha_server["posts"] == []


# ── 8. connected get_state on a nonexistent entity → unknown-entity + candidates ─


def test_get_state_nonexistent_entity_returns_unknown_entity(db, dmn_mp, ha_server):
    """``get_state`` on a nonexistent entity errors ``code=unknown-entity`` with a
    closest-match candidate — not the unhandled 404 the raw REST get_state would
    raise."""
    _connect_home(ha_server["url"])

    out = ToolDispatcher(dmn_mp).dispatch(
        "home", {"action": "get_state", "entity_id": "light.livingroom"}
    )

    assert "[home(status=error, code=unknown-entity" in out
    assert "code=error]" not in out
    assert "light.living_room" in out


# ── 9. connected trigger_automation on a missing automation → unknown-automation ─


def test_trigger_nonexistent_automation_returns_unknown_automation_no_post(
    db, dmn_mp, ha_server
):
    """``trigger_automation`` for an automation_id that does not exist errors
    ``code=unknown-automation`` with the real automation named as a candidate, and
    fires NO trigger POST — the same false-success guard, applied to automations."""
    _connect_home(ha_server["url"])
    _allow(db, "trigger_automation")

    out = ToolDispatcher(dmn_mp).dispatch(
        "home",
        {"action": "trigger_automation", "automation_id": "automation.goodmorning"},
    )

    assert "[home(status=error, code=unknown-automation" in out
    assert "status=success" not in out
    assert "automation.good_morning" in out
    assert ha_server["posts"] == []


# ── 10. connected trigger_automation on a real automation → success + POST ───────


def test_trigger_existing_automation_succeeds_and_posts(db, dmn_mp, ha_server):
    """``trigger_automation`` for an existing automation reaches the REST trigger
    service and returns a structured success body; the stub recorded the POST."""
    _connect_home(ha_server["url"])
    _allow(db, "trigger_automation")

    out = ToolDispatcher(dmn_mp).dispatch(
        "home",
        {"action": "trigger_automation", "automation_id": "automation.good_morning"},
    )

    assert "[home(status=success" in out
    body = _parse_body(out)
    assert body["status"] == "ok"
    assert body["automation_id"] == "automation.good_morning"
    posts = ha_server["posts"]
    assert len(posts) == 1
    assert posts[0]["path"] == "/api/services/automation/trigger"


# ── 11. list_automations → structured automations rows ──────────────────────────


def test_list_automations_returns_structured_rows(db, dmn_mp, ha_server):
    """``list_automations`` returns a structured dict body with the automation rows
    and a count — not a double-encoded JSON string."""
    _connect_home(ha_server["url"])

    out = ToolDispatcher(dmn_mp).dispatch("home", {"action": "list_automations"})

    assert "[home(status=success" in out
    body = _parse_body(out)
    assert isinstance(body, dict)
    ids = {a["entity_id"] for a in body["automations"]}
    assert ids == {"automation.good_morning"}
    assert body["count"] == 1
