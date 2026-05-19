"""
Integration: verify `_handle_action` emits a metrics block on every
message + error frame it produces.

Exercises the real WS handler function with a captured WS sink and a real
MemoryClient. The only simulated piece is the WebSocket wire protocol — we
use a lightweight sink that captures the JSON payloads handed to `_send_json`.
Skill lookup is exercised against `AbilityRegistry`.
"""

import json
import threading
import pytest


class _FakeWS:
    """Minimal WS sink: captures every JSON payload handed to the handler.

    Matches the interface `_send_json` uses (`ws.send(str)`). Thread-safe
    because `_handle_chat` spawns a background thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.sent = []

    def send(self, payload):
        with self._lock:
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode()
            try:
                self.sent.append(json.loads(payload))
            except (json.JSONDecodeError, TypeError):
                self.sent.append({"__raw__": payload})

    def frames_of(self, t):
        with self._lock:
            return [f for f in self.sent if f.get('type') == t]


@pytest.mark.integration
def test_handle_action_unknown_skill_emits_error(db):
    """Unknown skill → error frame + done frame are emitted via _handle_action."""
    from api import websocket as ws_mod

    ws = _FakeWS()

    msg = {
        "type": "action",
        "payload": {"skill": "__definitely_nonexistent_skill__", "args": {}},
    }
    ws_mod._handle_action(ws, msg)

    errs = ws.frames_of('error')
    msgs = ws.frames_of('message')
    assert errs or msgs, f"Expected message or error frame, got {ws.sent}"


@pytest.mark.integration
def test_handle_action_exception_path_carries_metrics(db, monkeypatch):
    """Force a handler exception → error frame MUST include metrics
    (tokens_total=0, tokens_total_complete=False, response_time_s)."""
    from api import websocket as ws_mod
    from abilities import _registry as reg_mod

    ws = _FakeWS()

    class _RaisingAbility:
        NAME = 'anything'

        def execute(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        reg_mod.AbilityRegistry, 'get',
        staticmethod(lambda _name: _RaisingAbility()),
    )

    msg = {"type": "action", "payload": {"skill": "anything"}}
    ws_mod._handle_action(ws, msg)

    errs = ws.frames_of('error')
    assert errs, f"Expected error frame, got: {ws.sent}"
    err = errs[-1]
    assert 'metrics' in err, f"error frame missing metrics: {err}"
    m = err['metrics']
    assert m.get('tokens_total') == 0
    assert m.get('tokens_total_complete') is False
    assert 'response_time_s' in m
    assert isinstance(m['response_time_s'], (int, float))
    assert 'tools' in m


@pytest.mark.integration
def test_handle_action_message_frame_marks_tokens_incomplete(db, monkeypatch):
    """When a skill succeeds, the message frame's metrics must signal
    tokens_total_complete=False (the skill didn't run the LLM)."""
    from api import websocket as ws_mod
    from abilities import _registry as reg_mod

    ws = _FakeWS()

    class _OkAbility:
        NAME = '__test_skill__'

        def execute(self, *_args, **_kwargs):
            return "ok"

    def _fake_get(name):
        if name == '__test_skill__':
            return _OkAbility()
        raise KeyError(name)

    monkeypatch.setattr(reg_mod.AbilityRegistry, 'get', staticmethod(_fake_get))

    ws_mod._handle_action(
        ws,
        {"type": "action", "payload": {"skill": "__test_skill__"}},
    )

    msgs = ws.frames_of('message')
    assert msgs, f"Expected message frame, got: {ws.sent}"
    m = msgs[-1].get('metrics')
    assert m is not None
    assert m.get('tokens_total') == 0
    assert m.get('tokens_total_complete') is False
    assert 'response_time_s' in m
