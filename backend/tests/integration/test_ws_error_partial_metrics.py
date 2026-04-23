"""
Integration: verify `_handle_chat` error path emits `partial_metrics`
on the error frame (spec contract #3).

The handler spawns a background thread that instantiates
UserMessageProcessor.send(). We force that to raise, then assert the
resulting error frame carries a metrics dict containing at minimum
`tokens_total`, `tokens_total_complete=False`, `tools`, and `response_time_s`.
"""

import json
import threading
import time
import pytest


class _FakeWS:
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
def test_handle_chat_error_frame_carries_partial_metrics(db, monkeypatch):
    """Force UserMessageProcessor.send() to raise → error frame must
    include a metrics block snapshot from the partial MetricsAccumulator."""
    from api import websocket as ws_mod
    from services.memory_client import MemoryClientService
    import services.user_message_processor as ump_mod

    store = MemoryClientService.create_connection()
    ws = _FakeWS()

    original_init = ump_mod.UserMessageProcessor.__init__

    def _bad_send(self, *, request_id=None):
        # Touch the accumulator so a snapshot has data in `tools` / timing.
        self._metrics.record_tool('test_tool')
        time.sleep(0.02)
        raise RuntimeError("synthetic send failure")

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Bind our failing send() to THIS instance so we don't pollute the
        # class for other tests.
        import types as _types
        self.send = _types.MethodType(_bad_send, self)

    monkeypatch.setattr(ump_mod.UserMessageProcessor, '__init__', _patched_init)

    msg = {"type": "chat", "text": "hello", "source": "text"}
    ws_mod._handle_chat(ws, store, msg, active_request={'id': None})

    errs = ws.frames_of('error')
    assert errs, f"Expected error frame, got: {ws.sent}"

    # Find the error frame that carries metrics — the handler may emit
    # one early error frame without metrics if the pub/sub race resolves
    # that way; our fix guarantees the final path populates partial_metrics.
    err_with_metrics = [e for e in errs if 'metrics' in e]
    assert err_with_metrics, (
        f"No error frame carried metrics. All errors: {errs}"
    )
    m = err_with_metrics[-1]['metrics']
    assert 'tokens_total' in m
    assert m.get('tokens_total_complete') is False
    assert 'response_time_s' in m
    # The failing send() recorded a tool — snapshot should reflect it.
    assert m.get('tools', {}).get('test_tool') == 1
