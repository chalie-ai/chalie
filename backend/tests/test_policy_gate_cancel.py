"""Feature test: a turn cancelled while parked on an interactive
`ask` permission gate must unwind instead of pinning the per-channel turn lock.

Drives the REAL tool-call chokepoint (``ToolDispatcher.dispatch`` — the same
entry the action endpoint calls) against the REAL policy gate and a REAL test
database. A non-internal probe tool with no seed row resolves to a lazily-created
'ask', so dispatch parks in the real ``PolicyManager._ask_user`` exactly as a
gated chat tool does. No mocks: dispatch sources ``should_stop`` off the
invoking mp-like ctx (the same duck-typed contract ``api/chat.py``'s
``_ActionCtx`` and the real ``MessageProcessor`` both expose) and threads it
through ``wrap → authorize → _ask_user``; the gate dict is the inspectable
surface ``POST /api/policies/respond`` normally wakes. Before the fix the park
is unbounded → the worker never joins (RED); after it, cancel wakes the gate,
the tool is denied without ever running, and the gate cleans up (GREEN)."""

import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._registry import _get_registry
from abilities._result import ToolResult
from services import policy_manager
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

CH = ProcessorConfig.PolicyChannel


class _GatePark(Ability):
    """Registered probe tool whose run() must NEVER fire on the cancel path: the
    turn is denied at the gate before the callback ever executes. Kept synthetic
    + non-discoverable so it never pollutes the real registry / find_tools."""

    _SYNTHETIC = True
    DISCOVERABLE = False
    ran = False  # class-level sentinel — True iff run() executed (it must not)

    def get_name(self) -> str: return "test_gate_park"
    def get_summary(self) -> str: return "Gate-park probe (feature-test fixture)."
    def get_search_tooltip(self) -> str: return "gate-park probe (feature-test fixture)"
    def get_examples(self) -> list[str]: return ["park at the ask gate", "wait for approval"]
    def get_parameters(self) -> dict[str, object]: return {"type": "object", "properties": {}, "required": []}

    def run(self, params: dict[str, object]) -> ToolResult:
        type(self).ran = True
        return ToolResult.ok("should never run on the cancel path")


class _TurnCtx:
    """Mirrors api/chat.py's _ActionCtx — the mp-like object dispatch consumes:
    config (for policy_channel), uid/anchor (ActTrail anchor), should_stop()."""

    def __init__(self, cancel_event: threading.Event) -> None:
        self.config = SimpleNamespace(policy_channel=CH.CHAT, broadcast_to=None)
        self.uid = None
        self.anchor = None
        self._cancel_event = cancel_event

    def should_stop(self) -> bool:
        return self._cancel_event.is_set()


def test_cancel_unwinds_parked_ask_gate(db: sqlite3.Connection) -> None:
    _GatePark.ran = False
    cancel = threading.Event()
    ctx = _TurnCtx(cancel)

    registry = _get_registry()
    registry["test_gate_park"] = _GatePark()
    before_gates = set(policy_manager._permission_gates)
    result: list[str] = []
    worker = threading.Thread(
        target=lambda: result.append(ToolDispatcher(ctx).dispatch("test_gate_park", {})),
        daemon=True,
    )
    try:
        worker.start()

        # Park: dispatch reaches the real _ask_user and registers a gate.
        for _ in range(80):
            if set(policy_manager._permission_gates) - before_gates:
                break
            time.sleep(0.025)
        assert set(policy_manager._permission_gates) - before_gates, "dispatch never reached the ask gate"
        assert worker.is_alive(), "worker finished instead of parking on the gate"
        assert _GatePark.ran is False, "gated tool ran before any approval"

        # Cancel the turn. On the unfixed code the gate waits unbounded, so the
        # worker never joins and the per-channel lock would stay held (deadlock).
        cancel.set()
        worker.join(timeout=3)
        assert not worker.is_alive(), "cancel did not unwind the parked ask gate"

        # Denied without executing; gate cleaned up; cancel logged no verdict.
        assert _GatePark.ran is False, "gated tool executed despite the cancel"
        assert set(policy_manager._permission_gates) - before_gates == set(), "permission gate leaked after cancel"
        assert result == ["The test_gate_park action is not allowed. Do NOT retry."]
        assert db.execute("SELECT COUNT(*) FROM policy_blocked_log").fetchone()[0] == 0, \
            "cancel wrongly recorded a user_denied audit row"
    finally:
        registry.pop("test_gate_park", None)
