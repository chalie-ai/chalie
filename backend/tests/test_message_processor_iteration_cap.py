"""Feature test: the per-channel iteration cap on the ACT loop (#1950).

``_step`` recurses until the model stops emitting tool calls. With no cap, a
model stuck re-searching / re-reading loops until the context window caps out —
most harmful in background channels (thinking gate, DMN, encoder) that run
without user visibility, silently burning LLM calls. ``ProcessorConfig`` now
carries a ``max_iterations`` ClassVar; when a turn exceeds it, ``_step`` stops
recursing and ends the turn with whatever the model last produced (graceful
degradation — COMPLETED, not CRASHED).

Same harness as ``test_message_processor_runaway_loop.py``: real
MessageProcessor, real DB, real services; only the provider network boundary is
substituted with a scripted replay. The scripted provider never converges
(always emits a tool call with distinct params), so without the cap the loop
would run until the test times out.

Against the pre-cap tree this test fails RED (the provider's distinct calls
never trip the runaway guard, so ``_step`` never returns and ``result()``
deadlocks). It passes GREEN once the iteration cap lands.
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    return {"name": name, "input": params}


class _NonConvergingProvider:
    """Always emits a tool call with DISTINCT params every step — so the runaway
    guard (which keys on identical ``(tool, params)``) never trips. Without an
    iteration cap this loops forever; the cap is the only thing that ends it."""

    def __init__(self, cap: int) -> None:
        self.sends = 0
        self._cap = cap

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        # Every step: distinct params (n increments), so no runaway trip.
        return ProviderResponse(
            text=f"step {self.sends}", model="scripted",
            tool_calls=[_tool("noop_probe", n=self.sends)],
        )


class _OverflowTerminalProvider:
    """Like _NonConvergingProvider, but on the step AFTER the cap it returns a
    clean terminal (no-tool) response — proving the cap end is a normal turn
    end that stores the model's final prose, not a silent truncation."""

    _TERMINAL = ProviderResponse(text="All done.", model="scripted", tool_calls=None)

    def __init__(self, cap: int) -> None:
        self.sends = 0
        self._cap = cap

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        if self.sends > self._cap:
            return self._TERMINAL
        return ProviderResponse(
            text=f"step {self.sends}", model="scripted",
            tool_calls=[_tool("noop_probe", n=self.sends)],
        )


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def test_turn_completes_at_iteration_cap_not_loops_forever(db: sqlite3.Connection) -> None:
    """A non-converging model (distinct tool calls every step) hits the
    iteration cap and the turn ends COMPLETED — not CRASHED, not deadlocked.
    The number of provider sends never exceeds the cap."""
    assert db is not None
    cap = 3
    # Patch max_iterations low for a fast test (UserConfig defaults to 12).
    with patch.object(UserConfig, "max_iterations", cap):
        provider = _NonConvergingProvider(cap)
        mp = MessageProcessor(UserConfig(), raw_input="loop forever")
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            mp.result()
            _drain_background_turns()

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED, (
        f"a capped turn should complete gracefully, not crash (state={execution.state})"
    )
    # The cap stops recursion: the number of tool-call steps on THIS turn never
    # exceeds the cap. (Provider send count includes background post-turn work
    # like skill-suggestion, which runs on a different config — so we count tool
    # rows on this turn instead.)
    tool_rows = mp.tool_call_service.by_turn()
    noop_rows = [c for c in tool_rows if c.tool_name == "noop_probe"]
    assert len(noop_rows) <= cap, (
        f"{len(noop_rows)} tool-call steps on this turn, cap was {cap} — the loop did not stop"
    )


def test_cap_does_not_fire_below_threshold(db: sqlite3.Connection) -> None:
    """Contrast: a turn that converges BEFORE the cap completes normally and
    makes the full expected number of provider calls — the cap is a ceiling,
    not an early exit."""
    assert db is not None
    cap = 5
    with patch.object(UserConfig, "max_iterations", cap):
        # Converges on step 3 (well under the cap of 5).
        provider = _OverflowTerminalProvider(cap=3)
        mp = MessageProcessor(UserConfig(), raw_input="converge early")
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            result = mp.result()
            _drain_background_turns()

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    # 3 tool-bearing steps (well under the cap of 5), then a clean terminal —
    # the cap never fired. Count tool rows on THIS turn (not total provider
    # sends, which include background post-turn work on other configs).
    noop_rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"]
    assert len(noop_rows) == 3
    assert result == "All done."
