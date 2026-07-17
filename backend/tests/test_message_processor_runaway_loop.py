"""Feature test: the hard runaway-loop guard.

A turn is a recursive ``MessageProcessor._step`` chain with NO iteration cap — a
runaway is a hard-restart condition, not a silently-capped one. Two measures,
both scoped to one ``turn_execution`` (one ``_drive`` / one MessageProcessor
instance, across the whole recursive step chain), trip a loud ``RunAwayLoop``:

1. the same ``(tool, params)`` invoked ``_RUNAWAY_TOOL_CALL_LIMIT`` (5) times, or
2. the same non-empty response text emitted ``_RUNAWAY_TEXT_LIMIT`` (3) times.

``_drive`` catches the raise and stamps the turn CRASHED, carrying the reason on
``turn_executions.stop_reason``.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — exactly what ``MessageProcessor.process()`` does) against the
real, fully-migrated SQLite DB, with the real ``DispatchService``,
``TurnExecutionService`` and ``TranscriptService`` doing every read/write. The
only substitution is the LLM network boundary: ``ProviderService`` builds its
thin transport client via ``services.provider_service.build_client`` (the exact
seam ``test_message_processor_cancel_mid_provider_call.py`` uses); the fake
client replays a scripted sequence of provider responses, one per step. No real
side effects run either: most tool calls name a tool that does not exist (the
dispatcher records a harmless ``Unknown tool`` row and returns), and the one test
that uses a real tool (``weather``, for its synonym-healed schema) relies on the
guard firing BEFORE dispatch — so that tool never executes. The guard fires on
the call/text tally, independent of what any tool does.
"""

import sqlite3
import threading
import time
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import (
    _RUNAWAY_TEXT_LIMIT,
    _RUNAWAY_TOOL_CALL_LIMIT,
    MessageProcessor,
)
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_cancel_mid_provider_call.py).
_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool, empty)
    response rather than raising: a completed ``user`` turn spawns a
    fire-and-forget skill-suggestion turn (``_proactive_suggestion``) that makes
    its own provider call on a daemon thread this test does not script for. This
    tolerance cannot mask a guard failure — the runaway measures each end the
    turn CRASHED, and a guard that failed to fire would instead consume this
    terminal response and end the *asserted* turn COMPLETED, which every crash
    test asserts against."""

    _TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns (skill suggestion, thread gist) so they run to completion inside THIS
    test's provider+DB patch and never leak into the next test.

    ``_end`` starts them unjoined (``services.skill_suggestion_message_processor``
    / ``thread_gist_message_processor``): each is a whole ``MessageProcessor``
    turn on its own daemon (named ``skill-suggest``/``thread-gist`` and,
    transitively, a ``turn-*`` drive thread). Left running, such a turn would call
    ``build_client`` while the *next* test holds the patch, hitting that test's
    scripted provider and repatched DB — the exact cross-test corruption this
    guards against."""
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


def _run(provider: _ScriptedProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _assistant_rows(mp: MessageProcessor) -> list[dict[str, object]]:
    return [r for r in Transcript.by_turn(mp.channel, mp.turn_id) if r["role"] == "assistant"]


# ── Measure 1: identical tool call ────────────────────────────────────────────


def test_identical_tool_call_at_limit_crashes_the_turn(db: sqlite3.Connection) -> None:
    """The same tool with the same params, emitted ``_RUNAWAY_TOOL_CALL_LIMIT``
    times in a single response, trips ``RunAwayLoop`` BEFORE any of them is
    dispatched: the turn ends CRASHED, the reason names the looping tool, and
    zero tool rows / zero assistant rows were written."""
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    batch = [_tool("noop_probe", q="loop") for _ in range(_RUNAWAY_TOOL_CALL_LIMIT)]
    provider = _ScriptedProvider(ProviderResponse(text="", model="scripted", tool_calls=batch))

    mp = _run(provider, "run the probe")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "identical parameters" in execution.stop_reason
    assert "noop_probe" in execution.stop_reason
    # Guard trips before dispatch and before the prose store: none of the
    # runaway calls executed (the lone row is the framework's turn-0 memory
    # seed, which bypasses _step and so never counts toward the guard), and no
    # assistant prose was written.
    assert [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"] == []
    assert _assistant_rows(mp) == []


def test_one_below_the_tool_limit_completes(db: sqlite3.Connection) -> None:
    """Contrast: the same call ``_RUNAWAY_TOOL_CALL_LIMIT - 1`` times is under
    the limit — every call dispatches and the turn completes normally. Proves it
    is the tally crossing the threshold, not merely emitting a repeated call,
    that trips the guard."""
    assert db is not None
    under = _RUNAWAY_TOOL_CALL_LIMIT - 1
    batch = [_tool("noop_probe", q="loop") for _ in range(under)]
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="scripted", tool_calls=batch),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "run the probe")

    assert mp.result() == "All done."
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
    probe_rows = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"]
    assert len(probe_rows) == under  # all four really dispatched


def test_synonym_keyed_calls_canonicalise_to_one_and_crash(db: sqlite3.Connection) -> None:
    """The guard keys on the params ``DispatchService`` will EXECUTE, not the raw
    provider args. A model cycling synonym KEYS for one tool — ``location``, ``city``,
    ``loc``, ``place``, ``region``, all naming the same value — heals to a single
    identical call, so ``_RUNAWAY_TOOL_CALL_LIMIT`` such calls trip ``RunAwayLoop``
    exactly as byte-identical calls would. Under a raw-param guard each distinct key
    would tally separately and never trip. ``weather`` is real (its schema declares
    ``location`` with those synonyms), but the guard fires BEFORE dispatch, so no
    forecast lookup — hence no network — ever runs."""
    assert db is not None
    synonym_keys = ["location", "city", "loc", "place", "region"]  # all heal to Keys.location
    batch = [
        _tool("weather", **{synonym_keys[i % len(synonym_keys)]: "Valletta"})
        for i in range(_RUNAWAY_TOOL_CALL_LIMIT)
    ]
    # These are genuinely DIFFERENT raw keys: a raw-param guard would tally each
    # separately and never trip — only key-healing collapses them to one.
    assert len({next(iter(cast("dict[str, object]", c["input"]))) for c in batch}) > 1
    provider = _ScriptedProvider(ProviderResponse(text="", model="scripted", tool_calls=batch))

    mp = _run(provider, "what's the weather")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "identical parameters" in execution.stop_reason
    assert "weather" in execution.stop_reason
    # Guard trips before dispatch: no weather call executed (no network), no prose.
    assert [c for c in mp.tool_call_service.by_turn() if c.tool_name == "weather"] == []
    assert _assistant_rows(mp) == []


def test_async_flag_toggle_canonicalises_to_one_and_crashes(db: sqlite3.Connection) -> None:
    """``async`` is a framework control flag ``DispatchService`` strips before
    ``run()`` — it picks sync-vs-async dispatch, never what the tool does. A model
    toggling it around otherwise-identical params runs byte-identical calls, so the
    guard (keyed on the executed identity) collapses them: ``_RUNAWAY_TOOL_CALL_LIMIT``
    such calls trip ``RunAwayLoop``. Under a guard keyed on the raw args the
    alternating flag would split the tally and never trip."""
    assert db is not None
    batch = [
        _tool("noop_probe", **{"q": "loop", "async": bool(i % 2)})
        for i in range(_RUNAWAY_TOOL_CALL_LIMIT)
    ]
    # The raw calls genuinely differ (async true and false both appear): only
    # stripping the framework flag collapses them to one executed identity.
    assert {cast("dict[str, object]", c["input"])["async"] for c in batch} == {True, False}
    provider = _ScriptedProvider(ProviderResponse(text="", model="scripted", tool_calls=batch))

    mp = _run(provider, "run the probe")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "identical parameters" in execution.stop_reason
    assert "noop_probe" in execution.stop_reason
    assert [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"] == []


# ── Measure 2: identical response text ────────────────────────────────────────


def test_identical_text_at_limit_crashes_the_turn(db: sqlite3.Connection) -> None:
    """The same non-empty prose emitted on ``_RUNAWAY_TEXT_LIMIT`` tool-bearing
    steps trips ``RunAwayLoop`` on the last one. The tool calls differ every step
    (distinct params), so it is unmistakably the TEXT tally that fired — the
    reason says so and does not mention tool parameters."""
    assert db is not None
    loop_text = "Let me look into that for you."
    provider = _ScriptedProvider(*[
        ProviderResponse(text=loop_text, model="scripted", tool_calls=[_tool("probe", n=i)])
        for i in range(_RUNAWAY_TEXT_LIMIT)
    ])

    mp = _run(provider, "help me")

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CRASHED
    assert execution.stop_reason is not None
    assert "identical response text" in execution.stop_reason
    assert "identical parameters" not in execution.stop_reason
    # The final (tripping) step never stored its prose; the earlier ones did.
    assert len(_assistant_rows(mp)) == _RUNAWAY_TEXT_LIMIT - 1


def test_one_below_the_text_limit_completes(db: sqlite3.Connection) -> None:
    """Contrast: the same prose ``_RUNAWAY_TEXT_LIMIT - 1`` times, then a fresh
    final answer, completes normally. Proves the text tally — not any repeated
    prose at all — is what crosses the threshold."""
    assert db is not None
    loop_text = "Let me look into that for you."
    steps = [
        ProviderResponse(text=loop_text, model="scripted", tool_calls=[_tool("probe", n=i)])
        for i in range(_RUNAWAY_TEXT_LIMIT - 1)
    ]
    provider = _ScriptedProvider(
        *steps, ProviderResponse(text="Here is your answer.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "help me")

    assert mp.result() == "Here is your answer."
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.COMPLETED
