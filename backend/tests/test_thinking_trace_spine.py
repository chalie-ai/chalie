# Feature tests for the thinking-trace spine — persistence and prompt-carriage.
#
# Drives the real production entry point (construct inertly, ``begin()``,
# ``result()``) against the real, fully-migrated SQLite DB, with the real
# ``MessageProcessor``, ``ProviderService``, ``TranscriptService``,
# ``ToolCallService`` and ``TranscriptThinking`` all running. The only
# substitution is the LLM network boundary: ``services.provider_service.build_client``
# (the same seam ``test_message_processor_runaway_loop.py`` uses). The fake client
# replays a scripted sequence of provider responses, one per step, and records
# every ``ProviderApiRequest`` it receives so the prompt body can be inspected.
#
# Three claims:
#
# 1. A 3-step turn (tool+text, tool+text, settle text), each with a distinct
#    ``thinking_block``, writes exactly 3 ``transcript_thinking`` rows:
#    steps 1–2 anchor to the assistant row that also anchors their tool calls;
#    step 3 anchors to the settled assistant row.
# 2. The request the double receives on call 2 carries step 1's trace
#    wrapped ``[thinking]…[end_thinking]`` in the user-message body, appearing
#    BEFORE that step's tool-call line.
# 3. A second turn (new user input) in the same channel/thread, driven through
#    a new ``MessageProcessor``, produces a first request with NO
#    ``[thinking]`` occurrence — traces are exchange-scoped and do not leak
#    across turns.

import sqlite3
import threading
import time
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from models.provider_response import ProviderResponse

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor
    from models.transcript_thinking import TranscriptThinking

pytestmark = pytest.mark.unit

_BUILD_CLIENT = "services.provider_service.build_client"


class _RecordingProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order,
    and records every ``ProviderApiRequest`` received.

    Past the end of the script it returns a benign terminal (no-tool, empty
    text) response — a completed ``user`` turn spawns a fire-and-forget
    skill-suggestion turn (``_proactive_suggestion``) that makes its own
    provider call on a daemon thread this test does not script for. This
    tolerance is benign: the spine only ever reads the request body for
    assertions inside the patched scope, and the recorded-request list grows
    only from calls made while the patch is active."""

    _TERMINAL = ProviderResponse(text="done", model="scripted-terminal", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0
        self.requests: list[object] = []

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        self.requests.append(dto)
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns so they run to completion inside THIS test's provider+DB patch and
    never leak into the next test."""
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


def _run(
    provider: _RecordingProvider, raw_input: str, *, turn_id: int = -1,
) -> "MessageProcessor":
    """Drive a real user turn to termination against *provider*, then quiesce
    any post-turn daemon turns inside the patch so the test stays hermetic."""
    from configs.channels.user import UserConfig
    from controllers.message_processor import MessageProcessor
    mp = MessageProcessor(UserConfig(), raw_input=raw_input, turn_id=turn_id)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


# ── helpers ───────────────────────────────────────────────────────────────────

def _assistant_rows(mp: "MessageProcessor") -> list[dict[str, object]]:
    from models.transcript import Transcript
    return [r for r in Transcript.by_turn(mp.channel, mp.turn_id) if r["role"] == "assistant"]


def _thinking_rows_by_turn(channel: str, turn_id: int) -> "list[TranscriptThinking]":
    from models.transcript_thinking import TranscriptThinking
    return TranscriptThinking.by_turn(channel, turn_id)


def _tool_call_rows(mp: "MessageProcessor") -> list[dict[str, object]]:
    return [c.to_dict() for c in mp.tool_call_service.by_turn()]


# ── Test 1: three-step execution persists three anchored rows ─────────────────

def test_three_step_execution_persists_three_anchored_rows(
    chat_provider: sqlite3.Connection,
) -> None:
    """Three provider responses — each with a distinct ``thinking_block`` —
    produce exactly three ``transcript_thinking`` rows after the turn completes.
    Steps 1 and 2 (tool-call + text) anchor to the assistant row that was just
    stored, the same row their tool calls anchor to. Step 3 (settle text, no
    tool calls) anchors to that settled assistant row."""
    trace_1 = "step one reasoning"
    trace_2 = "step two reasoning"
    trace_3 = "final settle reasoning"

    # Step 1: tool call + prose + thinking.
    resp_1 = ProviderResponse(
        text="looking into it now",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "first"}}],
        thinking_block=trace_1,
    )
    # Step 2: tool call + prose + thinking.
    resp_2 = ProviderResponse(
        text="got more info",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "second"}}],
        thinking_block=trace_2,
    )
    # Step 3: settle text, no tool calls + thinking.
    resp_3 = ProviderResponse(
        text="here is the answer",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_3,
    )
    provider = _RecordingProvider(resp_1, resp_2, resp_3)

    mp = _run(provider, "investigate this")

    rows = _thinking_rows_by_turn(mp.channel, mp.turn_id)
    assert len(rows) == 3, f"expected 3 thinking rows, got {len(rows)}"

    tool_calls = _tool_call_rows(mp)
    assistant_rows = _assistant_rows(mp)

    # Rows 1–2: each must anchor to the same transcript id as its corresponding
    # tool call (the assistant row written before dispatch in _step).
    for i in (0, 1):
        row = rows[i]
        assert row.transcript_id is not None
        # The assistant row for this step must exist and its id must match.
        assert any(
            cast("int", ar["id"]) == row.transcript_id
            for ar in assistant_rows
        )
        # At least one tool call for this step must anchor to the same id.
        assert any(
            cast("int", tc["transcript_id"]) == row.transcript_id
            for tc in tool_calls
        )

    # Row 3: must anchor to the final (settled) assistant row.
    row_3 = rows[2]
    settle_id = cast("int", assistant_rows[-1]["id"])
    assert row_3.transcript_id == settle_id


# ── Test 2: step-two prompt carries step-one trace ───────────────────────────

def test_step_two_prompt_carries_step_one_trace(
    chat_provider: sqlite3.Connection,
) -> None:
    """The user-message body of the request the double receives on call 2
    contains step 1's thinking trace wrapped in ``[thinking]…[end_thinking]``
    markers, and that block appears BEFORE the step 2 tool-call line in the
    rendered act trail."""
    trace_1 = "step one reasoning"
    trace_2 = "step two reasoning"

    resp_1 = ProviderResponse(
        text="looking into it now",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "first"}}],
        thinking_block=trace_1,
    )
    resp_2 = ProviderResponse(
        text="got more info",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "second"}}],
        thinking_block=trace_2,
    )
    resp_3 = ProviderResponse(
        text="here is the answer",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_2,
    )
    provider = _RecordingProvider(resp_1, resp_2, resp_3)

    _run(provider, "investigate this")

    # The second send is the step-2 request; inspect its user-message body.
    req_2 = provider.requests[1]
    messages = getattr(req_2, "messages", None)
    assert messages is not None and len(messages) >= 1
    body = cast("str", messages[0].get("content") or "")

    expected_marker = f"[thinking]{trace_1}[end_thinking]"
    assert expected_marker in body, (
        "step 1's trace marker not found in step 2 request body"
    )

    # The trace block must appear BEFORE the tool-call line of its own step in
    # the rendered trail. At request-2 build time only step 1's call exists in
    # the trail, so the ordering claim is: step-1 trace before step-1 tool line.
    assert body.index(expected_marker) < body.index("[noop_probe]"), (
        "the thinking trace must appear before its step's tool-call line"
    )


# ── Test 3: next-turn execution starts clean ─────────────────────────────────

def test_next_turn_execution_starts_clean(
    chat_provider: sqlite3.Connection,
) -> None:
    """A second turn (new user input) through a new ``MessageProcessor`` in
    the same channel/thread, driven against a fresh scripted provider,
    produces a first request whose user-message body contains NO
    ``[thinking]`` occurrence — traces are exchange-scoped and do not bleed
    across turns."""
    # First turn — already exercised, but we run it again to populate the
    # channel with thinking traces.
    trace_1 = "turn one reasoning"
    resp_1 = ProviderResponse(
        text="first turn done",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_1,
    )
    provider_1 = _RecordingProvider(resp_1)
    mp_1 = _run(provider_1, "first turn input")
    assert mp_1.turn_id is not None

    # Verify turn 1 wrote a thinking row (sanity).
    rows = _thinking_rows_by_turn(mp_1.channel, mp_1.turn_id)
    assert len(rows) == 1
    assert rows[0].thinking_trace == trace_1

    # Second turn — fresh MP, new scripted provider.
    trace_2 = "turn two reasoning"
    resp_2 = ProviderResponse(
        text="second turn done",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_2,
    )
    provider_2 = _RecordingProvider(resp_2)
    mp_2 = _run(provider_2, "second turn input", turn_id=mp_1.turn_id)
    assert mp_2.turn_id is not None

    # The first (and only) request of turn 2 must not carry any thinking trace.
    req_1 = provider_2.requests[0]
    messages = getattr(req_1, "messages", None)
    assert messages is not None and len(messages) >= 1
    body = cast("str", messages[0].get("content") or "")
    assert "[thinking]" not in body, (
        "turn 2's first request must not carry step-1 thinking traces"
    )

    # Turn 2's own trace must still be persisted. Both exchanges share the
    # turn (a reply forks into it), so by_turn holds both rows — the boundary
    # is prompt-side (the exchange floor), not storage-side.
    rows_2 = _thinking_rows_by_turn(mp_2.channel, mp_2.turn_id)
    assert len(rows_2) == 2
    assert rows_2[-1].thinking_trace == trace_2


# ── Test 4: step-one thinking anchors to its own tool-call transcript row ───

def test_tool_only_step_anchors_trace_with_its_calls(
    chat_provider: sqlite3.Connection,
) -> None:
    """Step 1 drives a tool call with EMPTY text (tools-only, no prose): the
    tool branch skips ``_store``, so no assistant row exists yet and both the
    tool call and the thinking trace anchor to the user input row
    (``current_transcript_id or uid``). Direct id-equality against the tool
    call's own anchor AND against ``mp.uid``."""
    trace_1 = "step one reasoning"
    trace_2 = "final settle reasoning"

    # Tools-only: empty text with a tool call is NOT the steer path (that
    # needs no text AND no tools) — it is the no-prose tool step.
    resp_1 = ProviderResponse(
        text="",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "first"}}],
        thinking_block=trace_1,
    )
    resp_2 = ProviderResponse(
        text="here is the answer",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_2,
    )
    provider = _RecordingProvider(resp_1, resp_2)

    mp = _run(provider, "investigate this")

    thinking_rows = _thinking_rows_by_turn(mp.channel, mp.turn_id)
    tool_calls = _tool_call_rows(mp)

    assert len(thinking_rows) == 2, f"expected 2 thinking rows, got {len(thinking_rows)}"
    # The turn-zero automatic memory recall records its own tool_calls row
    # against the same input anchor — scope to the scripted probe call.
    probe_calls = [c for c in tool_calls if c["tool_name"] == "noop_probe"]
    assert len(probe_calls) == 1, f"expected 1 probe call, got {len(probe_calls)}"

    # The no-prose step's trace and its tool call share the exact same anchor,
    # and that anchor is the exchange's user input row.
    anchor = cast("int", probe_calls[0]["transcript_id"])
    assert thinking_rows[0].transcript_id == anchor
    assert anchor == cast("int", mp.uid)


# ── Test 5: compaction cutoff suppresses pre-compaction traces in act_trail ──

def test_compaction_cutoff_hides_precompaction_traces(
    chat_provider: sqlite3.Connection,
) -> None:
    """A mid-turn ``chat_history_compactor`` call anchors to the last
    assistant row of the turn and acts as a fence: ``PromptService.act_trail``
    must not emit any thinking traces whose anchor is at or below that
    compactor's transcript id. The test builds the state directly against the
    DB and constructs a ``PromptService`` (reusing the ``mp`` from ``_run``)
    rather than driving a real compactor turn."""
    trace_1 = "pre-compaction reasoning"
    trace_2 = "post-compaction reasoning"

    resp_1 = ProviderResponse(
        text="looking into it now",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "before"}}],
        thinking_block=trace_1,
    )
    resp_2 = ProviderResponse(
        text="here is the answer",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_2,
    )
    provider = _RecordingProvider(resp_1, resp_2)

    mp = _run(provider, "investigate this")

    from services.prompt_service import PromptService

    assistant_rows = _assistant_rows(mp)
    last_assistant_id = cast("int", assistant_rows[-1]["id"])

    # Insert a synthetic compactor tool call anchored to the last assistant
    # row, mirroring how a real compaction call lands in the DB.
    from models.tool_call import ToolCall
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    ToolCall(
        transcript_id=last_assistant_id,
        tool_name="chat_history_compactor",
        params='{"reason": "mid-turn"}',
        result="",
        summary="",
        created_at=now,
        ended_at=now,
        state=ToolCall.DONE,
    ).save()

    rendered = PromptService(mp).act_trail()

    # The compactor's transcript_id is the cutoff; anything anchored at or
    # below it must be excluded from the trail, including thinking traces.
    assert "[thinking]" not in rendered, (
        "act_trail must not emit thinking traces anchored at or below the compactor"
    )
    assert trace_1 not in rendered, (
        "pre-compaction trace must be hidden by the compactor cutoff"
    )


# ── Test 6: serializer carries thinking sums per row ─────────────────────────

def test_serializer_carries_thinking_sums(
    chat_provider: sqlite3.Connection,
) -> None:
    """After a multi-step turn with thinking traces on multiple assistant
    rows, the turn serializer (``TurnSerializerService.serialize``, the same
    entry point used by ``GET /api/threads``) must project each row's
    thinking rows into ``msg["thinking"]`` with traces in row order and
    ``duration_ms`` / ``tokens`` equal to the sum across that row's anchor.
    Rows without thinking rows must not carry a ``"thinking"`` key at all."""
    trace_1 = "step one reasoning"
    trace_2a = "step two reasoning part a"
    trace_2b = "step two reasoning part b"
    trace_3 = "final settle reasoning"

    resp_1 = ProviderResponse(
        text="looking into it now",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "first"}}],
        thinking_block=trace_1,
    )
    # Step 2 fires two thinking passes — two separate ProviderResponse calls
    # are NOT needed: the model returns one thinking_block per response. To
    # get two thinking rows on the SAME assistant transcript, we drive a
    # second tool-call step (resp_2) whose assistant row then accumulates
    # both traces because the spine writes one transcript_thinking row per
    # provider call that produced a thinking_block.
    resp_2 = ProviderResponse(
        text="got more info",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "second"}}],
        thinking_block=trace_2a,
    )
    resp_3 = ProviderResponse(
        text="one more step",
        model="scripted",
        tool_calls=[{"name": "noop_probe", "input": {"q": "third"}}],
        thinking_block=trace_2b,
    )
    resp_4 = ProviderResponse(
        text="here is the answer",
        model="scripted",
        tool_calls=None,
        thinking_block=trace_3,
    )
    provider = _RecordingProvider(resp_1, resp_2, resp_3, resp_4)

    mp = _run(provider, "investigate this")

    from services.turn_serializer_service import get_service as _serializer

    result = _serializer().serialize(mp.channel, mp.turn_id)
    messages = cast("list[dict[str, object]]", result["messages"])

    # Build a map from transcript id to thinking rows for independent checks.
    thinking_rows = _thinking_rows_by_turn(mp.channel, mp.turn_id)
    from models.transcript import Transcript

    all_rows = Transcript.by_turn(mp.channel, mp.turn_id)
    by_id: dict[int, dict[str, object]] = {cast("int", r["id"]): r for r in all_rows}

    # Each message's transcript id.
    msg_ids = [cast("str", m["id"]) for m in messages]
    assert len(msg_ids) == len(all_rows), "serializer must return one message per transcript row"

    checked = 0
    for msg in messages:
        # Serialized message ids are strings (see _base_message); convert back
        # so the lookups below actually run rather than skipping every row.
        tid = int(cast("str", msg["id"]))
        assert tid in by_id, f"serialized message id {tid} has no transcript row"
        checked += 1

        has_thinking = "thinking" in msg
        row_thinking = [r for r in thinking_rows if r.transcript_id == tid]

        if row_thinking:
            assert has_thinking, (
                f"transcript row {tid} has thinking rows but msg has no 'thinking' key"
            )
            thinking = cast("dict[str, object]", msg["thinking"])
            traces = cast("list[str]", thinking["traces"])
            assert traces == [r.thinking_trace for r in row_thinking], (
                f"traces for row {tid} must be in row order"
            )
            total_duration = sum(r.duration_ms for r in row_thinking)
            total_tokens = sum(r.tokens for r in row_thinking)
            assert cast("int", thinking["duration_ms"]) == total_duration, (
                f"duration_ms for row {tid} must equal the sum of its traces"
            )
            assert cast("int", thinking["tokens"]) == total_tokens, (
                f"tokens for row {tid} must equal the sum of its traces"
            )
        else:
            assert not has_thinking, (
                f"transcript row {tid} has no thinking rows but msg carries 'thinking'"
            )

    assert checked == len(all_rows), "every serialized message must be checked"
