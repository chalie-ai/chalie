"""Feature test: a cancel that lands while the provider call is in flight must
discard the returned response, not persist it.

``_step()`` has no way to abort a provider call already in flight — every
client is one blocking, non-streaming request (``services/llm_clients/*``).
The only place a mid-flight cancel can be observed is the checkpoint right
after ``_send_with_retry`` returns and before the response is stored. Before
that checkpoint existed, the single-shot (no-tool-calls) branch called
``self._store(response.text)`` unconditionally, so a turn cancelled while the
model was still generating still persisted and rendered the full answer.

Drives the real production entry point (construct inertly, ``begin()``,
``result()`` — exactly what ``MessageProcessor.process()`` does) against the
real, fully-migrated SQLite database, with the real ``TurnExecutionService``
and ``TranscriptService`` doing every read/write. The only substitution is the
LLM network boundary: ``ProviderService`` builds its thin transport client via
``services.llm_clients.factory.build_client`` (mirrors the existing pattern in
``test_message_markdown_to_html.py``); the fake client's ``send()`` calls the
real ``TurnExecutionService.cancel()`` — the exact chokepoint both
``DELETE /api/threads/<turn_id>`` and a turn's own self-cancel ability route
through — before returning a normal, terminal (no tool calls) response. That
reproduces "the DELETE lands while the call is in flight" without touching any
internal flag directly.
"""

import json
import sqlite3
from typing import cast
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services.turn_serializer_service import get_service as _serializer
from services.websocket import Websocket

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_markdown_to_html.py for precedent).
_BUILD_CLIENT = "services.provider_service.build_client"


class _RecordingProvider:
    """Returns one fixed, terminal response and no tool calls."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def get_context_limit(self) -> int:
        return 200000


    def send(self, _dto: object) -> ProviderResponse:
        return ProviderResponse(text=self._response_text, model="recorder", tool_calls=None)


class _CancellingProvider(_RecordingProvider):
    """Same as ``_RecordingProvider``, but simulates a cancel request landing
    while this "network call" is in flight: it flips ``cancel_requested`` via
    the real ``TurnExecutionService.cancel()`` chokepoint before returning the
    response — the model still finished generating a full answer, but the
    turn was asked to stop before that answer came back."""

    def __init__(self, mp: MessageProcessor, response_text: str) -> None:
        super().__init__(response_text)
        self._mp = mp

    def send(self, dto: object) -> ProviderResponse:
        cancelled = self._mp.turn_execution_service.cancel()
        assert cancelled is not None  # sanity: the real cancel write succeeded
        return super().send(dto)


def _tool_step(text: str, probe: str) -> ProviderResponse:
    """A provider call that dispatches one (unknown-ability, so inert) tool call."""
    return ProviderResponse(
        text=text, model="recorder",
        tool_calls=[{"name": "noop_probe", "input": {"q": probe}}],
    )


class _ScriptedProvider:
    """Replays one scripted response per ``send()``, in order; past the end of
    the script it returns a terminal answer (a cancelled turn never gets there)."""

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._sent = 0

    def get_context_limit(self) -> int:
        return 200000

    def send(self, _dto: object) -> ProviderResponse:
        if self._sent >= len(self._responses):
            return ProviderResponse(text="unreached", model="recorder", tool_calls=None)
        response = self._responses[self._sent]
        self._sent += 1
        return response


class _CancelOnToolCallProvider(_RecordingProvider):
    """Cancel lands while the provider call is in flight, and the call that
    comes back made tool calls (no prose) — a call whose row would be an empty
    step row, were it stored."""

    def __init__(self, mp: MessageProcessor) -> None:
        super().__init__("")
        self._mp = mp

    def send(self, _dto: object) -> ProviderResponse:
        self._mp.turn_execution_service.cancel()
        return _tool_step("", "never dispatched")


class _CancelOnRowCount:
    """A real socket-registry subscriber (``send(str)`` Protocol) that requests
    the turn's cancel — the same chokepoint ``DELETE /api/threads/<turn_id>``
    uses — the moment the turn's Nth assistant row is stored. A row's store
    broadcasts its ``updated`` frame, so this lands the cancel after that call
    stored its row and before its tool calls dispatch."""

    def __init__(self, mp: MessageProcessor, stored_rows: int) -> None:
        self._mp = mp
        self._stored_rows = stored_rows
        self._fired = False

    def send(self, data: str) -> None:
        frame = json.loads(data)
        if self._fired or frame.get("status") != "updated" or frame.get("turn_id") != self._mp.turn_id:
            return
        stored = [
            r for r in Transcript.by_turn(self._mp.channel, self._mp.turn_id)
            if r["role"] == "assistant"
        ]
        if len(stored) >= self._stored_rows:
            self._fired = True
            self._mp.turn_execution_service.cancel()


def test_cancel_observed_after_provider_returns_discards_response_and_ends_cancelled(
    db: sqlite3.Connection,
) -> None:
    """A cancel that lands while the provider call is in flight, on a
    single-shot (no-tool-calls) turn, must discard the returned essay: zero
    assistant transcript rows for this turn, and the execution row ends
    CANCELLED rather than COMPLETED."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _CancellingProvider(mp, "Here is the full essay the model produced.")

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()

    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert assistant_rows == []  # the essay was discarded, never persisted

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.CANCELLED
    assert bool(execution.cancel_requested) is True


def test_uncancelled_turn_persists_the_same_response(db: sqlite3.Connection) -> None:
    """Contrast case, same essay, same single-shot shape, no cancel: this
    proves the checkpoint — not some unrelated no-op — is what makes the
    difference above. Without a cancel request the response IS persisted and
    the execution row ends COMPLETED."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="write me a short essay")  # inert (I2)
    provider = _RecordingProvider("Here is the full essay the model produced.")

    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        result = mp.result()

    assert result == "Here is the full essay the model produced."
    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "Here is the full essay the model produced."

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.ended_at is not None
    assert execution.state == TurnExecution.COMPLETED
    assert bool(execution.cancel_requested) is False


def test_cancel_after_a_tool_step_keeps_the_rows_it_already_wrote(db: sqlite3.Connection) -> None:
    """A cancel that lands once a provider call has returned and stored its row
    — before that call's tool calls run — stops the turn but leaves everything
    already written in place. Two tool steps are scripted and the cancel lands
    right after the SECOND stores its row: both (unsettled) rows survive, the
    first call's tool call is still anchored to the first row, the second call
    never dispatched, and the turn stays visible in the thread feed and the
    expanded thread with that row's tool chip."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="look into both things")  # inert (I2)
    provider = _ScriptedProvider(
        _tool_step("", "first"),
        _tool_step("Checking the second thing.", "second"),
        _tool_step("", "never reached"),
    )
    canceller = _CancelOnRowCount(mp, stored_rows=2)

    Websocket._connect(canceller)
    try:
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            mp.result()
    finally:
        Websocket._disconnect(canceller)

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CANCELLED
    rows = [r for r in Transcript.by_turn(mp.channel, mp.turn_id) if r["role"] == "assistant"]
    assert [r["content"] for r in rows] == ["", "Checking the second thing."]
    assert [r["settled"] for r in rows] == [0, 0]
    # Only the first call's tool call ran, and it hangs off the first row.
    probes = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"]
    assert [c.transcript_id for c in probes] == [rows[0]["id"]]
    # Still on the feed, and the expanded thread shows the first row's chip.
    assert [t["turn_id"] for t in Transcript.recent_threads(mp.channel)] == [mp.turn_id]
    assert Transcript.count_turns(mp.channel) == 1
    messages = cast("list[dict[str, object]]", _serializer().serialize(mp.channel, mp.turn_id)["messages"])
    by_id = {m["id"]: m for m in messages}
    assert str(rows[0]["id"]) in by_id, "the first step's row must be in the expanded thread"
    step_chips = cast("list[dict[str, object]]", by_id[str(rows[0]["id"])]["tool_calls"])
    assert [c["tool_name"] for c in step_chips] == ["noop_probe"]


def test_cancel_before_a_tool_call_is_stored_leaves_no_row_and_hides_the_turn(
    db: sqlite3.Connection,
) -> None:
    """The other side of the same rule: a cancel that lands while a tool-only
    call is still in flight is observed before that call's row exists, so no
    assistant row is written for it, its tool call never dispatches, the turn
    ends CANCELLED — and with nothing but the user's own input left, the turn
    is hidden from the thread feed exactly like any cancelled orphan."""
    assert db is not None  # fixture is taken for its binding side effect (real DB gateway)
    mp = MessageProcessor(UserConfig(), raw_input="look into both things")  # inert (I2)

    with patch(_BUILD_CLIENT, return_value=_CancelOnToolCallProvider(mp)):
        mp.begin()
        mp.result()

    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None
    assert execution.state == TurnExecution.CANCELLED
    rows = Transcript.by_turn(mp.channel, mp.turn_id)
    assert [r for r in rows if r["role"] == "assistant"] == []
    assert [c for c in mp.tool_call_service.by_turn() if c.tool_name == "noop_probe"] == []
    assert Transcript.recent_threads(mp.channel) == []
    assert Transcript.count_turns(mp.channel) == 0
