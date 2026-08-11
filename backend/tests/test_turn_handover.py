"""Feature test: the pre-compaction hand-over — ``compaction_b``.

Compaction erases the in-flight turn's act trail, and ``compaction_a`` cannot
fill the gap: it reads ``transcript_service.read()``, which floors every turn at
its ``settle0`` and therefore skips the unsettled turn entirely. The hand-over
pass owns exactly that gap — it reads the turn currently in flight and asks the
model to write itself a first-person hand-over BEFORE the erasure lands.

Six claims, one per branch the hand-over actually carries:

1. the hand-over is captured before the compaction marker exists — observed at
   the moment of the hand-over send, which is the only moment the ordering is
   visible (both writes land in the same second, so ``created_at`` cannot
   separate them, and the two ids come from independent sequences);
2. the hand-over row is never written under an input role — the count of
   ``role='user'`` rows must not grow by more than the turn's own input row,
   because ``Transcript.last_user_message_at()`` gates idle cognition on
   ``role = 'user'`` with no channel filter;
3. an empty hand-over from the model falls back to the anchor pointer rather
   than leaving the erasure standing alone;
4. the hand-over reaches the model through the user prompt;
5. it reaches the model through the SCHEDULE prompt too — the channel that
   carries most real compactions, and the one that had no continuity slot at
   all before this work;
6. a second compaction in the same turn carries the first hand-over forward,
   because ``turn_rows()`` is unfloored.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB, with the real
``ProviderService``, ``MessageProcessor`` step loop, dispatcher and
``TurnExecution`` state machine all running. The only substitution is the LLM
network boundary — ``services.provider_service.build_client``, the same seam
``test_context_limit_recovery.py`` uses — because the whole point is to script
when the provider refuses a request for length.
"""

import sqlite3
import threading
import time
from typing import cast
from unittest.mock import patch

import pytest

from abilities._turn_handover_config import TurnHandoverConfig
from abilities.chat_history_compactor import ChatHistoryCompactionConfig, ChatHistoryCompactor
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"

#: Deliberately below MAX_CONTEXT_WINDOW so an asserted window can only have come
#: from the source under test, never from a silently-substituted ceiling.
_CLIENT_WINDOW = 120_000

#: How the double tells the three passes apart: each sub-pass is identified by the
#: opening line of its OWN system prompt, read off the config rather than copied.
#: A reworded prompt would otherwise stop matching in silence, and the sub-pass
#: would fall through to the main-turn branch and be refused as the main turn.
_HANDOVER_MARK = TurnHandoverConfig().system_prompt.splitlines()[0]
_COMPACTION_MARK = ChatHistoryCompactionConfig().system_prompt.splitlines()[0]


def _compaction_markers() -> int:
    """How many compaction markers exist right now.

    ``DispatchService._execute`` opens the ``tool_calls`` row via
    ``tool_call_service.start`` BEFORE the ability runs, so this counts from the
    instant compaction is dispatched. Safe to call from the hand-over sub-turn's
    own thread: ``Model._bound_connection()`` re-derives the CALLING thread's
    connection on every access rather than freezing one.
    """
    return ToolCall.filter("tool_name", ChatHistoryCompactor.NAME).count()


def _system_of(dto: object) -> str:
    system = getattr(dto, "system", "")
    return system if isinstance(system, str) else ""


def _body_of(dto: object) -> str:
    """Every message body on a request DTO, joined — read whole rather than
    indexed so an extra framing message can never silently hide the payload."""
    messages = cast("list[dict[str, object]]", getattr(dto, "messages", None) or [])
    return "\n".join(str(m.get("content") or "") for m in messages)


class _HandoverProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol and scripts when the provider refuses a request for length.

    The refusal is addressed by CONTENT, not by call index: a sub-pass (the
    hand-over or the compactor) is always served, and the main turn is refused
    for its first ``main_rejections`` sends. Expressing it this way is what makes
    "two compactions happen" a statement about the turn rather than a
    hand-counted list of call slots that shifts whenever an inner pass adds one.

    Refusal comes from ``send`` rather than from anything measured before it,
    because that is the only shape a real one has: the provider is the sole
    authority on whether a request fit, and it can only say so once it has the
    request. Nothing in this path — production or double — sizes a payload.

    Records every hand-over body, and — at each hand-over send — how many
    ``chat_history_compactor`` markers already exist. That count is the only
    observable proof of ordering: both rows land in the same second, so
    ``created_at`` cannot separate them, and ``transcript.id`` and
    ``tool_calls.id`` come from independent AUTOINCREMENT sequences.
    """

    def __init__(self, main_rejections: int = 1, handover_text: str = "I was mid-task.") -> None:
        self._main_rejections = main_rejections
        self._handover_text = handover_text
        self.sends = 0
        self.main_sends = 0
        self.handover_bodies: list[str] = []
        self.markers_at_handover_send: list[int] = []

    def get_context_limit(self) -> int:
        return _CLIENT_WINDOW

    def send(self, dto: object) -> ProviderResponse:
        self.sends += 1
        system = _system_of(dto)
        if _HANDOVER_MARK in system:
            self.handover_bodies.append(_body_of(dto))
            self.markers_at_handover_send.append(_compaction_markers())
            return ProviderResponse(
                text=self._handover_text, model="scripted-handover",
                tool_calls=None, tokens_input=None,
            )
        if _COMPACTION_MARK in system:
            return ProviderResponse(
                text="## Now\nthe compacted checkpoint", model="scripted-compaction",
                tool_calls=None, tokens_input=None,
            )
        self.main_sends += 1
        if self.main_sends <= self._main_rejections:
            # No MessageProcessor: a client has no turn. ProviderService attaches
            # one before the handler sees it.
            raise ContextLimit("payload exceeds the model's maximum context length")
        return ProviderResponse(
            text="answered after making room", model="scripted-main",
            tool_calls=None, tokens_input=None,
        )


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed turn spawns
    so they run to completion inside THIS test's provider patch and never leak
    into the next — left running, each would call ``build_client`` while the
    NEXT test holds the patch."""
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


def _terminal_state(mp: MessageProcessor) -> str:
    """The turn's stamped terminal state, read back off the real row."""
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None, "the turn never stamped an execution row"
    return execution.state


def _run(provider: _HandoverProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _handover_rows(db: sqlite3.Connection, channel: str) -> list[sqlite3.Row]:
    return list(db.execute(
        "SELECT id, role, content FROM transcript WHERE channel = ? AND role = 'compaction' "
        "ORDER BY id ASC",
        (channel,),
    ))


def test_handover_is_captured_before_the_compaction_marker(db: sqlite3.Connection) -> None:
    """The hand-over is written while no compaction marker yet exists.

    This is the invariant the whole design rests on: the erasure must never land
    with nothing in its place. It is only observable DURING the hand-over send —
    afterwards both rows exist and neither ``created_at`` (one-second resolution)
    nor the two independent id sequences can order them. Reversing the two
    statements in ``ContextLimit.recover()`` turns the recorded count non-zero.
    """
    assert db is not None  # fixture taken for its binding side effect (real DB gateway)
    provider = _HandoverProvider(handover_text="I was halfway through the migration.")

    mp = _run(provider, "a very long conversation")

    assert _terminal_state(mp) == TurnExecution.COMPLETED
    assert provider.markers_at_handover_send == [0], (
        f"the hand-over must be captured before the compaction marker exists, "
        f"saw {provider.markers_at_handover_send} marker(s) already recorded"
    )
    rows = _handover_rows(db, mp.channel)
    assert len(rows) == 1, f"expected exactly one hand-over row, got {len(rows)}"
    assert cast("str", rows[0]["content"]) == "I was halfway through the migration."
    assert _compaction_markers() == 1, (
        "the compaction marker was never recorded — this turn never compacted"
    )


def test_handover_row_is_never_written_under_an_input_role(db: sqlite3.Connection) -> None:
    """The hand-over row carries ``role='compaction'`` and adds no input row.

    ``Transcript.last_user_message_at()`` is ``SELECT MAX(created_at) ... WHERE
    role = 'user'`` with NO channel filter, and it gates idle cognition
    (``IdleGatedJob._is_idle``). A hand-over written under an input role would
    fool it into believing the user had just spoken — on the schedule channel
    too, whose input row is also ``role='user'``. The delta is exactly 1: the
    turn's own input row, never two.
    """
    assert db is not None
    before = Transcript.filter("role", "user").count()

    mp = _run(_HandoverProvider(), "a very long conversation")

    rows = _handover_rows(db, mp.channel)
    assert len(rows) == 1
    assert cast("str", rows[0]["role"]) == "compaction"

    after = Transcript.filter("role", "user").count()
    assert after - before == 1, (
        f"the hand-over must not be written under an input role — role='user' "
        f"rows grew by {after - before}, expected only the turn's own input row"
    )


def test_empty_model_output_falls_back_to_the_anchor_pointer(db: sqlite3.Connection) -> None:
    """An empty hand-over never leaves the erasure standing alone.

    ``TurnHandoverService.capture`` is contractually never-empty: a model that
    returns nothing still yields the anchor pointer, so the turn always has
    SOMETHING telling it a cut-off happened.
    """
    assert db is not None
    provider = _HandoverProvider(handover_text="")  # the model returns nothing

    mp = _run(provider, "a very long conversation")

    assert mp.turn_handover, (
        "the hand-over must be non-empty even when the model returned nothing"
    )
    assert "Your session got cut-off" in mp.turn_handover, (
        f"expected the fallback pointer, got: {mp.turn_handover!r}"
    )


def test_handover_reaches_the_model_through_the_user_prompt(db: sqlite3.Connection) -> None:
    """The hand-over renders into the user prompt, inside the continuity frame.

    The transcript row alone can NEVER reach the model: ``transcript_service.read()``
    floors each turn at its ``settle0`` and the in-flight turn has none, so the
    row is skipped on the MAIN axis and excluded by the ``id < mp.uid`` ceiling on
    the FORK axis. The prompt slot is the only path that renders, which is why the
    row is durability and this is delivery.
    """
    assert db is not None
    text = "I was continuing the provider migration; next is the retry ladder."
    provider = _HandoverProvider(handover_text=text)

    mp = _run(provider, "a very long conversation")

    assert mp.turn_handover == text
    prompt = mp.prompt_service.user_prompt()
    assert text in prompt, f"the hand-over never reached the prompt: {prompt!r}"
    assert "not a new conversation" in prompt, (
        f"the hand-over rendered outside its continuity frame: {prompt!r}"
    )


def test_handover_reaches_the_model_through_the_schedule_prompt() -> None:
    """The schedule prompt carries the hand-over too.

    The schedule channel carries most real compactions, yet ``_user_prompt`` was
    the only one of the prompt builders with a continuity slot — a mid-turn
    compaction there erased the act trail and injected nothing in its place. The
    turn is built inertly and handed the state ``ContextLimit.recover()`` sets,
    because what is under test is delivery through ``_schedule_prompt``, which
    ``user_prompt()`` dispatches to on a SCHEDULED config.
    """
    mp = MessageProcessor(ScheduledConfig(), raw_input="run the weekly digest", turn_id=1)
    mp.turn_handover = "I had already pulled the digest rows; next is formatting."

    prompt = mp.prompt_service.user_prompt()

    assert mp.turn_handover in prompt, (
        f"the schedule prompt dropped the hand-over: {prompt!r}"
    )
    assert "not a new conversation" in prompt
    assert "Scheduled task:" in prompt, "this was not the schedule builder"


def test_second_compaction_carries_the_first_handover_forward(db: sqlite3.Connection) -> None:
    """Two compactions in one turn: the second hand-over sees the first.

    ``_fit`` reads ``turn_rows()``, which is UNFLOORED — it returns every row of
    the turn including the ``role='compaction'`` hand-over the first compaction
    wrote. So continuity chains instead of restarting from whatever survived the
    second erasure. A ``_fit`` switched to the floored ``read()`` would return
    nothing here and the chain would break.
    """
    assert db is not None
    first = "First hand-over: I was mid-migration."
    provider = _HandoverProvider(main_rejections=2, handover_text=first)

    mp = _run(provider, "a very long conversation")

    assert _terminal_state(mp) == TurnExecution.COMPLETED
    assert len(provider.handover_bodies) == 2, (
        f"expected two hand-over passes across two compactions, "
        f"got {len(provider.handover_bodies)}"
    )
    assert first in provider.handover_bodies[1], (
        f"the second hand-over must carry the first forward; its body was: "
        f"{provider.handover_bodies[1]!r}"
    )
    assert len(_handover_rows(db, mp.channel)) == 2, "each compaction writes its own row"
