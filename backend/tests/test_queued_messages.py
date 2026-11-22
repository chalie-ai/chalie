"""Feature tests: a message delivered to the chat while a turn there is still
working waits its turn, and each turn's end opens the one that waited longest.

An agent's message to the person's chat is a turn of its own — it never joins
the one working. ``MessageProcessor.deliver`` opens it at once on an idle chat
and otherwise queues it; every turn's end (``_drive``'s last step) and the boot
pass open the oldest waiting message through ``MessageProcessor.drain``. These
tests drive the real ``MessageProcessor`` ACT loop, the real ``transcript`` /
``turn_executions`` / ``queued_messages`` tables and the real transactions
against the real migrated SQLite database.

The only substitution is the LLM network boundary
(``services.provider_service.build_client``, the seam every turn-driving test
uses): a double that answers every call with a short reply and can hold a call
open, so a turn stays working while messages are delivered. Every assertion is
on stored rows — input rows, execution rows, waiting rows — never on how often
anything was called.
"""

import sqlite3
import threading
import time
from collections.abc import Iterator
from typing import cast

import pytest

import services.provider_service as provider_service
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.queued_message import QueuedMessage
from services.provider_api import ProviderApiRequest

pytestmark = pytest.mark.unit

_CHANNEL = "user"
# A channel no config opens a waiting message on: only the chat's are opened.
_UNOPENED_CHANNEL = "schedule"
_BUSY_QUESTION = "plan my week"
_TURN_THREAD_NAMES = ("skill-suggest", "thread-gist")
_TURN_THREAD_PREFIXES = ("turn-", "memory-step-")


# ── the provider double and the harness around it ────────────────────────────


class _Gate:
    """One held provider call: ``entered`` is set once the call is parked,
    and the call answers only once ``release`` is set."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()


class _Provider:
    """Answers every call with a short text reply. Each ``hold_next`` gate is
    taken by the next call to arrive, in order, and parks it until released —
    the window in which a test delivers messages to a chat whose turn is
    working. Every call a user turn with a text-only reply makes is its own
    drive thread's single step, so the next call to arrive is the next turn's."""

    _REPLY = ProviderResponse(text="Noted.", model="scripted", tool_calls=None)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: list[_Gate] = []
        self._made: list[_Gate] = []

    def hold_next(self) -> _Gate:
        gate = _Gate()
        with self._lock:
            self._waiting.append(gate)
            self._made.append(gate)
        return gate

    def release_all(self) -> None:
        with self._lock:
            self._waiting.clear()
            made = list(self._made)
        for gate in made:
            gate.release.set()

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        with self._lock:
            gate = self._waiting.pop(0) if self._waiting else None
        if gate is not None:
            gate.entered.set()
            assert gate.release.wait(30), "the held provider call was never released"
        return self._REPLY


def _turn_threads() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.name in _TURN_THREAD_NAMES or t.name.startswith(_TURN_THREAD_PREFIXES)
    ]


def _settle(timeout_s: float = 60.0) -> None:
    """Join every thread a turn spawns — the drive threads, including each one
    a turn's end opens for a waiting message, and their post-turn daemons — so
    the chat has fully settled and nothing leaks into the next test."""
    deadline = time.monotonic() + timeout_s
    while (pending := _turn_threads()) and time.monotonic() < deadline:
        for t in pending:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
    leftover = [t.name for t in _turn_threads()]
    assert not leftover, f"turn threads never finished: {leftover}"


@pytest.fixture
def provider(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Provider]:
    double = _Provider()
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    yield double
    double.release_all()  # a failed test must not strand a parked drive thread
    _settle()


def _start_busy_chat(provider: _Provider) -> tuple[MessageProcessor, _Gate]:
    """The person asks the chat something and the turn is still working on it:
    its provider call is parked."""
    gate = provider.hold_next()
    chat = MessageProcessor.process(UserConfig(), raw_input=_BUSY_QUESTION)
    assert gate.entered.wait(30), "the chat's turn never reached its provider call"
    return chat, gate


# ── reading what was stored ──────────────────────────────────────────────────


def _queued(db: sqlite3.Connection, channel: str = _CHANNEL) -> list[tuple[str, int | None]]:
    """The messages waiting on ``channel`` as ``(text, tool_call_id)``, in the
    order they arrived."""
    rows = db.execute(
        "SELECT text, tool_call_id FROM queued_messages WHERE channel = ? ORDER BY id", (channel,),
    ).fetchall()
    return [(str(r["text"]), cast("int | None", r["tool_call_id"])) for r in rows]


def _input_row(db: sqlite3.Connection, row_id: int | None) -> tuple[str, str | None]:
    """The input row ``row_id`` as ``(content, tool_call_id)``."""
    row = db.execute("SELECT content, tool_call_id FROM transcript WHERE id = ?", (row_id,)).fetchone()
    assert row is not None, f"no input row {row_id}"
    return str(row["content"]), cast("str | None", row["tool_call_id"])


def _inputs(db: sqlite3.Connection) -> list[tuple[int, int, str, str | None]]:
    """The chat's input rows as ``(id, turn_id, content, tool_call_id)``, in
    the order they were written."""
    rows = db.execute(
        "SELECT id, turn_id, content, tool_call_id FROM transcript "
        "WHERE channel = ? AND role = 'user' ORDER BY id",
        (_CHANNEL,),
    ).fetchall()
    return [
        (int(r["id"]), int(r["turn_id"]), str(r["content"]), cast("str | None", r["tool_call_id"]))
        for r in rows
    ]


def _executions(db: sqlite3.Connection) -> list[tuple[int, str | None, str | None, str]]:
    """The chat's executions as ``(turn_id, input text, input tool_call_id,
    state)``, in the order they opened."""
    rows = db.execute(
        "SELECT e.turn_id, t.content, t.tool_call_id, e.state FROM turn_executions e "
        "LEFT JOIN transcript t ON t.id = e.input_id WHERE e.channel = ? ORDER BY e.id",
        (_CHANNEL,),
    ).fetchall()
    return [
        (
            int(r["turn_id"]),
            cast("str | None", r["content"]),
            cast("str | None", r["tool_call_id"]),
            str(r["state"]),
        )
        for r in rows
    ]


def _open_inputs(db: sqlite3.Connection) -> list[int | None]:
    """The input row of every execution still working on the chat."""
    rows = db.execute(
        "SELECT input_id FROM turn_executions WHERE channel = ? AND ended_at IS NULL ORDER BY id", (_CHANNEL,),
    ).fetchall()
    return [cast("int | None", r["input_id"]) for r in rows]


# ── 1. an idle chat ──────────────────────────────────────────────────────────


def test_a_message_delivered_to_an_idle_chat_opens_its_own_turn_before_deliver_returns(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """Nothing is working on the chat and nothing waits, so an agent's message
    opens its turn at once: by the time ``deliver`` returns, the message is the
    chat's input row — carrying the tool call that sent it — and the turn's
    execution is open on that row. Nothing was queued, and the turn runs to
    completion."""
    gate = provider.hold_next()

    opened = MessageProcessor.deliver(UserConfig(), "x", 7)

    assert opened is not None
    assert _input_row(db, opened.uid) == ("x", "7")
    assert _open_inputs(db) == [opened.uid]
    assert _queued(db) == []

    assert gate.entered.wait(30), "the opened turn never reached its provider call"
    gate.release.set()
    _settle()
    assert _executions(db) == [(opened.turn_id, "x", "7", "completed")]


# ── 2. a busy chat: one waiting message per turn end, oldest first ───────────


def test_messages_delivered_while_the_chat_works_wait_and_its_end_opens_only_the_oldest(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """The person's question is still being answered when two agents report
    back. Neither message joins the working turn and neither opens a second
    one: both wait, in arrival order. When the question's turn ends, exactly
    one new turn opens — the FIRST message's, carrying its own tool call —
    and the second message is still waiting behind it."""
    chat, first = _start_busy_chat(provider)

    assert MessageProcessor.deliver(UserConfig(), "the search finished", 11) is None
    assert MessageProcessor.deliver(UserConfig(), "the browse finished", 12) is None
    assert _queued(db) == [("the search finished", 11), ("the browse finished", 12)]
    assert _open_inputs(db) == [chat.uid]

    second = provider.hold_next()
    first.release.set()
    assert second.entered.wait(30), "the question's turn end opened no waiting message"

    assert [(text, call, state) for _, text, call, state in _executions(db)] == [
        (_BUSY_QUESTION, None, "completed"),
        ("the search finished", "11", "working"),
    ]
    assert _queued(db) == [("the browse finished", 12)]
    second.release.set()


# ── 3. a queue of two runs out, in order ─────────────────────────────────────


def test_two_waiting_messages_become_two_consecutive_turns_in_arrival_order(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """Two messages wait behind the person's question. Once everything has
    settled, each became a turn of its own, after the question and in the order
    it arrived; every turn completed, nothing was lost or answered twice, and
    nothing is left waiting."""
    _, gate = _start_busy_chat(provider)
    assert MessageProcessor.deliver(UserConfig(), "the search finished", 21) is None
    assert MessageProcessor.deliver(UserConfig(), "the browse finished", 22) is None

    gate.release.set()
    _settle()

    inputs = _inputs(db)
    assert [text for _, _, text, _ in inputs] == [_BUSY_QUESTION, "the search finished", "the browse finished"]
    turns = [turn_id for _, turn_id, _, _ in inputs]
    assert turns == sorted(set(turns)), f"each message must open its own later turn: {turns}"
    assert [(text, call, state) for _, text, call, state in _executions(db)] == [
        (_BUSY_QUESTION, None, "completed"),
        ("the search finished", "21", "completed"),
        ("the browse finished", "22", "completed"),
    ]
    assert _queued(db) == []


# ── 4. the boot pass ─────────────────────────────────────────────────────────


def test_at_boot_a_message_left_waiting_opens_and_a_drain_while_the_chat_works_opens_nothing(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """A restart cut off the turn a message was waiting behind, so no turn end
    is left to open it. The boot pass opens it on the now-idle chat and removes
    it from the queue. A message waiting on a channel the queue cannot open
    stays where it is, and the boot pass carries on rather than failing. While
    the reopened turn works, a further drain opens nothing — a second message
    keeps waiting — and that turn's own end opens it."""
    QueuedMessage(channel=_CHANNEL, text="left waiting before the restart", tool_call_id=31).save()
    QueuedMessage(channel=_UNOPENED_CHANNEL, text="a reminder", tool_call_id=None).save()
    gate = provider.hold_next()

    opened = {channel: MessageProcessor.drain(channel) for channel in QueuedMessage.channels()}

    chat_turn = opened[_CHANNEL]
    assert chat_turn is not None
    assert _input_row(db, chat_turn.uid) == ("left waiting before the restart", "31")
    assert _open_inputs(db) == [chat_turn.uid]
    assert _queued(db) == []
    assert opened[_UNOPENED_CHANNEL] is None
    assert _queued(db, _UNOPENED_CHANNEL) == [("a reminder", None)]

    assert MessageProcessor.deliver(UserConfig(), "a second message", 32) is None
    assert MessageProcessor.drain(_CHANNEL) is None
    assert _open_inputs(db) == [chat_turn.uid]
    assert _queued(db) == [("a second message", 32)]

    assert gate.entered.wait(30), "the reopened turn never reached its provider call"
    gate.release.set()
    _settle()
    assert [text for _, text, _, _ in _executions(db)] == ["left waiting before the restart", "a second message"]
    assert _queued(db) == []


# ── 5. delivering while a turn ends ──────────────────────────────────────────


_SENDERS = 4
# Bounds the run should the chat's end ever be slow to observe.
_MAX_REPORTS_PER_SENDER = 25


def test_messages_delivered_as_the_chats_turn_ends_are_each_answered_exactly_once_one_turn_at_a_time(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """Several agents report back at the very moment the person's question is
    answered, and keep reporting while its turn closes. Whichever side of the
    turn's end each message lands on — opened at once on an idle chat, or
    queued and opened by a later turn's end — once everything settles every
    message was answered exactly once, by a turn of its own carrying its own
    tool call; one agent's messages were answered in the order it sent them;
    no two turns ever ran at once (no turn's rows interleave with another's);
    and nothing is left waiting."""
    chat, gate = _start_busy_chat(provider)
    start = threading.Barrier(_SENDERS + 1)
    chat_ended = threading.Event()
    lock = threading.Lock()
    sent: list[tuple[int, str, int, int | None]] = []
    errors: list[BaseException] = []

    def send_from(sender: int) -> None:
        """Report from the moment the question's turn is released until one
        report has gone out after that turn's end, so the reports straddle it."""
        try:
            start.wait(30)
            for k in range(_MAX_REPORTS_PER_SENDER):
                after_the_end = chat_ended.is_set()
                text, call_id = f"agent {sender} report {k}", 100 * (sender + 1) + k
                opened = MessageProcessor.deliver(UserConfig(), text, call_id)
                with lock:
                    sent.append((sender, text, call_id, None if opened is None else opened.uid))
                if after_the_end:
                    return
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assertion below
            with lock:
                errors.append(exc)

    senders = [threading.Thread(target=send_from, args=(n,), name=f"agent-{n}") for n in range(_SENDERS)]
    for t in senders:
        t.start()
    try:
        start.wait(30)
        gate.release.set()
        chat.result()
        chat_ended.set()
    finally:
        chat_ended.set()
        for t in senders:
            t.join(30)
    _settle(timeout_s=120)

    assert errors == []
    inputs = _inputs(db)
    assert sorted(text for _, _, text, _ in inputs) == sorted([_BUSY_QUESTION, *(text for _, text, _, _ in sent)])
    by_text = {text: (row_id, turn_id, call) for row_id, turn_id, text, call in inputs}
    for _, text, call_id, opened_uid in sent:
        assert by_text[text][2] == str(call_id), f"{text!r} carries another message's tool call"
        if opened_uid is not None:
            assert by_text[text][0] == opened_uid, f"the turn opened for {text!r} carries another message"
    assert len({turn_id for _, turn_id, _, _ in inputs}) == len(inputs), "two messages shared one turn"
    for sender in range(_SENDERS):
        own = [by_text[text][1] for n, text, _, _ in sent if n == sender]
        assert own == sorted(own), f"agent {sender}'s reports were answered out of order: {own}"

    turn_order = [int(r["turn_id"]) for r in db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? ORDER BY id", (_CHANNEL,),
    ).fetchall()]
    assert turn_order == sorted(turn_order), f"two turns ran at once: {turn_order}"
    assert {state for _, _, _, state in _executions(db)} == {"completed"}
    assert len(_executions(db)) == len(inputs)
    assert _queued(db) == []
