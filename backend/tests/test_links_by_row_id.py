"""Feature tests: the chat-side links to a thread are found by transcript row
id, and a spine checkpoint never hides an exchange it did not fold.

A turn's execution records its input row, a thread's label is stored against
its opening row, a thread checkpoint is scoped to its opening row, and a spine
checkpoint records the last row it folded. These tests prove each link with
more than one thread on the channel: a follow-up joins the execution opened
for its own thread's opening row and never a reply running in a thread, each
label shows on its own thread only, each thread checkpoint stays with its own
thread, and a spine checkpoint hides only the rows it folded, so an older
thread's answer written after it still shows.

The last two drive a real turn through the real recovery path: the provider
rejects the turn's first request for length, the turn compacts its spine and
retries. While that first request is out, another thread writes a row. The
spine checkpoint the compaction writes must not cover the working turn's own
rows, which it never folded, whichever side of the working turn the other
thread sits on.

The only substitution is the LLM network boundary, the same scripted double the
join tests use.
"""

import sqlite3
from collections.abc import Callable, Iterator
from typing import cast

import pytest
from flask.testing import FlaskClient

import services.provider_service as provider_service
from abilities.chat_history_compactor import ChatHistoryCompactor
from configs.channels.user import UserConfig
from contracts.params.chat_history_compactor_params_bag import ChatHistoryCompactorParamsBag
from controllers.message_processor import MessageProcessor
from exceptions import ContextLimit
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from services.provider_api import ProviderApiRequest
from services.thread_gist_message_processor import maybe_ingest_gist
from tests._tool_result_harness import built
from tests.test_turn_join import (
    _drain,
    _executions,
    _finished_thread,
    _post,
    _Provider,
    _result,
    _rows,
    _say,
    _start_working_turn,
)

pytestmark = pytest.mark.unit

_CHANNEL = "user"

AuthedClient = tuple[FlaskClient, sqlite3.Connection, object]


@pytest.fixture
def provider(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Provider]:
    double = _Provider()
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    yield double
    double.release.set()  # a failed test must not strand a parked drive thread
    _drain()


def _settled_turn(user_text: str, answer: str) -> int:
    """One settled exchange on a new turn: the input row and its settled answer."""
    turn_id = Transcript.next_turn_id(None, _CHANNEL)
    Transcript(channel=_CHANNEL, role="user", content=user_text, turn_id=turn_id, settled=0, xml_migrated=1).save()
    Transcript(channel=_CHANNEL, role="assistant", content=answer, turn_id=turn_id, settled=1, xml_migrated=1).save()
    return turn_id


def _compact(mp: MessageProcessor, provider: _Provider, summary: str) -> None:
    """Fold ``mp``'s history view into a checkpoint through the real compactor,
    the model writing ``summary``."""
    provider.script(_say(summary))
    assert ChatHistoryCompactor(mp).run(built(ChatHistoryCompactorParamsBag.from_params({}))).status == "success"
    _drain()


def _labelled_thread(client: FlaskClient, provider: _Provider, question: str, label: str) -> int:
    """Open a thread through the send endpoint and run it to the end, then run
    the real label step on it, the model writing ``label``."""
    provider.script(_say("Happy to help."))
    turn_id = cast("int", _result(_post(client, -1, question))["turn_id"])
    _drain()
    provider.script(_say(label))
    maybe_ingest_gist(_CHANNEL, turn_id, UserConfig().type_value())
    _drain()
    return turn_id


def _feed(client: FlaskClient, query: str = "") -> dict[int, object]:
    """The thread feed as ``{turn_id: label}``."""
    resp = client.get("/api/threads/all", query_string={"type": "user", "q": query})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    threads = cast("list[dict[str, object]]", cast("dict[str, object]", resp.get_json())["result"])
    return {cast("int", t["turn_id"]): t["gist"] for t in threads}


def _history(mp: MessageProcessor) -> list[str]:
    return [r.content for r in mp.transcript_service.read()]


def test_a_follow_up_joins_the_execution_opened_for_its_threads_opening_row(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """Two executions are working at once: a reply into a finished first
    thread, and a second thread's opening exchange. The second thread's own
    follow-up (``join=1``) joins the execution opened for that thread's opening
    row. The first thread's follow-up finds only the execution opened for the
    reply's row, so it starts a new thread and the reply is left holding no
    joined message."""
    client, db, _store = authed_client
    first = cast("int", _finished_thread(client, provider, "first question", "First answer.")["turn_id"])
    provider.script(_say("Reply answer."), _say("Second answer."))
    provider.hold_next()
    _result(_post(client, first, "reply into the first thread"))
    assert provider.entered.wait(10), "the reply never reached its provider call"
    opened = _start_working_turn(client, provider, "second question")
    second = cast("int", opened["turn_id"])

    joined = _result(_post(client, second, "and another", join="1"))
    side = _result(_post(client, first, "side question", join="1"))

    assert joined["id"] == opened["id"]
    assert _executions(db, second) == [(cast("int", opened["id"]), "working")]
    assert ("user", "and another", 0, 1) in _rows(second)
    assert side["turn_id"] not in (first, second)
    assert [state for _, state in _executions(db, first)] == ["completed", "working"]
    assert all(flag == 0 for _, _, _, flag in _rows(first))
    provider.release.set()
    _drain()


def test_each_thread_label_belongs_to_its_own_opening_row(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """Two threads on the channel, each labelled by the real label step. The
    feed shows each label on its own thread only, and a search for one
    thread's label finds that thread and not the other."""
    client, _db, _store = authed_client
    trip = _labelled_thread(client, provider, "plan the trip", "Lisbon Trip Plans")
    dinner = _labelled_thread(client, provider, "what should we cook tonight", "Dinner Ideas")

    assert _feed(client) == {trip: "Lisbon Trip Plans", dinner: "Dinner Ideas"}
    assert _feed(client, "Lisbon") == {trip: "Lisbon Trip Plans"}
    assert _feed(client, "Dinner") == {dinner: "Dinner Ideas"}


def test_a_spine_checkpoint_hides_only_the_rows_it_folded(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """A checkpoint of the spine records the last row it folded. An older
    thread is still waiting on its answer when the spine compacts past a newer
    thread, so the checkpoint folds both. The older thread's answer, written
    after the checkpoint, is a later row than any it folded, so the next spine
    read shows it even though its turn sits below the newer thread's."""
    assert db is not None
    older = Transcript.next_turn_id(None, _CHANNEL)
    Transcript(channel=_CHANNEL, role="user", content="older question", turn_id=older, settled=0, xml_migrated=1).save()
    _settled_turn("newer question", "newer answer")
    _compact(MessageProcessor(UserConfig(), raw_input="compact"), provider, "Summary of the earlier turns.")
    assert _history(MessageProcessor(UserConfig(), raw_input="next")) == []

    Transcript(channel=_CHANNEL, role="assistant", content="older answer", turn_id=older, settled=1, xml_migrated=1).save()

    assert _history(MessageProcessor(UserConfig(), raw_input="next")) == ["older answer"]


def test_each_thread_checkpoint_stays_with_its_own_thread(
    db: sqlite3.Connection, provider: _Provider,
) -> None:
    """Two threads each compact through the real compactor. A reply into
    either reads its own thread's checkpoint and not the other's, and a reply
    into a third thread that never compacted reads none and sees its own rows
    whole."""
    assert db is not None
    first = _settled_turn("first question", "first answer")
    second = _settled_turn("second question", "second answer")
    third = _settled_turn("third question", "third answer")
    _compact(MessageProcessor(UserConfig(), turn_id=first, raw_input="reply"), provider, "First thread summary.")
    _compact(MessageProcessor(UserConfig(), turn_id=second, raw_input="reply"), provider, "Second thread summary.")

    into_first = MessageProcessor(UserConfig(), turn_id=first, raw_input="reply")
    body = into_first.compaction_service.checkpoint("body")
    assert "First thread summary." in body and "Second thread summary." not in body
    assert _history(into_first) == []
    into_second = MessageProcessor(UserConfig(), turn_id=second, raw_input="reply")
    body = into_second.compaction_service.checkpoint("body")
    assert "Second thread summary." in body and "First thread summary." not in body
    assert _history(into_second) == []
    into_third = MessageProcessor(UserConfig(), turn_id=third, raw_input="reply")
    assert into_third.compaction_service.get() is None
    assert into_third.compaction_service.checkpoint("body") == "body"
    assert _history(into_third) == ["third question", "third answer"]


# ── a spine compaction while another thread writes ───────────────────────────


class _TooLongOnce(_Provider):
    """The scripted double, except that its first request is rejected for
    length, the way a real provider rejects an oversized payload. Just before
    rejecting it, ``meanwhile`` runs: another thread writing its row while this
    turn's request is out."""

    def __init__(self, meanwhile: Callable[[], None]) -> None:
        super().__init__()
        self._meanwhile = meanwhile
        self._rejected = False

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        if not self._rejected:
            self._rejected = True
            self._meanwhile()
            raise ContextLimit("payload exceeds the model's maximum context length")
        return super().send(dto)


def _run_compacting_turn(monkeypatch: pytest.MonkeyPatch, meanwhile: Callable[[], None]) -> MessageProcessor:
    """Send "working question" as a new thread and drive it to the end. Its
    first request is rejected for length, so it writes its hand-over, compacts
    its spine and retries: the compaction's model writes "Summary of the
    backlog." and the retry answers "working answer"."""
    double = _TooLongOnce(meanwhile)
    double.script(_say("Hand-over note."), _say("Summary of the backlog."), _say("working answer"))
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    mp = MessageProcessor(UserConfig(), raw_input="working question")
    mp.begin()
    mp.result()
    _drain()
    execution = mp.turn_execution_service.latest_for_turn()
    assert execution is not None and execution.state == "completed"
    return mp


def _assert_spine_keeps_the_working_exchange() -> None:
    nxt = MessageProcessor(UserConfig(), raw_input="next")
    checkpoint = nxt.compaction_service.get()
    assert checkpoint is not None and checkpoint.content == "Summary of the backlog.", "the turn never compacted"
    history = _history(nxt)
    assert "working question" in history
    assert "working answer" in history


def test_an_older_thread_writing_during_a_spine_compaction_does_not_hide_the_working_turn(
    chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older thread is still working when a new one opens, and writes a row
    while the new one's first request is out. The new turn's compaction folds
    the older thread, row and all, but none of its own rows, so the next read
    of the spine still shows the new turn's question and answer."""
    assert chat_provider is not None
    older = Transcript.next_turn_id(None, _CHANNEL)
    Transcript(channel=_CHANNEL, role="user", content="older question", turn_id=older, settled=0, xml_migrated=1).save()

    def older_thread_answers() -> None:
        Transcript(
            channel=_CHANNEL, role="assistant", content="older interim answer", turn_id=older, settled=0, xml_migrated=1,
        ).save()

    mp = _run_compacting_turn(monkeypatch, older_thread_answers)

    assert mp.turn_id > older
    _assert_spine_keeps_the_working_exchange()


def test_a_newer_thread_opening_during_a_spine_compaction_does_not_hide_the_working_turn(
    chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer thread opens while the working turn's first request is out. The
    working turn's compaction folds the newer thread but none of its own rows,
    so the next read of the spine still shows its question and answer."""
    assert chat_provider is not None
    newer: list[int] = []

    def newer_thread_opens() -> None:
        newer.append(Transcript.next_turn_id(None, _CHANNEL))
        Transcript(
            channel=_CHANNEL, role="user", content="newer question", turn_id=newer[0], settled=0, xml_migrated=1,
        ).save()

    mp = _run_compacting_turn(monkeypatch, newer_thread_opens)

    assert newer == [mp.turn_id + 1]
    _assert_spine_keeps_the_working_exchange()
