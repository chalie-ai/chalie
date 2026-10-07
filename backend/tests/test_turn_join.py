"""Feature tests: a message sent into a turn that is still working JOINS it.

The thread is one conversation. A user who adds to a question while the model is
still answering it must be heard by the running loop — not queued behind it, not
opened as a second competing execution. These tests drive the real send endpoint
(``POST /api/threads/<turn_id>``), the real ``MessageProcessor`` ACT loop, the
real transcript / turn-execution tables and the real turn serializer and act
trail, against the real migrated SQLite database.

The only substitution is the LLM network boundary
(``services.provider_service.build_client``, the seam every turn-driving test
uses): a scripted double that can hold one call open so a message can be sent
while the provider is "thinking". One test also holds a tool call open the same
way, wrapping the real dispatcher so the real tool still runs.

Assertions are on observable state — transcript rows, execution rows, the
settled flag, the model-facing request bodies, HTTP status — never on how many
times the provider was called: a completed user turn spawns best-effort
post-turn calls of its own on shared daemon threads, so that count is not the
turn's.
"""

import io
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

import services.provider_service as provider_service
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.thread_gist import ThreadGist
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.dispatch_service import DispatchService
from services.file_mapper_service import FileMapperService
from services.prompt_service import PromptService
from services.provider_api import ProviderApiRequest
from services.transcript_service import TranscriptService
from services.turn_serializer_service import get_service as _serializer
from tests.test_act_trail_exchange_scope import _assistant_row, _bare_mp, _input_row, _tool_call

pytestmark = pytest.mark.unit

_CHANNEL = "user"
_DRAIN_NAMES = ("skill-suggest", "thread-gist")
_DRAIN_PREFIXES = ("turn-", "memory-step-")

AuthedClient = tuple[FlaskClient, sqlite3.Connection, object]


# ── the provider double and the harness around it ────────────────────────────


class _Provider:
    """Replays scripted responses in order and records each request body.

    Past the script it answers with a benign terminal reply (the best-effort
    post-turn calls a finished turn spawns are not scripted for). ``hold_next``
    makes the next ``send`` park until ``release`` is set, signalling ``entered``
    first — the window in which a test sends a message into the working turn."""

    _TERMINAL = ProviderResponse(text="done", model="scripted-terminal", tool_calls=None)

    def __init__(self) -> None:
        self._queue: list[ProviderResponse] = []
        self._lock = threading.Lock()
        self._holding = False
        self.bodies: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def script(self, *responses: ProviderResponse) -> None:
        with self._lock:
            self._queue.extend(responses)

    def hold_next(self) -> None:
        self.entered.clear()
        self.release.clear()
        with self._lock:
            self._holding = True

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        with self._lock:
            self.bodies.append(str(dto.messages[0]["content"]))
            response = self._queue.pop(0) if self._queue else self._TERMINAL
            hold, self._holding = self._holding, False
        if hold:
            self.entered.set()
            assert self.release.wait(10), "the held provider call was never released"
        return response


def _drain(timeout_s: float = 10.0) -> None:
    """Join every daemon a turn spawns — the drive thread, the post-turn
    suggestion, the thread label and the memory step — so the turn has fully
    settled and nothing leaks into the next test."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in _DRAIN_NAMES or t.name.startswith(_DRAIN_PREFIXES)
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
    leftover = [t.name for t in threading.enumerate() if t.name in _DRAIN_NAMES or t.name.startswith(_DRAIN_PREFIXES)]
    assert not leftover, f"background turn threads never finished: {leftover}"


@pytest.fixture
def provider(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Provider]:
    double = _Provider()
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    yield double
    double.release.set()  # a failed test must not strand a parked drive thread
    _drain()


def _say(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="scripted", tool_calls=None)


def _post(client: FlaskClient, turn_id: int, text: str, **form: str) -> TestResponse:
    return client.post(f"/api/threads/{turn_id}", data={"text": text, **form})


def _result(resp: TestResponse) -> dict[str, object]:
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return cast("dict[str, object]", cast("dict[str, object]", resp.get_json())["result"])


def _rows(turn_id: int) -> list[tuple[object, object, object, object]]:
    """The turn's transcript as ``(role, content, settled, joined)``, in order."""
    return [
        (r["role"], r["content"], r["settled"], r["joined"])
        for r in Transcript.by_turn(_CHANNEL, turn_id)
    ]


def _answers(turn_id: int) -> list[tuple[object, object]]:
    """The turn's assistant rows as ``(content, settled)``, in order."""
    return [(content, settled) for role, content, settled, _ in _rows(turn_id) if role == "assistant"]


def _executions(db: sqlite3.Connection, turn_id: int) -> list[tuple[int, str]]:
    cur = db.execute(
        "SELECT id, state FROM turn_executions WHERE channel = ? AND turn_id = ? ORDER BY id",
        (_CHANNEL, turn_id),
    )
    return [(int(r["id"]), str(r["state"])) for r in cur.fetchall()]


def _label_thread(turn_id: int) -> None:
    """Give a thread its label up front. A reply into an unlabelled thread
    launches a label-writing call of its own at the same provider, which would
    race the reply's own call for the scripted responses."""
    ThreadGist(channel=_CHANNEL, turn_id=turn_id, gist="seeded label").upsert()


def _start_working_turn(client: FlaskClient, provider: _Provider, text: str) -> dict[str, object]:
    """Send a new thread and wait until its first provider call is in flight."""
    provider.hold_next()
    opened = _result(_post(client, -1, text))
    assert provider.entered.wait(10), "the turn never reached its provider call"
    return opened


def _finished_thread(client: FlaskClient, provider: _Provider, text: str, answer: str) -> dict[str, object]:
    """Send a new thread and run it to completion, labelled, so a later reply
    into it is the only caller of the provider."""
    provider.script(_say(answer))
    opened = _result(_post(client, -1, text))
    _drain()
    _label_thread(cast("int", opened["turn_id"]))
    return opened


# ── 1. joining ───────────────────────────────────────────────────────────────


def test_message_sent_into_a_working_turn_joins_it_instead_of_opening_a_second_execution(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """While the first question is still being answered, two more messages are
    sent into the thread — one plain, one as the thread's own follow-up
    (``join=1``). Both are written into the working turn and come back with the
    SAME execution: no second execution opens, and the two rows are marked
    joined, unread, until the loop takes them in."""
    client, db, _store = authed_client
    provider.script(_say("First answer."), _say("Second answer."), _say("Third answer."))
    opened = _start_working_turn(client, provider, "question")
    turn_id = cast("int", opened["turn_id"])

    plain = _result(_post(client, turn_id, "one more thing"))
    flagged = _result(_post(client, turn_id, "and another", join="1"))

    assert plain["id"] == opened["id"] == flagged["id"]
    assert plain["state"] == "working"
    assert _executions(db, turn_id) == [(cast("int", opened["id"]), "working")]
    assert _rows(turn_id) == [
        ("user", "question", 0, 0),
        ("user", "one more thing", 0, 1),
        ("user", "and another", 0, 1),
    ]

    provider.release.set()
    _drain()
    assert _executions(db, turn_id) == [(cast("int", opened["id"]), "completed")]


def test_a_join_that_lands_before_the_turn_settles_is_answered_in_the_same_turn(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """The user adds a thought while the model's text-only answer is on its way.
    That answer was written without having seen the addition, so it must not
    close the turn: it is kept as an unsettled interim answer, the loop reads
    the joined message and answers again, and only that second answer settles.
    The second request shows the model its own first answer and then the
    addition, in that order."""
    client, db, _store = authed_client
    provider.script(_say("First answer."), _say("Updated answer."))
    opened = _start_working_turn(client, provider, "question")
    turn_id = cast("int", opened["turn_id"])
    assert _result(_post(client, turn_id, "one more thing", join="1"))["id"] == opened["id"]

    provider.release.set()
    _drain()

    assert _answers(turn_id) == [("First answer.", 0), ("Updated answer.", 1)]
    assert ("user", "one more thing", 0, 1) in _rows(turn_id)
    assert _executions(db, turn_id) == [(cast("int", opened["id"]), "completed")]
    assert "one more thing" not in provider.bodies[0], "the join must have arrived mid-call, after request 0 was built"
    second = provider.bodies[1]
    assert second.index("First answer.") < second.index("one more thing")


# ── 2. a follow-up that must NOT join ────────────────────────────────────────


def test_follow_up_into_a_finished_turn_replies_and_never_joins(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """Once a turn has settled there is nothing to join. A plain message into it
    is a reply: a new execution on the same thread, its row a real input row
    (not joined). The thread's own follow-up (``join=1``) on a finished thread
    starts a NEW thread instead of replying, and leaves the finished thread's
    rows exactly as they were."""
    client, db, _store = authed_client
    opener = _finished_thread(client, provider, "question", "A1")
    turn_id = cast("int", opener["turn_id"])
    assert _executions(db, turn_id) == [(cast("int", opener["id"]), "completed")]

    provider.script(_say("Reply answer."))
    reply = _result(_post(client, turn_id, "follow up"))
    _drain()

    assert reply["turn_id"] == turn_id
    assert reply["id"] != opener["id"]
    assert ("user", "follow up", 0, 0) in _rows(turn_id)
    assert [state for _, state in _executions(db, turn_id)] == ["completed", "completed"]

    before = _rows(turn_id)
    moved = _result(_post(client, turn_id, "side question", join="1"))
    _drain()

    new_thread = cast("int", moved["turn_id"])
    assert new_thread != turn_id
    assert _rows(turn_id) == before
    assert ("user", "side question", 0, 0) in _rows(new_thread)


def test_follow_up_flag_starts_a_new_thread_while_only_a_reply_is_running(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """The thread's own follow-up belongs to the thread's first exchange. When
    that exchange is done and only a later reply is running, the follow-up must
    not be swallowed by the reply: it starts a new thread, and the running reply
    is left holding no joined message."""
    client, db, _store = authed_client
    opener = _finished_thread(client, provider, "question", "A1")
    turn_id = cast("int", opener["turn_id"])

    provider.hold_next()
    reply = _result(_post(client, turn_id, "reply into the thread"))
    assert provider.entered.wait(10)

    side = _result(_post(client, turn_id, "side question", join="1"))

    assert side["turn_id"] != turn_id
    assert [state for _, state in _executions(db, turn_id)] == ["completed", "working"]
    assert reply["id"] == _executions(db, turn_id)[-1][0]
    assert all(joined == 0 for _, _, _, joined in _rows(turn_id))
    provider.release.set()
    _drain()


# ── 3. stopping ──────────────────────────────────────────────────────────────


def test_stopping_a_turn_with_a_joined_message_keeps_both_visible_and_ends_it_cancelled(
    authed_client: AuthedClient, provider: _Provider,
) -> None:
    """The user sends an addition, then hits stop before the model answers. The
    turn ends cancelled (not crashed), stores no answer, and refetching the
    thread still shows the question AND the addition — a message the user sent
    into a turn they were watching must not vanish. A message sent after the
    stop is a reply, not another join into the dead turn."""
    client, db, _store = authed_client
    provider.script(_say("never stored"))
    opened = _start_working_turn(client, provider, "long job")
    turn_id = cast("int", opened["turn_id"])
    _result(_post(client, turn_id, "stranded"))

    stop = cast("dict[str, object]", client.delete(f"/api/threads/{turn_id}").get_json())
    assert cast("dict[str, object]", stop["result"])["cancelled"] is True
    provider.release.set()
    _drain()

    assert _executions(db, turn_id) == [(cast("int", opened["id"]), "cancelled")]
    assert _answers(turn_id) == []
    block = cast("dict[str, object]", client.get(f"/api/threads/{turn_id}").get_json())
    messages = cast("list[dict[str, object]]", cast("dict[str, object]", block["result"])["messages"])
    assert [m["content"] for m in messages] == ["long job", "stranded"]
    # The flag the interface reads to know a stop keeps the addition.
    assert [m["joined"] for m in messages] == [False, True]

    after = _result(_post(client, turn_id, "after the stop"))
    _drain()
    assert after["id"] != opened["id"]
    assert ("user", "after the stop", 0, 0) in _rows(turn_id)


# ── 4. a join during a tool call ─────────────────────────────────────────────


def test_a_message_sent_while_a_tool_runs_is_read_only_after_the_tool_returns(
    db: sqlite3.Connection, provider: _Provider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model calls a tool, and the user adds a message while that tool is
    still running. The loop must not read the message until the tool is done:
    the next request shows the tool's line first and the addition after it, the
    addition is answered by that very next call (no extra round), and the turn
    settles on that answer."""
    tool_entered, tool_release = threading.Event(), threading.Event()
    real_dispatch = DispatchService.dispatch

    def held_dispatch(self: DispatchService, tool_name: str, params: dict[str, object]) -> str:
        if tool_name == "noop_probe":
            tool_entered.set()
            assert tool_release.wait(10), "the held tool was never released"
        return real_dispatch(self, tool_name, params)

    monkeypatch.setattr(DispatchService, "dispatch", held_dispatch)
    provider.script(
        ProviderResponse(text="Looking.", model="scripted", tool_calls=[{"name": "noop_probe", "input": {"q": 1}}]),
        _say("Done, and noted."),
    )
    mp = MessageProcessor(UserConfig(), raw_input="check it")
    try:
        mp.begin()
        assert tool_entered.wait(10), "the turn never reached its tool call"
        joined = MessageProcessor(UserConfig(), mp.turn_id).join("also check Y", spine_only=False)
        assert joined is not None
        assert mp.consumed_joins == {}, "a message was read while the tool it arrived during was still running"
    finally:
        tool_release.set()
    mp.result()
    _drain()

    assert _answers(mp.turn_id) == [("Looking.", 0), ("Done, and noted.", 1)]
    second = provider.bodies[1]
    assert second.index("noop_probe") < second.index("also check Y")


# ── 5. files ─────────────────────────────────────────────────────────────────


def test_files_cannot_join_a_working_turn_but_start_a_reply_once_it_has_finished(
    authed_client: AuthedClient, provider: _Provider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file can only ever start an exchange. Sent into a turn that is still
    working it is refused (409) and writes nothing — no row, no stored file, no
    second execution. Sent into the same thread after it finished, it is an
    ordinary reply carrying its attachment."""
    client, db, _store = authed_client
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(FileMapperService, "get_file_index_db_path", lambda *_: tmp_path / "file_index.sqlite")

    def send_file(turn_id: int) -> TestResponse:
        return client.post(
            f"/api/threads/{turn_id}",
            data={"text": "see attached", "files": (io.BytesIO(b"quarterly numbers"), "note.txt")},
        )

    provider.script(_say("First answer."))
    opened = _start_working_turn(client, provider, "question")
    turn_id = cast("int", opened["turn_id"])

    assert send_file(turn_id).status_code == 409
    assert _rows(turn_id) == [("user", "question", 0, 0)]
    assert _executions(db, turn_id) == [(cast("int", opened["id"]), "working")]
    assert db.execute("SELECT COUNT(*) FROM transcript_files").fetchone()[0] == 0

    provider.release.set()
    _drain()
    _label_thread(turn_id)
    provider.script(_say("Got the file."))

    reply = send_file(turn_id)
    _drain()

    assert reply.status_code == 200
    assert cast("dict[str, object]", cast("dict[str, object]", reply.get_json())["result"])["id"] != opened["id"]
    assert ("user", "see attached", 0, 0) in _rows(turn_id)
    assert db.execute("SELECT COUNT(*) FROM transcript_files").fetchone()[0] == 1


# ── 6. what the user sees: the serialized thread ─────────────────────────────


@pytest.mark.usefixtures("db")
def test_a_joined_message_is_part_of_the_opener_not_the_start_of_a_reply() -> None:
    """A thread is the opener plus everything from the SECOND real user message
    on. A joined message was read by the opener's own exchange, so it must not
    count as that second message: the opener's answer stays an opener row, and
    only the genuine reply and its answer are tagged as thread replies."""
    turn_id = 7101
    opener = MessageProcessor(UserConfig(), turn_id, "question")  # only writes rows; never run
    opener_uid = opener.transcript_service.append_input(opener.raw_input)
    opener.uid = opener_uid
    opener.current_transcript_id = opener_uid
    joined_id = opener.transcript_service.append_input("one more thing", joined=True)
    answer_id = opener.transcript_service.append_assistant("Answer to both.", settled=True)

    reply = MessageProcessor(UserConfig(), turn_id, "a real reply")  # only writes rows; never run
    reply_uid = reply.transcript_service.append_input(reply.raw_input)
    reply.uid = reply_uid
    reply.current_transcript_id = reply_uid
    reply_answer_id = reply.transcript_service.append_assistant("Reply answer.", settled=True)

    messages = cast("list[dict[str, object]]", _serializer().serialize(_CHANNEL, turn_id)["messages"])
    tagged = {m["id"]: m.get("thread_message", False) for m in messages}

    assert tagged == {
        str(opener_uid): False,
        str(joined_id): False,
        str(answer_id): False,
        str(reply_uid): True,
        str(reply_answer_id): True,
    }


@pytest.mark.usefixtures("db")
def test_rich_card_in_the_answer_still_resolves_after_a_message_joined_mid_turn() -> None:
    """The model calls a rich-media tool, the user adds a message, and the
    answer cites the tool's card. The joined message continues the running
    exchange, so the card must still resolve to the call made BEFORE it arrived
    — it must not render as plain text."""
    turn_id = 7102
    mp = MessageProcessor(UserConfig(), turn_id, "Show me the northern lights")  # only writes rows; never run
    uid = mp.transcript_service.append_input(mp.raw_input)
    mp.uid = uid
    mp.current_transcript_id = uid
    step_id = mp.transcript_service.append_assistant("Fetching it.", settled=False)
    mp.current_transcript_id = step_id
    call_id = mp.tool_call_service.start("image_preview", {"file_path": "https://north.example/a.jpg"})
    assert call_id is not None
    payload = '{"url":"https://north.example/a.jpg","subtitle":"northern lights","alt":"northern lights"}'
    mp.tool_call_service.finish(
        call_id,
        f"[image_preview(status=success)]\n{payload}\n\nWrap it in <span id='image_preview_1'>x</span>.\n[end:image_preview]",
        ToolCall.DONE,
    )
    mp.transcript_service.append_input("make it brighter", joined=True)
    final_id = mp.transcript_service.append_assistant(
        "Here: <span id='image_preview_1'>Northern lights.</span>", settled=True,
    )

    messages = cast("list[dict[str, object]]", _serializer().serialize(_CHANNEL, turn_id)["messages"])
    final = next(m for m in messages if m["id"] == str(final_id))
    rich = [s for s in cast("list[dict[str, object]]", final["segments"]) if s.get("type") == "rich"]

    assert len(rich) == 1, final["segments"]
    assert cast("dict[str, object]", rich[0]["payload"])["subtitle"] == "northern lights"


# ── 7. what the model sees: the act trail ────────────────────────────────────


@pytest.mark.usefixtures("db")
def test_joined_messages_render_after_the_step_they_were_read_with_and_never_before_it() -> None:
    """The trail shows the model what it saw in the order it saw it: a message
    read after step one's tools appears after those tools and before step two's
    prose and call; a message read at the very end of the last step closes the
    trail; a message the loop has not read yet does not appear at all."""
    ch, turn = _CHANNEL, 1
    uid = _input_row(ch, turn, "question")
    a1 = _assistant_row(ch, turn, "Checking both.")
    _tool_call(a1, "weather", {"loc": "here"}, "24C here")
    _tool_call(a1, "weather", {"loc": "Sicily"}, "26C Sicily")
    _joined_row(ch, turn, "first addition")
    a2 = _assistant_row(ch, turn, "Now the forecast.")
    _tool_call(a2, "forecast", {"days": 3}, "sunny")
    _joined_row(ch, turn, "second addition")

    read_both = _bare_mp(ch, turn, uid=uid)
    joined = TranscriptService(read_both).joined_since(uid)
    assert [r.content for r in joined] == ["first addition", "second addition"]
    read_both.consumed_joins = {a1: [joined[0]], a2: [joined[1]]}
    lines = PromptService(read_both).act_trail().split("\n")

    def at(fragment: str) -> int:
        hits = [i for i, line in enumerate(lines) if fragment in line]
        assert len(hits) == 1, f"{fragment!r} should appear exactly once: {lines}"
        return hits[0]

    assert at("here") < at("Sicily") < at("user: first addition") < at("Now the forecast.") < at("forecast]")
    assert at("forecast]") < at("user: second addition") == len(lines) - 1
    assert lines[at("user: first addition")].startswith("[")

    read_first_only = _bare_mp(ch, turn, uid=uid)
    read_first_only.consumed_joins = {a1: [joined[0]]}
    trail = PromptService(read_first_only).act_trail()
    assert "user: first addition" in trail
    assert "second addition" not in trail


@pytest.mark.usefixtures("db")
def test_a_trail_with_no_joined_messages_renders_exactly_the_calls_and_interim_prose() -> None:
    """With nothing joined the trail is the plain record of the exchange: each
    step's interim prose once — not once per call it anchored — followed by its
    calls, and the final answer (no calls) adds nothing."""
    ch, turn = _CHANNEL, 1
    uid = _input_row(ch, turn, "question")
    a1 = _assistant_row(ch, turn, "Checking both places.")
    _tool_call(a1, "weather", {"loc": "here"}, "24C here")
    _tool_call(a1, "weather", {"loc": "Sicily"}, "26C Sicily")
    a2 = _assistant_row(ch, turn, "Now the forecast.")
    _tool_call(a2, "forecast", {"days": 3}, "sunny")
    _assistant_row(ch, turn, "It will be sunny.")

    trail = PromptService(_bare_mp(ch, turn, uid=uid)).act_trail()

    assert trail == "\n".join([
        "[interim_response] Checking both places.",
        '[weather] {"loc": "here"} → 24C here',
        '[weather] {"loc": "Sicily"} → 26C Sicily',
        "[interim_response] Now the forecast.",
        '[forecast] {"days": 3} → sunny',
    ])


def _joined_row(channel: str, turn_id: int, content: str) -> int:
    return cast("int", Transcript(
        channel=channel, role="user", content=content, turn_id=turn_id,
        settled=0, joined=1, xml_migrated=1, deliberation_score=0.0,
    ).save().id)
