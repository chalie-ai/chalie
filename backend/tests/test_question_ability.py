"""Feature tests: the model asks the user one question and waits for the answer.

The chat's model calls ``question`` with 2 to 4 answers. The question opens as
an ask of its own kind on the asks every screen hears and lists — no permission
ask comes first, since the seeded policy allows it — and the run waits. The user
answers from their signed-in session: one answer, several, or their own text.
An answer that does not fit the question is refused and the question keeps
waiting. A question the user skips, or a stopped run, closes it unanswered,
and the model reads that it went unanswered. A call the model gets
wrong opens nothing and tells the model what to fix. Neither the question nor
the answer is ever written to a log line.

Real Flask app, real cookie session, real socket registry broadcast, real turn
loop, dispatcher and policy gate on the real database; the model provider is the
one stand-in, at its network boundary.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterator
from typing import NamedTuple, cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.question import QuestionAbility
from configs.channels.user import UserConfig
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from services.ask_user import AskUser
from tests._browser_live import signed_in
from tests._delegates import (
    USER,
    WAIT_S,
    ChannelProvider,
    RecordingClient,
    begin,
    calls,
    calls_of,
    execution,
    listening,
    patched,
    text,
    tool,
    wait,
)

pytestmark = pytest.mark.unit

_QUESTION = "Which day suits you?"
_ANSWERS: list[dict[str, object]] = [{"label": "Tuesday", "description": "Morning only"}, {"label": "Friday"}]
_SENT: list[dict[str, object]] = [{"label": "Tuesday", "description": "Morning only"}, {"label": "Friday", "description": ""}]
_THREE: list[dict[str, object]] = [*_ANSWERS, {"label": "Sunday", "description": "After lunch"}]
_REFUSED = "A question ask cannot take that value"
_DONE = "Noted."


@pytest.fixture
def app(db: sqlite3.Connection) -> Flask:
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


@pytest.fixture(autouse=True)
def no_ask_left_open() -> Iterator[None]:
    """Asks live in the process, not the database: each test starts with none
    open and leaves none open, so one test's question never answers another's."""
    assert AskUser.open_asks() == [], "an earlier test left an ask open"
    yield
    left = AskUser.open_asks()
    for ask in left:
        AskUser.cancel(ask.ask_id)
    assert left == [], "the test left an ask open"


def _asking(**params: object) -> dict[str, object]:
    """The model's call of the question tool with *params*."""
    return tool(QuestionAbility.NAME, **params)


def _listing(client: FlaskClient) -> list[dict[str, object]]:
    """The open asks, as a screen that has just connected reads them."""
    response = client.get("/api/asks/open")
    assert response.status_code == 200, response.data
    return cast("list[dict[str, object]]", cast("dict[str, object]", response.get_json())["result"])


def _reply(client: FlaskClient, ask_id: str, value: object) -> tuple[int, object]:
    """The user answers ask *ask_id* with *value*; the reply's status and error text."""
    response = client.post(f"/api/asks/reply/{ask_id}", json={"value": value})
    body = response.get_json(silent=True)
    return response.status_code, body.get("error") if isinstance(body, dict) else None


def _of_type(ui: RecordingClient, kind: str) -> list[dict[str, object]]:
    return [frame for frame in ui.frames if frame.get("type") == kind]


def _got(tool_result: str) -> str:
    """A successful question result as the model reads it."""
    return f"[{QuestionAbility.NAME}(status=success)]\n{tool_result}\n[end:{QuestionAbility.NAME}]"


class _Asks:
    """The ids of the asks broadcast while it listens, in order."""

    def __init__(self) -> None:
        self._opened: queue.Queue[str] = queue.Queue()

    def note(self, frame: dict[str, object]) -> None:
        if frame.get("type") == "ask_opened":
            self._opened.put(cast("str", frame["ask_id"]))

    def next(self) -> str:
        return self._opened.get(timeout=WAIT_S)


class _Asked(NamedTuple):
    """One turn in which the model asked a question and the user answered it."""

    opened: list[dict[str, object]]
    closed: list[dict[str, object]]
    listing: list[dict[str, object]]  # the open asks while the question waited
    read: list[str]  # what the chat's model read on each of its calls
    turn_id: int


def _ask_and_answer(
    client: FlaskClient, params: dict[str, object], answer: Callable[[str], None]
) -> _Asked:
    """Run a chat turn whose model asks the question *params* and then says it is
    done; *answer* answers the question by its ask id."""
    asks = _Asks()
    provider = ChannelProvider(chat=[calls(_asking(**params)), text(_DONE)])
    with listening(asks.note) as ui, patched(provider):
        run = begin(UserConfig(), "help me pick a day")
        ask_id = asks.next()
        listing = _listing(client)
        try:
            answer(ask_id)
        except BaseException:
            AskUser.cancel(ask_id)  # let the run end so the failure surfaces at once
            raise
    # Read once the block has drained every turn: a question left waiting
    # fails that drain within its deadline instead of hanging the suite.
    run.result()
    assert provider.unused() == {}
    return _Asked(_of_type(ui, "ask_opened"), _of_type(ui, "ask_closed"), listing, provider.read("chat"), run.turn_id)


def test_a_picked_answer_reaches_the_model_after_the_question_shows_on_every_screen(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    statuses: list[tuple[int, object]] = []
    asked = _ask_and_answer(
        client,
        {"question": _QUESTION, "answers": _ANSWERS},
        lambda ask_id: statuses.append(_reply(client, ask_id, {"labels": ["Tuesday"]})),
    )

    # The seeded policy allows the tool: the question is the only ask.
    [opened] = asked.opened
    assert opened["kind"] == "question"
    assert opened["data"] == {
        "question": _QUESTION,
        "answers": _SENT,
        "multi_select": False,
        "origin": {"type": "user", "turn_id": asked.turn_id, "forked": False},
    }
    # A screen that connected while it waited rebuilt the same ask.
    assert asked.listing == [{k: v for k, v in opened.items() if k != "type"}]
    assert statuses == [(204, None)]
    assert asked.closed == [{"type": "ask_closed", "ask_id": opened["ask_id"]}]
    assert _listing(client) == []

    first, after = asked.read
    assert f"[{QuestionAbility.NAME}(" not in first
    assert _got('{"answer":"Tuesday"}') in after
    [call] = calls_of(chat_provider, QuestionAbility.NAME)
    assert call["state"] == ToolCall.DONE, call["result"]
    assert execution(USER, asked.turn_id).state == TurnExecution.COMPLETED


def test_several_picked_answers_reach_the_model_in_the_order_the_user_sent_them(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    statuses: list[tuple[int, object]] = []
    asked = _ask_and_answer(
        client,
        {"question": _QUESTION, "answers": _THREE, "multi_select": True},
        # Neither the order offered nor alphabetical: the order the user picked.
        lambda ask_id: statuses.append(_reply(client, ask_id, {"labels": ["Sunday", "Friday"]})),
    )

    [opened] = asked.opened
    assert cast("dict[str, object]", opened["data"])["multi_select"] is True
    assert statuses == [(204, None)]
    assert _got('{"answers":["Sunday","Friday"]}') in asked.read[1]


def test_the_users_own_answer_reaches_the_model_trimmed_and_is_never_logged(
    app: Flask, chat_provider: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client, _token = signed_in(app)
    own = "Any evening next week"
    statuses: list[tuple[int, object]] = []
    asked = _ask_and_answer(
        client,
        # Their own text wins whatever the mode.
        {"question": _QUESTION, "answers": _ANSWERS, "multi_select": True},
        lambda ask_id: statuses.append(_reply(client, ask_id, {"other": f"  {own} \n"})),
    )

    assert statuses == [(204, None)]
    assert _got(f'{{"other":"{own}"}}') in asked.read[1]
    lines = [logging.Formatter().format(record) for record in caplog.records]
    assert [line for line in lines if own in line or _QUESTION in line] == []


def test_an_answer_that_does_not_fit_the_question_is_refused_and_the_question_still_waits(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    asks = _Asks()
    provider = ChannelProvider(chat=[
        calls(_asking(question=_QUESTION, answers=_ANSWERS)),
        calls(_asking(question=_QUESTION, answers=_THREE, multi_select=True)),
        text(_DONE),
    ])
    with listening(asks.note) as ui, patched(provider):
        run = begin(UserConfig(), "help me pick a day")

        single = asks.next()
        try:
            refused = [
                _reply(client, single, {"labels": ["Monday"]}),  # not one of its answers
                _reply(client, single, {"labels": ["Tuesday", "Friday"]}),  # one answer only
                _reply(client, single, {"other": " \n "}),  # blank text of their own
            ]
            misshapen = [
                _reply(client, single, {"labels": ["Tuesday"], "other": "Friday"})[0],
                _reply(client, single, "Tuesday")[0],
                _reply(client, single, {"labels": []})[0],
            ]
            still = [ask["ask_id"] for ask in _listing(client)]
            closed_early = _of_type(ui, "ask_closed")
            took = _reply(client, single, {"labels": ["Friday"]})
        except BaseException:
            AskUser.cancel(single)
            raise

        several = asks.next()
        try:
            refused_several = [
                _reply(client, several, {"labels": ["Tuesday", "Tuesday"]}),  # the same answer twice
                _reply(client, several, {"labels": ["Sunday", "Monday"]}),  # one not among its answers
            ]
            still_several = [ask["ask_id"] for ask in _listing(client)]
            took_several = _reply(client, several, {"labels": ["Sunday", "Friday"]})
        except BaseException:
            AskUser.cancel(several)
            raise
    run.result()

    assert refused == [(400, _REFUSED)] * 3
    assert misshapen == [422, 422, 422]
    assert (still, closed_early, took) == ([single], [], (204, None))
    assert refused_several == [(400, _REFUSED)] * 2
    assert (still_several, took_several) == ([several], (204, None))
    assert provider.unused() == {}
    after_single, after_several = provider.read("chat")[1:]
    assert _got('{"answer":"Friday"}') in after_single
    assert _got('{"answers":["Sunday","Friday"]}') in after_several
    assert execution(USER, run.turn_id).state == TurnExecution.COMPLETED


def test_a_run_stopped_while_the_question_waits_closes_it_and_the_model_hears_it_went_unanswered(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    asked = threading.Event()
    closed = threading.Event()

    def on_frame(frame: dict[str, object]) -> None:
        if frame["type"] == "ask_opened":
            asked.set()
        elif frame["type"] == "ask_closed":
            closed.set()

    provider = ChannelProvider(chat=[calls(_asking(question=_QUESTION, answers=_ANSWERS))])
    with listening(on_frame) as ui, patched(provider):
        run = begin(UserConfig(), "help me pick a day")
        wait(asked, "the model to ask its question")
        stop = client.delete(f"/api/threads/{run.turn_id}")
        wait(closed, "the stopped run to close its question")
        run.result()

    assert stop.status_code == 200 and cast("dict[str, dict[str, object]]", stop.get_json())["result"]["cancelled"] is True
    [opened] = _of_type(ui, "ask_opened")
    assert _of_type(ui, "ask_closed") == [{"type": "ask_closed", "ask_id": opened["ask_id"]}]
    assert _listing(client) == []
    # The call's stored result — what the transcript shows for it.
    assert [(r["state"], r["result"]) for r in calls_of(chat_provider, QuestionAbility.NAME)] == [(
        ToolCall.ERROR,
        f"[{QuestionAbility.NAME}(status=error, code=question-unanswered)]\n"
        f"Failed - the question closed before the user answered.\n[end:{QuestionAbility.NAME}]",
    )]
    # The stopped run never calls the model again.
    assert len(provider.read("chat")) == 1
    assert execution(USER, run.turn_id).state == TurnExecution.CANCELLED


def test_a_question_the_user_skips_closes_unanswered_and_the_model_reads_exactly_that(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    statuses: list[int] = []

    def skip(ask_id: str) -> None:
        statuses.append(client.post(f"/api/asks/dismiss/{ask_id}").status_code)
        # A second screen skipping the same question finds it already closed.
        statuses.append(client.post(f"/api/asks/dismiss/{ask_id}").status_code)

    asked = _ask_and_answer(client, {"question": _QUESTION, "answers": _ANSWERS}, skip)

    [opened] = asked.opened
    assert statuses == [204, 404]
    assert asked.closed == [{"type": "ask_closed", "ask_id": opened["ask_id"]}]
    assert _listing(client) == []
    unanswered = (
        f"[{QuestionAbility.NAME}(status=error, code=question-unanswered)]\n"
        f"Failed - the question closed before the user answered.\n[end:{QuestionAbility.NAME}]"
    )
    assert unanswered in asked.read[1]
    assert [(r["state"], r["result"]) for r in calls_of(chat_provider, QuestionAbility.NAME)] == [(ToolCall.ERROR, unanswered)]
    # The run carries on past the skipped question to the model's reply.
    assert execution(USER, asked.turn_id).state == TurnExecution.COMPLETED


@pytest.mark.parametrize(
    ("answers", "told"),
    [
        (
            [{"label": "Tuesday"}],
            "'answers' must list 2 to 4 answers.\n"
            "hint: pass 'answers' as a list of 2 to 4 {\"label\": ..., \"description\": ...} objects.",
        ),
        (
            [*_THREE, {"label": "Monday"}, {"label": "Wednesday"}],
            "'answers' must list 2 to 4 answers.\n"
            "hint: pass 'answers' as a list of 2 to 4 {\"label\": ..., \"description\": ...} objects.",
        ),
        (
            [{"label": "Tuesday"}, {"description": "Any time"}],
            "'answers' item at index 1 needs a 'label'.\nhint: give each answer a non-empty 'label' string.",
        ),
        (
            [{"label": "Tuesday"}, {"label": "Friday", "description": 3}],
            "'answers' item at index 1: 'description' must be text.\n"
            "hint: pass 'description' as a string, or leave it out.",
        ),
        (
            [{"label": "Tuesday"}, {"label": "Friday"}, {"label": "Tuesday"}],
            "'answers' has the label 'Tuesday' twice.\nhint: give each answer a different label.",
        ),
    ],
    ids=["one-answer", "five-answers", "no-label", "description-not-text", "label-twice"],
)
def test_a_question_the_model_got_wrong_opens_nothing_and_tells_the_model_what_to_fix(
    app: Flask, chat_provider: sqlite3.Connection, answers: list[dict[str, object]], told: str
) -> None:
    provider = ChannelProvider(chat=[calls(_asking(question=_QUESTION, answers=answers)), text(_DONE)])
    with listening() as ui, patched(provider):
        run = begin(UserConfig(), "help me pick a day")
    run.result()

    assert provider.unused() == {}
    assert _of_type(ui, "ask_opened") == []
    error = f"[{QuestionAbility.NAME}(status=error, code=invalid-param)]\n{told}\n[end:{QuestionAbility.NAME}]"
    assert error in provider.read("chat")[1]
    assert [r["state"] for r in calls_of(chat_provider, QuestionAbility.NAME)] == [ToolCall.ERROR]
