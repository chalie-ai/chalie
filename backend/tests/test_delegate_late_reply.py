"""Feature test: a subagent the person's chat calls answers later, as a reply.

When the person's chat calls a subagent that runs a turn of its own
(``web_search``, ``web_browse``, ``code_agent``, ``pim``), the call does not
keep the chat waiting: it returns at once with the subagent's number and the
label its answer will arrive under. Once the subagent's turn ends, its outcome
comes back to the chat as a reply under that label, and the chat answers it.
The reply goes where the call was made — a new turn for a call from the
thread's opening exchange, the thread itself for a call from a thread reply,
the very turn that made the call while that turn is still working — and never
into some other turn that happens to be working.

The reply holds exactly what the subagent said: its answer, or a one-line notice
when it finished with nothing to say, was stopped, crashed, or was cut off by a
restart. The warning that a subagent's answer is third-party content, and the
next step the subagent suggests, reach the model only while it answers that
reply — directly after the answer — never in what the person reads, never in a
later exchange, and never on a notice. The chat is offered the subagent on
every request; the turn that answers the reply is offered, besides, the tools
the subagent's next step names. A subagent's own turn is never offered itself.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``), the real boot sweep and the real Flask routes against the real,
fully-migrated SQLite DB. The only substitution is the LLM network boundary
(``services.provider_service.build_client``); each subagent is held inside its
own model call on a ``threading.Event`` so it finishes exactly when the test
says. ``pim`` is connected to a mailbox through the product's own setup route,
against an IMAP peer on loopback.
"""

import sqlite3
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities._delegate import DelegateAbility
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from abilities.code_agent import CodeAgentAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_fetch import WebFetchAbility
from configs.channels import config_for
from configs.channels.user import UserConfig
from configs.enums.config_type import ConfigTypeEnum
from contracts.params.delegate_params_bag import DelegateParamsBag
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from models.turn_execution import TurnExecution
from run import _reply_to_swept_delegates
from tests._delegates import (
    USER,
    WAIT_S,
    ChannelProvider,
    Entry,
    Held,
    agent_number,
    assistant_texts,
    begin,
    block,
    calls,
    calls_of,
    child_turn,
    connected_mail,
    delegate_channel,
    dies,
    drain,
    execution,
    messages,
    patched,
    process_deaths_expected,
    raises,
    replies,
    run_turn,
    text,
    tool,
    wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_QUESTION = "what is the capital of France?"
_INSTRUCTIONS = "find the capital of France"
_ANSWER = "Paris is the capital of France."
_STARTED = "I have started it."
_ANSWERED = "Here is what it found."
_BLOCK_OPENER = "[follow_up_instruction]"
_BLOCK_CLOSER = "[end:follow_up_instruction]"
_EXHAUSTED = "The AI provider failed to respond after several attempts. Please try again in a moment."

_Client = tuple[FlaskClient, sqlite3.Connection, object]


def _user_turns(db: sqlite3.Connection) -> list[int]:
    """Every turn the person's chat holds rows in."""
    return [r[0] for r in db.execute("SELECT DISTINCT turn_id FROM transcript WHERE channel = ? ORDER BY 1", (USER,))]


def _held_threads_end(*held: Held) -> None:
    """Wait for each released subagent's own turn to end — the turn that sends
    its reply on the way out."""
    for h in held:
        assert h.thread is not None, "the subagent never reached its model call"
        h.thread.join(timeout=WAIT_S)
        assert not h.thread.is_alive(), f"the subagent's turn {h.thread.name} did not end"


def _mailbox(client: FlaskClient, delegate: str) -> AbstractContextManager[None]:
    """A connected mailbox for ``pim``, which refuses without one; nothing for
    any other subagent."""
    return connected_mail(client) if delegate == "pim" else nullcontext()


def _after_reply(request: str, label: str, answer: str) -> str:
    """What the model read right after a reply's *answer* in *request*. The
    answer appears once, directly after its *label* — nothing between the two
    but a separator."""
    assert request.count(answer) == 1, request[-1500:]
    at = request.index(answer)
    label_at = request.rfind(label, 0, at)
    assert label_at >= 0, f"no {label!r} before the answer: {request[-1500:]}"
    assert not request[label_at + len(label):at].strip(" :\n"), request[label_at:at + len(answer)]
    return request[at + len(answer):]


def _block_opening(tail: str) -> str | None:
    """The follow-up block *tail* opens with, on the line right after the
    answer — or ``None`` when it opens with anything else."""
    if not tail.startswith(f"\n{_BLOCK_OPENER}\n"):
        return None
    return tail[:tail.index(_BLOCK_CLOSER) + len(_BLOCK_CLOSER)]


@pytest.mark.parametrize("delegate", ["web_search", "web_browse", "code_agent", "pim"])
def test_a_subagent_called_from_the_chat_returns_at_once_and_its_answer_comes_back_as_a_reply(
    authed_client: _Client, delegate: str,
) -> None:
    """The call ends at once saying the subagent was spawned, under its number
    and the label its answer will arrive under, with no answer and no warning
    in it. Once the
    chat's turn has ended, the subagent finishes and its answer comes back as one
    reply that opens a turn of its own, holding exactly the answer — which is
    also all the person reads under that label. The model answering it reads
    the label, then the answer, then — directly after it — the
    third-party-content warning and suggested next step the subagent carries
    (``code_agent`` carries none). The subagent is on offer on every chat
    request; the tools its next step names are on offer only while the chat
    answers the reply. A later, unrelated exchange still reads the label and
    the answer in its history, with neither the warning nor the next step."""
    client, db, _store = authed_client
    ability = cast("DelegateAbility[DelegateParamsBag]", AbilityRegistry.get(delegate))
    follow_up_tools = set(ability.FOLLOW_UP_TOOLS)
    steer = ability.UNTRUSTED_CONTENT.get("", "")
    nudge = ability.get_follow_up(ToolResult.ok(_ANSWER))
    held = Held(text(_ANSWER))
    script: list[Entry] = [held]
    provider = ChannelProvider(
        chat=[calls(tool(delegate, instructions=_INSTRUCTIONS)), text(_STARTED), text(_ANSWERED), text("Noon.")],
        **{delegate: script},
    )

    with _mailbox(client, delegate):
        parent, _frames = run_turn(provider, UserConfig(), _QUESTION)
        with patched(provider):
            later = begin(UserConfig(), "thanks, unrelated: what time is it?")
            later.result()
            drain()

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, delegate)
    n = agent_number(delegate, child_turn(db, delegate_channel(delegate), call["id"]))
    label = f"Reply from {delegate} #{n}"
    assert call["state"] == "done", call["result"]
    assert (
        f'Subagent {delegate} #{n} has been spawned. It will reply to you once it\'s done as "{label}". '
        "Acknowledge to the user that the agent has been spawned and you'll be invoked automatically "
        "once it replies."
    ) in call["result"]
    assert _ANSWER not in call["result"] and _BLOCK_OPENER not in call["result"]
    assert assistant_texts(db, USER, parent.turn_id)[-1] == _STARTED

    [reply] = replies(db, call["id"])
    assert reply["content"] == _ANSWER
    assert (reply["turn_id"] != parent.turn_id, reply["joined"]) == (True, 0)
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == [_ANSWERED]
    opener = messages(block(client, f"/api/threads/{reply['turn_id']}"))[0]
    assert (opener["role"], opener["content"]) == ("user", _ANSWER)
    # The reply carries plain data for the interface to word — the subagent's
    # tool, and no summary since the call carried none — and never the label
    # the model reads.
    assert opener["reply_from"] == {"tool_name": delegate, "summary": ""}
    assert opener.get("speaker") is None

    opening, _second, answering, unrelated = provider.read("chat")
    follow_up = _block_opening(_after_reply(answering, label, _ANSWER))
    if steer or nudge:
        assert follow_up is not None, answering[-1500:]
        assert steer in follow_up and nudge in follow_up
        assert answering.count(_BLOCK_OPENER) == 1
    else:
        assert follow_up is None and _BLOCK_OPENER not in answering
    offered = provider.offered("chat")
    assert all(ability.NAME in names for names in offered), offered
    assert follow_up_tools <= offered[2]
    assert not any(names & follow_up_tools for names in (*offered[:2], *offered[3:])), offered

    assert label in unrelated and _ANSWER in unrelated
    assert _BLOCK_OPENER not in unrelated
    if steer:
        assert steer[:60] not in unrelated
    assert _BLOCK_OPENER not in opening


def test_a_plain_text_answer_reaches_the_chats_reply_exactly_as_the_subagent_wrote_it(
    authed_client: _Client,
) -> None:
    """A subagent replies in plain text, so its reply on the chat holds what it
    wrote character for character — line breaks kept, ``& < >`` and anything
    that looks like a tag or an emphasis marker left as typed, never escaped,
    stripped or turned into markup."""
    client, db, _store = authed_client
    answer = "Opens at 9 & closes at 5.\nIf a < b and c > d, see <div> or *notes* and _hours_.\n\nSee you."
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), text(_ANSWERED)],
        web_search=[Held(text(answer))],
    )

    run_turn(provider, UserConfig(), _QUESTION)

    [call] = calls_of(db, "web_search")
    [reply] = replies(db, call["id"])
    opener = messages(block(client, f"/api/threads/{reply['turn_id']}"))[0]
    assert (opener["role"], opener["content"]) == ("user", answer)


def test_a_subagent_called_from_a_thread_reply_answers_in_that_thread(db: sqlite3.Connection) -> None:
    """A thread reply on a finished exchange calls ``web_search``; the search
    ends after the thread reply has ended. Its answer comes back into that
    thread — the same turn, never a new one — and the chat's answer to it ends
    the thread's latest exchange."""
    held = Held(text(_ANSWER))
    provider = ChannelProvider(
        chat=[
            text("Paris."),
            calls(tool("web_search", instructions=_INSTRUCTIONS)),
            text(_STARTED),
            text(_ANSWERED),
        ],
        web_search=[held],
    )

    with patched(provider):
        opener = begin(UserConfig(), _QUESTION)
        opener.result()
        begin(UserConfig(), "can you double-check that?", turn_id=opener.turn_id).result()
        wait(held.entered, "the search's model call")
        held.gate.set()
        _held_threads_end(held)
        drain()

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, "web_search")
    [reply] = replies(db, call["id"])
    assert (reply["turn_id"], reply["joined"]) == (opener.turn_id, 0)
    assert _user_turns(db) == [opener.turn_id]
    assert execution(USER, opener.turn_id).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, opener.turn_id)[-1] == _ANSWERED


def test_a_subagent_that_finishes_without_an_answer_says_so_in_its_reply(db: sqlite3.Connection) -> None:
    """A search that ends with nothing to say still replies, so the chat is
    not left waiting on a promise: the reply says the search finished without
    an answer, and the model answering it reads that notice with no
    third-party-content warning and no suggested next step — there is no
    third-party content — then answers it on its own turn."""
    held = Held(calls(tool("recall", query="capital of France")))
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), text("It found nothing.")],
        web_search=[held, text("")],
    )

    parent, _frames = run_turn(provider, UserConfig(), _QUESTION)

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, "web_search")
    [reply] = replies(db, call["id"])
    assert reply["content"] == "This web_search finished without an answer."
    assert reply["turn_id"] != parent.turn_id
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["It found nothing."]
    answering = provider.read("chat")[-1]
    assert "This web_search finished without an answer." in answering
    assert _BLOCK_OPENER not in answering
    assert AbilityRegistry.get("web_search").UNTRUSTED_CONTENT[""][:60] not in answering


def test_two_subagents_that_answer_while_their_caller_works_are_each_answered_once(db: sqlite3.Connection) -> None:
    """One chat turn calls ``web_browse`` and ``code_agent``; both finish
    while that turn is still working. Both answers join the working turn —
    no turn of their own — and the turn reads them together on one more step,
    so each answer is answered exactly once. On that step the browse's answer
    is followed directly by its third-party-content warning; ``code_agent``'s,
    which carries none, is not. That step is offered both subagents and, as
    well, the tool the browse's next step names (``web_fetch``), which no
    earlier request of the turn was offered."""
    caller = Held(text("Both started."))
    browse = Held(text("Browse: Paris."))
    code = Held(text("Code: the script prints Paris."))
    provider = ChannelProvider(
        chat=[
            calls(
                tool(WebBrowseAbility.NAME, instructions=_INSTRUCTIONS),
                tool(CodeAgentAbility.NAME, instructions="write a script that prints the capital of France"),
            ),
            caller,
            text("Both answered."),
        ],
        web_browse=[browse],
        code_agent=[code],
    )
    steer = AbilityRegistry.get(WebBrowseAbility.NAME).UNTRUSTED_CONTENT[""]

    with patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        for h, what in ((caller, "the caller's second step"), (browse, "the browse"), (code, "the code agent")):
            wait(h.entered, what)
        browse.gate.set()
        code.gate.set()
        _held_threads_end(browse, code)
        caller.gate.set()
        parent.result()
        drain()

    assert caller.opened and browse.opened and code.opened and provider.unused() == {}
    numbers: dict[str, int] = {}
    for tool_name in (WebBrowseAbility.NAME, CodeAgentAbility.NAME):
        [call] = calls_of(db, tool_name)
        [reply] = replies(db, call["id"])
        assert (reply["turn_id"], reply["joined"]) == (parent.turn_id, 1), tool_name
        numbers[tool_name] = agent_number(tool_name, child_turn(db, delegate_channel(tool_name), call["id"]))
    assert _user_turns(db) == [parent.turn_id]
    chat = provider.read("chat")
    assert len(chat) == 3
    answering = chat[-1]
    browse_block = _block_opening(_after_reply(
        answering, f"Reply from {WebBrowseAbility.NAME} #{numbers[WebBrowseAbility.NAME]}", "Browse: Paris.",
    ))
    assert browse_block is not None and steer in browse_block
    code_tail = _after_reply(
        answering, f"Reply from {CodeAgentAbility.NAME} #{numbers[CodeAgentAbility.NAME]}",
        "Code: the script prints Paris.",
    )
    assert _block_opening(code_tail) is None
    assert answering.count(_BLOCK_OPENER) == 1
    offered = provider.offered("chat")
    assert {WebBrowseAbility.NAME, CodeAgentAbility.NAME, WebFetchAbility.NAME} <= offered[-1], offered[-1]
    assert not any(WebFetchAbility.NAME in names for names in offered[:-1]), offered
    assert execution(USER, parent.turn_id).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, parent.turn_id)[-1] == "Both answered."


def test_a_reply_never_lands_in_another_turn_that_is_working(db: sqlite3.Connection) -> None:
    """The chat calls ``web_search`` and its turn ends; the person starts a
    different conversation, and while that turn is working the search
    finishes. The answer opens a turn of its own — it does not join the turn
    that happens to be working — and both turns end with their own answers."""
    other = Held(text("Here is the weather."))
    held = Held(text(_ANSWER))
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), other, text(_ANSWERED)],
        web_search=[held],
    )

    with patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        working = begin(UserConfig(), "what is the weather today?")
        wait(other.entered, "the other turn's model call")
        wait(held.entered, "the search's model call")
        held.gate.set()
        _held_threads_end(held)
        other.gate.set()
        working.result()
        drain()

    assert held.opened and other.opened and provider.unused() == {}
    [call] = calls_of(db, "web_search")
    [reply] = replies(db, call["id"])
    assert reply["turn_id"] not in (parent.turn_id, working.turn_id)
    assert reply["joined"] == 0
    assert assistant_texts(db, USER, working.turn_id) == ["Here is the weather."]
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == [_ANSWERED]


def test_a_subagent_the_person_stops_tells_the_chat_it_was_stopped(authed_client: _Client) -> None:
    """After the chat's turn has ended, the person stops the search from its
    transcript panel while it is mid-call. The search ends cancelled with its
    late answer never stored, and the chat is told, in a reply that opens a
    turn of its own, that the search was stopped before it finished — with no
    third-party-content warning — and answers it."""
    client, db, _store = authed_client
    channel = delegate_channel("web_search")
    held = Held(text(_ANSWER))
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), text("It was stopped.")],
        web_search=[held],
    )

    with patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        wait(held.entered, "the search's model call")
        [call] = calls_of(db, "web_search")
        n = child_turn(db, channel, call["id"])
        stop = client.delete(f"/api/threads/{n}?channel={channel}")
        held.gate.set()
        drain()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert held.opened and provider.unused() == {}
    assert execution(channel, n).state == TurnExecution.CANCELLED
    assert _ANSWER not in assistant_texts(db, channel, n)
    [reply] = replies(db, call["id"])
    assert reply["content"] == "This web_search was stopped before it finished."
    assert (reply["turn_id"] != parent.turn_id, reply["joined"]) == (True, 0)
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["It was stopped."]
    answering = provider.read("chat")[-1]
    assert "This web_search was stopped before it finished." in answering
    assert _BLOCK_OPENER not in answering


def test_a_subagent_whose_provider_keeps_failing_tells_the_chat_why_it_crashed(db: sqlite3.Connection) -> None:
    """The search's provider fails on every attempt after the chat's turn has
    ended. The search crashes, and the chat is told why, in a reply that opens
    a turn of its own — with no third-party-content warning — and answers
    it."""
    boom = RuntimeError("boom")
    first = Held(boom)
    script: list[Entry] = [first, *(raises(boom) for _ in range(_MAX_PROVIDER_ATTEMPTS - 1))]
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), text("It failed.")],
        web_search=script,
    )

    parent, _frames = run_turn(provider, UserConfig(), _QUESTION)

    assert first.opened and provider.unused() == {}
    assert len(provider.texts("web_search")) == _MAX_PROVIDER_ATTEMPTS
    [call] = calls_of(db, "web_search")
    search = execution(delegate_channel("web_search"), child_turn(db, delegate_channel("web_search"), call["id"]))
    assert (search.state, search.stop_reason) == (TurnExecution.CRASHED, _EXHAUSTED)
    [reply] = replies(db, call["id"])
    assert reply["content"] == f"This web_search crashed: {_EXHAUSTED}"
    assert (reply["turn_id"] != parent.turn_id, reply["joined"]) == (True, 0)
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["It failed."]
    assert _BLOCK_OPENER not in provider.read("chat")[-1]


def test_boot_tells_the_chat_a_subagent_was_cut_off_by_a_restart(db: sqlite3.Connection) -> None:
    """The process dies while the search is mid-call, so nothing is sent back
    then. The next boot closes the search's turn as cut off by the restart and
    tells the chat so, once, in a reply that opens a turn of its own — with no
    third-party-content warning — which the chat answers."""
    channel = delegate_channel("web_search")
    script: list[Entry] = [dies]
    provider = ChannelProvider(
        chat=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(_STARTED), text("It was cut off.")],
        web_search=script,
    )

    with process_deaths_expected():
        parent, _frames = run_turn(provider, UserConfig(), _QUESTION)
    [call] = calls_of(db, "web_search")
    n = child_turn(db, channel, call["id"])
    assert [(r.channel, r.turn_id) for r in TurnExecution.open_rows()] == [(channel, n)]
    assert replies(db, call["id"]) == []

    with patched(provider):
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_delegates(swept)
        drain()

    assert provider.unused() == {}
    assert [(r.channel, r.turn_id, r.state, r.stop_reason) for r in swept] == [
        (channel, n, TurnExecution.CRASHED, "process death"),
    ]
    [reply] = replies(db, call["id"])
    assert reply["content"] == "This web_search was cut off by a restart before it finished."
    assert (reply["turn_id"] != parent.turn_id, reply["joined"]) == (True, 0)
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["It was cut off."]
    assert _BLOCK_OPENER not in provider.read("chat")[-1]
    assert list(TurnExecution.open_rows()) == []


@pytest.mark.parametrize("delegate", ["web_search", "web_browse", "code_agent", "pim", "subtask"])
def test_a_subagents_own_turn_is_never_offered_itself(authed_client: _Client, delegate: str) -> None:
    """A subagent's own turn opens with a row carrying the call that started it,
    just as a reply does — yet while the chat is offered the subagent on every
    request, the subagent's own turn never is."""
    client, db, _store = authed_client
    route = "subtask" if delegate == "subtask" else delegate
    start = (
        tool("subtask", action="start", instructions=_INSTRUCTIONS) if delegate == "subtask"
        else tool(delegate, instructions=_INSTRUCTIONS)
    )
    own: list[Entry] = [Held(text(_ANSWER))]
    provider = ChannelProvider(chat=[calls(start), text(_STARTED), text(_ANSWERED)], **{route: own})

    with _mailbox(client, delegate):
        run_turn(provider, UserConfig(), _QUESTION)

    assert provider.unused() == {}
    [call] = calls_of(db, delegate)
    [reply] = replies(db, call["id"])
    assert reply["content"] == _ANSWER
    [own_turn] = provider.offered(route)
    assert delegate not in own_turn
    assert delegate in provider.offered("chat")[-1]
