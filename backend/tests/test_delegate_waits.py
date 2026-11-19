"""Feature test: every caller but the person's chat waits for its subagent.

Only the person's chat is answered later by a reply. Any other caller of a
subagent that runs a turn of its own — a subtask's turn, a subagent's own turn, a
scheduled task (including the person's reply inside its thread), discovery,
an external agent — waits for the subagent to end and gets its answer as
the call's result, read on its very next step; nothing is ever sent back to
it, or to the person's chat, as a reply. The next step the subagent
suggests rides on that result. A subagent the person stopped comes back to a waiting scheduled task as
stopped, without retry advice; a subagent that ends with nothing to say comes
back as no answer, with the tool's retry advice.

Drives the real production entry points (a chat turn, a scheduled run and a
discovery run, each through ``MessageProcessor``; the external agents' MCP tool;
the Flask thread send and stop routes) against the real, fully-migrated SQLite
DB. The only substitution is the LLM network boundary
(``services.provider_service.build_client``).
"""

import asyncio
import re
import sqlite3
from collections.abc import Callable
from typing import cast

import pytest
from flask.testing import FlaskClient
from mcp.types import CallToolResult, TextContent

from abilities._delegate import DelegateAbility
from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from abilities.message_parent import SPAWN_CONTEXT
from abilities.email import EmailAbility
from configs.channels.discovery import DiscoveryConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from contracts.params.delegate_params_bag import DelegateParamsBag
from controllers.message_processor import MessageProcessor
from mcp_server.server import create_mcp_server
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.processor_config import ProcessorConfig
from tests._delegates import (
    USER,
    ChannelProvider,
    Entry,
    Held,
    Request,
    _as_read,
    as_sent,
    assistant_texts,
    calls,
    calls_of,
    child,
    connected_mail,
    delegate_channel,
    drain,
    execution,
    linked_rows,
    patched,
    replies,
    run_turn,
    text,
    tool,
    wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_SCHEDULE_ID = 7
_ERROR = re.compile(r"\A\[[\w.]+\(status=error, code=([\w-]+)")
_HINT_LINE = re.compile(r"^hint: \S", re.MULTILINE)
_BLOCK_OPENER = "[follow_up_instruction]"
_ANSWER = "Paris is the capital of France."
_NO_MAIL = "I could not read the mail."
_AGENT = "atlas-reviewer"
_PROJECT = "lighthouse-migration"


def _error_code(result: str) -> str | None:
    """The outcome code in the header of a failed call's result envelope."""
    match = _ERROR.match(result)
    return match.group(1) if match else None


def _delegate_route(tool_name: str) -> str:
    """The provider route a subagent's own turn is answered from."""
    return "subtask" if tool_name == "subtask" else tool_name


def _delegate_call(tool_name: str) -> dict[str, object]:
    if tool_name == "subtask":
        return tool("subtask", action="start", instructions="find out the capital of France")
    return tool(tool_name, instructions="find the capital of France")


def _delegate_turn(db: sqlite3.Connection, tool_name: str) -> tuple[str, int]:
    """The channel and turn of the subagent turn the one call of *tool_name* opened."""
    kind = _SUBTASK if tool_name == "subtask" else delegate_channel(tool_name)
    return child(db, kind, calls_of(db, tool_name)[0]["id"])


def _the_one_execution(db: sqlite3.Connection, channel: str, turn_id: int) -> str:
    """The state of a turn that must have run exactly once."""
    [row] = db.execute(
        "SELECT state FROM turn_executions WHERE channel = ? AND turn_id = ?", (channel, turn_id),
    ).fetchall()
    return cast("str", row["state"])


def _talk(message: str, *, loop_in_human: bool = False) -> list[str]:
    """What the outside agent :data:`_AGENT` gets back from one
    ``talk_to_chalie`` call about :data:`_PROJECT`."""
    result = asyncio.run(create_mcp_server().call_tool("talk_to_chalie", {
        "message": message, "agent_name": _AGENT, "project_or_task_name": _PROJECT,
        "loop_in_human": loop_in_human,
    }))
    assert isinstance(result, CallToolResult) and not result.is_error
    return [block.text for block in result.content if isinstance(block, TextContent)]


def _blocked(db: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Every call the permission gate refused: its permission, policy channel and reason."""
    return [
        (r["action_id"], r["context"], r["reason"])
        for r in db.execute("SELECT action_id, context, reason FROM policy_blocked_log ORDER BY id")
    ]


def _reached(db: sqlite3.Connection, kind: str, call_id: int) -> int:
    """How many rows on the own channel of any subagent of *kind* (that kind's
    bare channel) carry the call *call_id* — none when the call never opened
    such a subagent's turn."""
    return cast("int", db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel >= ? AND channel < ? AND tool_call_id = ?",
        (f"{kind}:", f"{kind};", str(call_id)),
    ).fetchone()[0])


def _executions(db: sqlite3.Connection, channel: str) -> int:
    """How many turns ever opened on *channel*."""
    return cast("int", db.execute(
        "SELECT COUNT(*) FROM turn_executions WHERE channel = ?", (channel,),
    ).fetchone()[0])


def _kind_executions(db: sqlite3.Connection, kind: str) -> int:
    """How many turns ever opened on the own channels of the subagents of
    *kind* (that kind's bare channel), all told."""
    return cast("int", db.execute(
        "SELECT COUNT(*) FROM turn_executions WHERE channel >= ? AND channel < ?", (f"{kind}:", f"{kind};"),
    ).fetchone()[0])


def _run_waiting(provider: ChannelProvider, config: ProcessorConfig, turn_id: int) -> MessageProcessor:
    """Run one turn of a caller that is not the person's chat, as its own
    entry point runs it, to its end and the end of everything it spawned."""
    with patched(provider):
        mp = MessageProcessor.process(config, raw_input="write a short note on France", turn_id=turn_id)
        mp.result()
        drain()
    return mp


def test_a_subtask_and_a_subagent_inside_it_wait_for_the_subagents_they_call(db: sqlite3.Connection) -> None:
    """A subtask the chat started searches the web, then hands a coding task
    to ``code_agent``, which searches the web itself. Each of those calls waits:
    its result is the subagent's answer, bare, and the caller reads it on its
    next step.
    None of them is answered by a reply; the one reply the chat gets is the
    subtask's own, which the chat answers."""
    provider = ChannelProvider(
        chat=[
            calls(tool("subtask", action="start", instructions="find the capital of France and script it")),
            text("I have started a subtask."),
            text("The subtask is done."),
        ],
        subtask=[
            Held(calls(tool("web_search", instructions="capital of France"))),
            calls(tool("code_agent", instructions="write a script that prints the capital")),
            text("Paris, and the script prints it."),
        ],
        code_agent=[calls(tool("web_search", instructions="how to print in Python")), text("print('Paris')")],
        web_search=[text("Search one: Paris."), text("Search two: use print().")],
    )
    run_turn(provider, UserConfig(), "find the capital of France and write a script that prints it")

    assert provider.unused() == {}
    first, second = calls_of(db, "web_search")
    [coding] = calls_of(db, "code_agent")
    for call, answer in ((first, "Search one: Paris."), (second, "Search two: use print().")):
        assert call["state"] == "done", call["result"]
        assert answer in call["result"] and _BLOCK_OPENER not in call["result"]
    assert coding["state"] == "done", coding["result"]
    assert "print('Paris')" in coding["result"] and _BLOCK_OPENER not in coding["result"]
    assert "Search one: Paris." in provider.texts("subtask")[1]
    assert "print('Paris')" in provider.texts("subtask")[2]
    assert "Search two: use print()." in provider.texts("code_agent")[1]

    for call in (first, second, coding):
        assert replies(db, call["id"]) == []
    [subtask] = calls_of(db, "subtask")
    assert [row["tool_call_id"] for row in linked_rows(db, USER) if row["role"] == "user"] == [str(subtask["id"])]
    [reply] = replies(db, subtask["id"])
    assert assistant_texts(db, USER, reply["turn_id"]) == ["The subtask is done."]


@pytest.mark.parametrize(
    ("make_caller", "turn_id", "delegate"),
    [
        (ScheduledConfig, _SCHEDULE_ID, "web_search"),
        (ScheduledConfig, _SCHEDULE_ID, "web_browse"),
        (ScheduledConfig, _SCHEDULE_ID, "code_agent"),
        (DiscoveryConfig, -1, "web_search"),
    ],
    ids=["scheduled-web_search", "scheduled-web_browse", "scheduled-code_agent", "discovery-web_search"],
)
def test_a_scheduled_task_or_discovery_waits_for_its_subagent_and_reads_the_answer(
    db: sqlite3.Connection, make_caller: Callable[[], ProcessorConfig], turn_id: int, delegate: str,
) -> None:
    """A scheduled task, or a discovery run, that calls a subagent waits for it:
    the call's result is the subagent's answer, stored bare, and the caller's
    model reads it on its next step together with the suggested next step the
    subagent carries. Nothing is sent back as a reply — not to the
    caller's own channel, not to the person's chat — and the caller's turn runs
    once and completes."""
    caller = make_caller()
    ability = cast("DelegateAbility[DelegateParamsBag]", AbilityRegistry.get(delegate))
    nudge = ability.get_follow_up(ToolResult.ok(_ANSWER))
    answers: list[Entry] = [text(_ANSWER)]
    provider = ChannelProvider(chat=[calls(_delegate_call(delegate)), text("Noted.")], **{delegate: answers})

    mp = _run_waiting(provider, caller, turn_id)

    assert provider.unused() == {}
    [call] = calls_of(db, delegate)
    assert call["state"] == "done", call["result"]
    assert _ANSWER in call["result"] and _BLOCK_OPENER not in call["result"]
    assert _ANSWER in provider.texts("chat")[-1]
    if nudge:
        assert nudge in provider.read("chat")[-1]
    assert replies(db, call["id"]) == []
    assert replies(db, call["id"], caller.channel) == []
    assert _the_one_execution(db, caller.channel, mp.turn_id) == TurnExecution.COMPLETED
    assert assistant_texts(db, caller.channel, mp.turn_id)[-1] == "Noted."


@pytest.mark.parametrize("delegate", ["web_search", "subtask"])
def test_a_subagent_the_person_stops_comes_back_to_the_scheduled_task_as_stopped(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate: str,
) -> None:
    """The person stops a subagent a scheduled task is waiting on, from the
    subagent's transcript panel. The subagent ends cancelled with its late answer
    never stored; the scheduled task's call fails as ``delegate-stopped`` with
    no retry advice, nothing is sent back as a reply, and the scheduled task
    still completes with its own answer."""
    client, db, _store = authed_client
    held = Held(text(_ANSWER))
    script: list[Entry] = [held]
    provider = ChannelProvider(chat=[calls(_delegate_call(delegate)), text("Done.")], **{_delegate_route(delegate): script})

    with patched(provider):
        schedule = MessageProcessor.process(ScheduledConfig(), raw_input="write a note", turn_id=_SCHEDULE_ID)
        wait(held.entered, "the subagent's model call")
        channel, n = _delegate_turn(db, delegate)
        stop = client.delete(f"/api/threads/{n}?channel={channel}")
        held.gate.set()
        schedule.result()
        drain()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert held.opened and provider.unused() == {}
    assert execution(channel, n).state == TurnExecution.CANCELLED
    assert _ANSWER not in assistant_texts(db, channel, n)
    [call] = calls_of(db, delegate)
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-stopped"
    assert not _HINT_LINE.search(call["result"])
    assert linked_rows(db, USER) == []
    assert replies(db, call["id"], ScheduledConfig().channel) == []
    assert _the_one_execution(db, ScheduledConfig().channel, _SCHEDULE_ID) == TurnExecution.COMPLETED
    assert assistant_texts(db, ScheduledConfig().channel, _SCHEDULE_ID)[-1] == "Done."


@pytest.mark.parametrize("delegate", ["web_search", "code_agent"])
def test_an_external_agent_waits_for_its_subagent_and_reads_the_answer_in_the_call_result(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], delegate: str,
) -> None:
    """An external agent, through ``talk_to_chalie``, asks something Chalie
    hands to ``web_search`` or ``code_agent``, both of which a fresh install
    allows external agents — nothing is refused. The call waits for the
    subagent: its result is the answer, bare — Chalie's model reads
    it on its next step, and what the agent gets back is Chalie's answer built
    on it. Nothing is sent back as a reply — not to the agent's channel, not
    to the person's chat."""
    _client, db, _store = authed_client
    channel = Channel.for_external_agent("reviewer")
    answers: list[Entry] = [text(_ANSWER)]
    provider = ChannelProvider(chat=[calls(_delegate_call(delegate)), text("Paris.")], **{delegate: answers})

    with patched(provider):
        result = asyncio.run(create_mcp_server().call_tool("talk_to_chalie", {
            "message": "what is the capital of France?", "agent_name": "reviewer", "project_or_task_name": "sample",
        }))
        drain()

    assert provider.unused() == {}
    assert isinstance(result, CallToolResult) and not result.is_error
    said = [block.text for block in result.content if isinstance(block, TextContent)]
    assert said == ["Paris."]
    assert _blocked(db) == []
    [call] = calls_of(db, delegate)
    assert call["state"] == "done", call["result"]
    assert call["result"].split("\n")[1:-1] == [_ANSWER]
    assert _ANSWER in provider.texts("chat")[-1]
    assert replies(db, call["id"]) == []
    assert replies(db, call["id"], channel) == []
    [turn] = {row["turn_id"] for row in db.execute("SELECT turn_id FROM transcript WHERE channel = ?", (channel,))}
    assert _the_one_execution(db, channel, turn) == TurnExecution.COMPLETED
    assert assistant_texts(db, channel, turn)[-1] == "Paris."


def test_a_persons_reply_in_a_scheduled_tasks_thread_waits_for_its_subagent(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The person replies inside a scheduled task's thread, and that reply
    calls ``web_search``. The reply runs as the scheduled task, not as the
    chat, so the call waits: its result is the search's answer, the model
    reads it on its next step, and the thread's latest exchange ends with the
    answer built on it. Nothing is sent back as a reply — not into the
    scheduled task's thread, not to the person's chat."""
    client, db, _store = authed_client
    schedule = ScheduledConfig().channel
    provider = ChannelProvider(
        chat=[text("Noted."), calls(_delegate_call("web_search")), text("Paris.")],
        web_search=[text(_ANSWER)],
    )

    with patched(provider):
        MessageProcessor.process(ScheduledConfig(), raw_input="write a short note on France", turn_id=_SCHEDULE_ID).result()
        sent = client.post(
            f"/api/threads/{_SCHEDULE_ID}", data={"text": "what is its capital?", "type": "scheduled"},
        )
        drain()

    assert sent.status_code == 200, sent.get_data(as_text=True)
    assert provider.unused() == {}
    [call] = calls_of(db, "web_search")
    assert call["state"] == "done", call["result"]
    assert _ANSWER in call["result"]
    assert _ANSWER in provider.texts("chat")[-1]
    assert replies(db, call["id"]) == []
    assert replies(db, call["id"], schedule) == []
    assert linked_rows(db, USER) == []
    assert assistant_texts(db, schedule, _SCHEDULE_ID)[-1] == "Paris."
    assert execution(schedule, _SCHEDULE_ID).state == TurnExecution.COMPLETED


@pytest.mark.parametrize("delegate", ["web_search", "subtask"])
def test_a_subagent_with_nothing_to_say_comes_back_to_the_scheduled_task_as_no_answer(
    db: sqlite3.Connection, delegate: str,
) -> None:
    """A subagent a scheduled task waits on finishes — nobody stopped it — with
    nothing to say. That is not mistaken for an answer or for a stop: the
    scheduled task's call fails as ``delegate-no-answer`` with the tool's retry
    advice, nothing is sent back as a reply, and the scheduled task completes
    with its own answer."""
    script: list[Entry] = [calls(tool("recall", query="capital of France")), text("")]
    provider = ChannelProvider(chat=[calls(_delegate_call(delegate)), text("Done.")], **{_delegate_route(delegate): script})

    _run_waiting(provider, ScheduledConfig(), _SCHEDULE_ID)

    assert provider.unused() == {}
    channel, n = _delegate_turn(db, delegate)
    assert execution(channel, n).state == TurnExecution.COMPLETED
    [call] = calls_of(db, delegate)
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-no-answer"
    assert _HINT_LINE.search(call["result"])
    assert linked_rows(db, USER) == []
    assert replies(db, call["id"], ScheduledConfig().channel) == []
    assert _the_one_execution(db, ScheduledConfig().channel, _SCHEDULE_ID) == TurnExecution.COMPLETED
    assert assistant_texts(db, ScheduledConfig().channel, _SCHEDULE_ID)[-1] == "Done."


def test_an_outside_agents_subtask_runs_as_that_agent_with_a_clean_context_and_answers_the_call(
    db: sqlite3.Connection,
) -> None:
    """An outside agent tells Chalie a codename, then, in a later call that
    loops the person in, asks for work Chalie hands to a subtask. The agent's
    turn is sent a ``subtask`` tool that can only start one and says the call
    waits; the start waits, its result is the subtask's answer with nothing in
    front of it, and the agent's answer is built on it.

    The subtask runs as that agent and nothing more: its system prompt names
    the agent and the project; the outside agent's permissions refuse it the
    device lookup the person's chat is allowed; it reads its task with only
    the request that opened the agent's turn as context, and nothing from the
    agent's earlier turns — never the codename; and it cannot start a subtask
    of its own. The person is told about the exchange once, by the
    agent's own turn — the subtask's turn tells them nothing — and nothing is
    sent back as a reply."""
    asked = "find out the capital of France for the migration notes"
    work = "find out the capital of France"
    seen: list[Request] = []

    def answering_the_subtask(response: ProviderResponse) -> Entry:
        # The agent's turn waits on its subtask, so nothing else chat-shaped
        # runs while the subtask calls: the latest chat call is the subtask's.
        def entry() -> ProviderResponse:
            seen.append(provider.last("chat"))
            return response
        return entry

    provider = ChannelProvider(chat=[
        text("Noted."),
        calls(tool("find_tools", query=["subtask"])),
        calls(tool("subtask", action="start", instructions=work)),
        answering_the_subtask(
            calls(tool("user_device"), tool("subtask", action="start", instructions="double-check it")),
        ),
        answering_the_subtask(text(_ANSWER)),
        text("Paris."),
        text("Told the person."),
    ])

    with patched(provider):
        first = _talk("remember the codename Bluebird")
        drain()
        second = _talk(asked, loop_in_human=True)
        drain()

    assert provider.unused() == {}
    assert (first, second) == (["Noted."], ["Paris."])

    agents_turn = [
        r for r in provider.requests
        if r.route == "chat" and _AGENT in r.system and asked in r.text and r not in seen
    ]
    assert all("Bluebird" in r.text for r in agents_turn)
    before_discovery, *sent = [r.definition("subtask") for r in agents_turn]
    assert before_discovery is None
    assert len(sent) == 2
    for description, actions, params in map(as_sent, sent):
        assert actions == ["start"]
        assert params == {"action", "instructions", "act_summary"}
        assert "waits" in description

    start, nested = calls_of(db, "subtask")
    assert start["state"] == "done", start["result"]
    assert start["result"].split("\n")[1:-1] == [_ANSWER]
    assert _ANSWER in agents_turn[-1].text

    assert len(seen) == 2
    for request in seen:
        assert _AGENT in request.system and _PROJECT in request.system
        assert f"## Task\n{work}\n\n{SPAWN_CONTEXT.format(context=asked)}## Instructions\n" in _as_read(request.text)
        assert "Bluebird" not in request.text

    [device] = calls_of(db, "user_device")
    assert device["state"] == "error", device["result"]
    assert _blocked(db) == [("user_device", "external_agent", "deny")]
    assert nested["state"] == "error"
    assert _error_code(nested["result"]) == "subtask-nested"
    assert _reached(db, _SUBTASK, nested["id"]) == 0
    assert _the_one_execution(db, *child(db, _SUBTASK, start["id"])) == TurnExecution.COMPLETED

    assert _executions(db, USER) == 1
    assert replies(db, start["id"]) == []
    assert replies(db, start["id"], Channel.for_external_agent(_AGENT)) == []


def test_an_outside_agent_starts_pim_but_its_mail_call_is_refused_and_its_web_search_cannot_start_a_subtask(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """On a fresh install an outside agent's turn asks ``pim`` to read the
    person's connected mail. Starting a subagent never asks, so pim's turn opens; its
    mail search is checked under the outside agent's channel, which a fresh
    install denies, so the refusal lands in the blocked log and pim still
    answers. The agent then asks ``web_search``; the search's own turn tries to
    start a subtask and is refused, as every subagent's turn is, so no subtask
    opens and the search still answers the call."""
    client, db, _store = authed_client
    provider = ChannelProvider(
        chat=[
            calls(tool("pim", instructions="read the latest email")),
            calls(_delegate_call("web_search")),
            text("Done."),
        ],
        pim=[calls(tool(EmailAbility.NAME, action="search", keyword="invoice")), text(_NO_MAIL)],
        web_search=[calls(tool("subtask", action="start", instructions="look it up")), text(_ANSWER)],
    )

    with connected_mail(client), patched(provider):
        said = _talk("read the person's mail, then look up the capital of France")
        drain()

    assert provider.unused() == {}
    assert said == ["Done."]
    [mail] = calls_of(db, "pim")
    assert mail["state"] == "done", mail["result"]
    assert _NO_MAIL in mail["result"]
    assert _kind_executions(db, delegate_channel("pim")) == 1
    [search_mail] = calls_of(db, EmailAbility.NAME)
    assert search_mail["state"] == "error", search_mail["result"]
    assert _blocked(db) == [(f"{EmailAbility.NAME}.search", "external_agent", "deny")]

    [search] = calls_of(db, "web_search")
    assert search["state"] == "done", search["result"]
    assert _ANSWER in search["result"]
    [nested] = calls_of(db, "subtask")
    assert nested["state"] == "error"
    assert _error_code(nested["result"]) == "subtask-nested"
    assert _kind_executions(db, _SUBTASK) == 0
