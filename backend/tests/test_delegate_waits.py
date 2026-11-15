"""Feature test: every caller but the person's chat waits for its subagent.

Only the person's chat is answered later by a reply. Any other caller of a
subagent that runs a turn of its own — a subtask's turn, a subagent's own turn, a
scheduled task (including the person's reply inside its thread), discovery,
an external agent — waits for the subagent to end and gets its answer as
the call's result, read on its very next step; nothing is ever sent back to
it, or to the person's chat, as a reply. The warning that the answer is
third-party content, and the next step the subagent suggests, ride on that
result. A subagent the person stopped comes back to a waiting scheduled task as
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
from configs.channels.discovery import DiscoveryConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from contracts.params.delegate_params_bag import DelegateParamsBag
from controllers.message_processor import MessageProcessor
from mcp_server.server import create_mcp_server
from models.turn_execution import TurnExecution
from services.processor_config import ProcessorConfig
from tests._delegates import (
    USER,
    ChannelProvider,
    Entry,
    Held,
    assistant_texts,
    calls,
    calls_of,
    child_turn,
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
    channel = _SUBTASK if tool_name == "subtask" else delegate_channel(tool_name)
    return channel, child_turn(db, channel, calls_of(db, tool_name)[0]["id"])


def _the_one_execution(db: sqlite3.Connection, channel: str, turn_id: int) -> str:
    """The state of a turn that must have run exactly once."""
    [row] = db.execute(
        "SELECT state FROM turn_executions WHERE channel = ? AND turn_id = ?", (channel, turn_id),
    ).fetchall()
    return cast("str", row["state"])


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
    its result is the subagent's answer — a search's with its third-party-content
    warning, ``code_agent``'s bare — and the caller reads it on its next step.
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
    steer = AbilityRegistry.get("web_search").UNTRUSTED_CONTENT[""]

    run_turn(provider, UserConfig(), "find the capital of France and write a script that prints it")

    assert provider.unused() == {}
    first, second = calls_of(db, "web_search")
    [coding] = calls_of(db, "code_agent")
    for call, answer in ((first, "Search one: Paris."), (second, "Search two: use print().")):
        assert call["state"] == "done", call["result"]
        assert answer in call["result"] and steer in call["result"]
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
    the call's result is the subagent's answer, with the third-party-content
    warning and suggested next step the subagent carries, and the caller's model
    reads it on its next step. Nothing is sent back as a reply — not to the
    caller's own channel, not to the person's chat — and the caller's turn runs
    once and completes."""
    caller = make_caller()
    ability = cast("DelegateAbility[DelegateParamsBag]", AbilityRegistry.get(delegate))
    steer = ability.UNTRUSTED_CONTENT.get("", "")
    nudge = ability.get_follow_up(ToolResult.ok(_ANSWER))
    answers: list[Entry] = [text(_ANSWER)]
    provider = ChannelProvider(chat=[calls(_delegate_call(delegate)), text("Noted.")], **{delegate: answers})

    mp = _run_waiting(provider, caller, turn_id)

    assert provider.unused() == {}
    [call] = calls_of(db, delegate)
    assert call["state"] == "done", call["result"]
    assert _ANSWER in call["result"]
    if steer or nudge:
        assert steer in call["result"] and nudge in call["result"]
    else:
        assert _BLOCK_OPENER not in call["result"]
    assert _ANSWER in provider.texts("chat")[-1]
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


def test_an_external_agent_waits_for_its_subagent_and_reads_the_answer_in_the_call_result(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """An external agent, through ``talk_to_chalie``, asks something Chalie
    answers with ``web_search`` (which the person allowed for external
    agents). The call waits for the search: its result is the answer with the
    third-party-content warning, Chalie's model reads it on its next step, and
    what the agent gets back is Chalie's answer built on it. Nothing is sent
    back as a reply — not to the agent's channel, not to the person's chat."""
    client, db, _store = authed_client
    allow = client.post(
        "/api/policies/-1", json={"channel": "external_agent", "permission": "web_search", "setting": "allow"},
    )
    assert allow.status_code == 204, allow.get_data(as_text=True)
    channel = Channel.for_external_agent("reviewer")
    provider = ChannelProvider(
        chat=[calls(_delegate_call("web_search")), text("Paris.")],
        web_search=[text(_ANSWER)],
    )
    steer = AbilityRegistry.get("web_search").UNTRUSTED_CONTENT[""]

    with patched(provider):
        result = asyncio.run(create_mcp_server().call_tool("talk_to_chalie", {
            "message": "what is the capital of France?", "agent_name": "reviewer", "project_or_task_name": "sample",
        }))
        drain()

    assert provider.unused() == {}
    assert isinstance(result, CallToolResult) and not result.is_error
    said = [block.text for block in result.content if isinstance(block, TextContent)]
    assert said == ["Paris."]
    [call] = calls_of(db, "web_search")
    assert call["state"] == "done", call["result"]
    assert _ANSWER in call["result"] and steer in call["result"]
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
