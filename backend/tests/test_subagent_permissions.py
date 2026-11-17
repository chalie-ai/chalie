"""Feature tests: starting a subagent never asks, and a tool a subagent calls is
checked as the agent at the top of the chain.

Starting a subagent — any of the five kinds — runs without a permission check
for every caller that can start one: the person's chat, a scheduled task, an
outside agent and the background discovery pass. (The housekeeping runs on the
subconscious policy channel hold no spawn tool.) Even a ``deny`` left behind for
a spawn tool is not read.

A tool a subagent calls is checked under the policy channel, and raised to the
person under the turn, of the agent that started the chain. A mail search the
chat's ``pim`` makes asks on the chat, once per call, listed under the chat turn
even after that turn has ended. A scheduled task's ``pim`` asks under the
scheduled task's turn, on the scheduled task's policy channel. A file write the
person set to ask, three levels down — chat, then subtask, then ``code_agent`` —
asks on the chat, under the chat turn. A script a code agent runs is checked
under its own permission: an outside agent's is refused once the owner denies
it, and a chat-started one set to ask asks on the chat.

Drives real turns against the real, fully migrated SQLite DB, observed through
the real socket registry and answered through the real Flask app. The mailbox
is connected through the product's own setup route, against an IMAP peer on
loopback. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test.
"""

from __future__ import annotations

import contextlib
import queue
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.email import EmailAbility
from abilities.find_tools import FindToolsAbility
from abilities.pim import PimAbility
from abilities.run_script import RunScriptAbility
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from abilities.write_file import WriteFileAbility
from configs.channels.discovery import DiscoveryConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from services.file_mapper_service import FileMapperService
from services.policy_manager import _BLOCK
from tests._delegates import (
    WAIT_S,
    ChannelProvider,
    Entry,
    Held,
    begin,
    calls,
    calls_of,
    child_turn,
    connected_mail,
    delegate_channel,
    drain,
    execution,
    listening,
    patched,
    replies,
    run_turn,
    text,
    tool,
    wait,
)
from tests.test_delegate_waits import _blocked, _talk

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SCHEDULE_ID = 7
_USER = Channel.USER.value
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_CHAT = PolicyChannel.CHAT.value
_ANSWER = "Paris is the capital of France."
_NO_MAIL = "I could not read the mail."
_MAIL_SEARCH = f"{EmailAbility.NAME}.search"
_FILE_WRITE = WriteFileAbility.NAME
_SCRIPT = RunScriptAbility.NAME
_OUTSIDE = PolicyChannel.EXTERNAL_AGENT.value

_SPAWNS = (
    SubtaskAbility.NAME, CodeAgentAbility.NAME, WebSearchAbility.NAME, WebBrowseAbility.NAME, PimAbility.NAME,
)
_CALLERS = ("chat", "schedule", "outside agent", "discovery")
_CASES = [
    (caller, spawn) for caller in _CALLERS for spawn in _SPAWNS
    # The background discovery pass may not start a subtask at all.
    if spawn != SubtaskAbility.NAME or caller != "discovery"
]

Ask = dict[str, object]


class _Asks:
    """The permission prompts broadcast while it listens, in order; each one is
    read back from the pending list and answered as the signed-in person."""

    def __init__(self) -> None:
        self._raised: queue.Queue[str] = queue.Queue()

    def note(self, frame: dict[str, object]) -> None:
        if frame.get("type") == "permission_request":
            self._raised.put(cast("str", frame["request_id"]))

    def deny_next(self, client: FlaskClient) -> Ask:
        """Wait for the next prompt, check it is the only one pending, deny it,
        and return it as the pending list showed it."""
        request_id = self._raised.get(timeout=WAIT_S)
        response = client.get("/api/policies/pending")
        assert response.status_code == 200, response.data
        [ask] = cast("list[Ask]", cast("dict[str, object]", response.get_json())["result"])
        assert ask["request_id"] == request_id
        answered = client.post("/api/policies/respond", json={"request_id": request_id, "approved": False})
        assert answered.status_code == 204, answered.data
        return ask

    def left(self) -> int:
        return self._raised.qsize()


def _origin(kind: str, turn_id: int) -> dict[str, object]:
    return {"type": kind, "turn_id": turn_id, "forked": False}


def _spawn_call(spawn: str) -> dict[str, object]:
    if spawn == SubtaskAbility.NAME:
        return tool(spawn, action="start", instructions="find out the capital of France")
    return tool(spawn, instructions="find out the capital of France")


def _deny_every_spawn_tool() -> None:
    """A ``deny`` on every channel for every spawn tool, as an install that had
    them in its permissions list might carry; starting a subagent must not read
    them, and a gate that did would refuse at once instead of waiting on an ask."""
    for channel in PolicyChannel:
        for permission in (*_SPAWNS, f"{SubtaskAbility.NAME}.start"):
            assert Policy.upsert(channel.value, permission, "deny") == 1


@pytest.mark.parametrize(("caller", "spawn"), _CASES, ids=[f"{c}-{s}" for c, s in _CASES])
def test_starting_a_subagent_never_asks_on_any_channel(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], caller: str, spawn: str,
) -> None:
    """The caller starts one subagent with a ``deny`` stored for every spawn
    tool; a caller that does not hold the spawn tool finds it first. No prompt
    is raised, nothing is refused, and the subagent's own turn opens and
    completes. The chat reads its answer as a late reply; every other caller
    waits for it."""
    client, db, _store = authed_client
    _deny_every_spawn_tool()
    answers: list[Entry] = [text(_ANSWER)]
    if caller == "chat":
        script: list[Entry] = [calls(_spawn_call(spawn)), text("Started."), text("Got the answer.")]
    else:
        script = [calls(tool(FindToolsAbility.NAME, query=[spawn])), calls(_spawn_call(spawn)), text("Got the answer.")]
    if caller == "outside agent" and spawn == SubtaskAbility.NAME:
        # An outside agent's subtask runs on a config of the agent's own kind,
        # so it answers on the same route, while the agent's turn waits.
        script.insert(2, answers.pop())
        provider = ChannelProvider(chat=script)
    else:
        provider = ChannelProvider(chat=script, **{"subtask" if spawn == SubtaskAbility.NAME else spawn: answers})
    mail = connected_mail(client) if spawn == PimAbility.NAME else contextlib.nullcontext()

    frames: list[dict[str, object]] = []
    with mail:
        if caller == "chat":
            _mp, frames = run_turn(provider, UserConfig(), "what is the capital of France?")
        elif caller == "schedule":
            _mp, frames = run_turn(provider, ScheduledConfig(), "what is the capital of France?", turn_id=_SCHEDULE_ID)
        elif caller == "discovery":
            _mp, frames = run_turn(provider, DiscoveryConfig(), "what is the capital of France?")
        else:
            with listening() as listener, patched(provider):
                assert _talk("what is the capital of France?") == ["Got the answer."]
                drain()
            frames = listener.frames

    assert provider.unused() == {}
    assert [f for f in frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    [call] = calls_of(db, spawn)
    assert call["state"] == ToolCall.DONE, call["result"]
    channel = _SUBTASK if spawn == SubtaskAbility.NAME else delegate_channel(spawn)
    assert execution(channel, child_turn(db, channel, call["id"])).state == TurnExecution.COMPLETED


def test_each_mail_search_a_chat_spawned_pim_makes_asks_on_the_chat_after_the_chat_turn_ended(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat starts ``pim`` and ends its turn. pim then searches the mail
    twice: each search asks on the chat, listed under the chat turn, one prompt
    per search. Both are denied, so neither runs; pim says so, and its answer
    reaches the chat as a reply."""
    client, db, _store = authed_client
    asks = _Asks()
    held = Held(calls(tool(EmailAbility.NAME, action="search", keyword="invoice")))
    provider = ChannelProvider(
        chat=[calls(tool(PimAbility.NAME, instructions="read the latest invoice")), text("Asked."), text("pim answered.")],
        pim=[held, calls(tool(EmailAbility.NAME, action="search", keyword="receipt")), text(_NO_MAIL)],
    )

    with connected_mail(client), listening(asks.note), patched(provider):
        chat = begin(UserConfig(), "read my latest invoice")
        wait(held.entered, "pim's first step")
        chat.result()
        assert execution(_USER, chat.turn_id).state == TurnExecution.COMPLETED
        held.gate.set()
        first, second = asks.deny_next(client), asks.deny_next(client)
        drain()

    assert held.opened and provider.unused() == {} and asks.left() == 0
    for ask in (first, second):
        assert (ask["action_id"], ask["origin"]) == (_MAIL_SEARCH, _origin("user", chat.turn_id))
    assert first["request_id"] != second["request_id"]
    assert _blocked(db) == [(_MAIL_SEARCH, _CHAT, "user_denied")] * 2
    for search in calls_of(db, EmailAbility.NAME):
        assert search["state"] == ToolCall.ERROR
        assert _BLOCK.format(permission=_MAIL_SEARCH) in search["result"]
    [pim] = calls_of(db, PimAbility.NAME)
    [reply] = replies(db, pim["id"])
    assert reply["content"] == _NO_MAIL
    assert reply["turn_id"] != chat.turn_id


def test_a_mail_search_a_scheduled_pim_makes_is_asked_exactly_as_the_schedules_own(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A scheduled task finds ``pim`` and has it search the mail. The search
    asks under the scheduled task's own turn, on the scheduled task's policy
    channel, and is denied. pim's answer comes back to the task, which waited
    for it, and nothing is sent as a reply."""
    client, db, _store = authed_client
    asks = _Asks()
    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[PimAbility.NAME])),
            calls(tool(PimAbility.NAME, instructions="read the latest invoice")),
            text("Done."),
        ],
        pim=[calls(tool(EmailAbility.NAME, action="search", keyword="invoice")), text(_NO_MAIL)],
    )

    with connected_mail(client), listening(asks.note), patched(provider):
        schedule = MessageProcessor.process(ScheduledConfig(), raw_input="read the latest invoice", turn_id=_SCHEDULE_ID)
        ask = asks.deny_next(client)
        schedule.result()
        drain()

    assert provider.unused() == {} and asks.left() == 0
    assert ScheduledConfig().policy_channel == PolicyChannel.CHAT
    assert (ask["action_id"], ask["origin"]) == (_MAIL_SEARCH, _origin("scheduled", _SCHEDULE_ID))
    assert _blocked(db) == [(_MAIL_SEARCH, _CHAT, "user_denied")]
    [pim] = calls_of(db, PimAbility.NAME)
    assert pim["state"] == ToolCall.DONE, pim["result"]
    assert _NO_MAIL in pim["result"]
    assert replies(db, pim["id"]) == []
    assert replies(db, pim["id"], ScheduledConfig().channel) == []


def test_a_file_write_three_subagents_down_asks_on_the_chat_that_started_the_chain(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The person has set file writes to ask on the chat. The chat starts a
    subtask and ends its turn; the subtask hands the work to ``code_agent``,
    which writes a file. The write asks on the chat, under the chat turn, and
    is checked on the chat's policy channel; denied, it does not run. The
    subtask's answer reaches the chat."""
    client, db, _store = authed_client
    assert Policy.upsert(_CHAT, _FILE_WRITE, "ask") == 1
    asks = _Asks()
    held = Held(calls(tool(CodeAgentAbility.NAME, instructions="create a notes file")))
    provider = ChannelProvider(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions="make a notes file")),
            text("I have started a subtask."),
            text("The subtask answered."),
        ],
        subtask=[held, text("The notes file could not be written.")],
        code_agent=[
            calls(tool(WriteFileAbility.NAME, path=str(FileMapperService.get_code_agent_workspace_path("notes.txt")), contents="")),
            text("I was not allowed to write it."),
        ],
    )

    with listening(asks.note), patched(provider):
        chat = begin(UserConfig(), "make me a notes file")
        wait(held.entered, "the subtask's first step")
        chat.result()
        assert execution(_USER, chat.turn_id).state == TurnExecution.COMPLETED
        held.gate.set()
        ask = asks.deny_next(client)
        drain()

    assert held.opened and provider.unused() == {} and asks.left() == 0
    assert (ask["action_id"], ask["origin"]) == (_FILE_WRITE, _origin("user", chat.turn_id))
    assert _blocked(db) == [(_FILE_WRITE, _CHAT, "user_denied")]
    [write] = calls_of(db, WriteFileAbility.NAME)
    assert write["state"] == ToolCall.ERROR
    assert _BLOCK.format(permission=_FILE_WRITE) in write["result"]
    [coding] = calls_of(db, CodeAgentAbility.NAME)
    assert coding["state"] == ToolCall.DONE, coding["result"]
    [subtask] = calls_of(db, SubtaskAbility.NAME)
    [reply] = replies(db, subtask["id"])
    assert reply["content"] == "The notes file could not be written."


def _run_script_call() -> dict[str, object]:
    return tool(_SCRIPT, path=str(FileMapperService.get_code_agent_workspace_path("hello.ts")))


def test_a_script_an_outside_agents_code_agent_runs_is_refused_once_the_owner_denies_it(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A new install allows scripts for outside agents, as the code agent's own
    row did before. The owner sets it to deny; an outside agent then has a code
    agent run a script. The script is refused, nothing asks, and the code
    agent's answer reaches the outside agent, which waited for it."""
    _client, db, _store = authed_client
    assert Policy.get_or_create_default(_OUTSIDE, _SCRIPT) == "allow"
    assert Policy.upsert(_OUTSIDE, _SCRIPT, "deny") == 1
    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[CodeAgentAbility.NAME])),
            calls(tool(CodeAgentAbility.NAME, instructions="print hello")),
            text("The script was not allowed."),
        ],
        code_agent=[calls(_run_script_call()), text("I was not allowed to run it.")],
    )

    with listening() as listener, patched(provider):
        assert _talk("print hello with a script") == ["The script was not allowed."]
        drain()

    assert provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == [(_SCRIPT, _OUTSIDE, "deny")]
    [script] = calls_of(db, _SCRIPT)
    assert script["state"] == ToolCall.ERROR
    assert _BLOCK.format(permission=_SCRIPT) in script["result"]
    [coding] = calls_of(db, CodeAgentAbility.NAME)
    assert coding["state"] == ToolCall.DONE, coding["result"]
    assert "I was not allowed to run it." in coding["result"]


def test_a_script_a_chat_spawned_code_agent_runs_asks_on_the_chat(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The person sets scripts to ask on the chat. The chat starts a code agent,
    which runs a script: the script asks on the chat, under the chat turn, on
    the chat's policy channel. Denied, it does not run, and the code agent's
    answer reaches the chat as a reply."""
    client, db, _store = authed_client
    assert Policy.upsert(_CHAT, _SCRIPT, "ask") == 1
    asks = _Asks()
    provider = ChannelProvider(
        chat=[
            calls(tool(CodeAgentAbility.NAME, instructions="print hello")),
            text("I have started a coding agent."),
            text("The coding agent answered."),
        ],
        code_agent=[calls(_run_script_call()), text("I was not allowed to run it.")],
    )

    with listening(asks.note), patched(provider):
        chat = begin(UserConfig(), "print hello with a script")
        ask = asks.deny_next(client)
        chat.result()
        drain()

    assert provider.unused() == {} and asks.left() == 0
    assert (ask["action_id"], ask["origin"]) == (_SCRIPT, _origin("user", chat.turn_id))
    assert _blocked(db) == [(_SCRIPT, _CHAT, "user_denied")]
    [script] = calls_of(db, _SCRIPT)
    assert script["state"] == ToolCall.ERROR
    assert _BLOCK.format(permission=_SCRIPT) in script["result"]
    [coding] = calls_of(db, CodeAgentAbility.NAME)
    [reply] = replies(db, coding["id"])
    assert reply["content"] == "I was not allowed to run it."
