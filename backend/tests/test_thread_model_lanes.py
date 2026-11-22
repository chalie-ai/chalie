"""Feature tests: which model the work a thread starts runs on.

A subagent on the chat lane — a subtask, the coding agent — starts its thread
on the model and thinking level of the turn that spawned it and keeps them on
its later turns. Web browse keeps the Delegate Provider's model and its own
level. A compaction and its hand-over run on the model of the turn they shrink
and start no thread. A caller's model deleted mid-turn reaches its subtask as a
stale id, and the subtask's thread heals to the default like any other.

Real throughout: the ``MessageProcessor`` step loop, the subagent tools, the
thread settings and provider delete endpoints and the migrated SQLite database.
The only substitution is the LLM network boundary
(``services.provider_service.build_client``): it builds one client per provider
row, as the real factory does, and records the row and the thinking level of
every call before the scripted double answers it.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Protocol, cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.thinking_level import ThinkingLevel
from controllers.message_processor import MessageProcessor
from models.agent import Agent
from models.provider_response import ProviderResponse
from models.thread import Thread
from services.provider_api import ProviderApiRequest
from tests._delegates import (
    BUILD_CLIENT,
    ChannelProvider,
    Held,
    calls,
    calls_of,
    child,
    delegate_channel,
    drain,
    text,
    tool,
    wait,
)
from tests.helpers import is_planning_request
from tests.test_agent_tools import _find_agent_tools
from tests.test_thread_settings import _change, _stored
from tests.test_turn_handover import _COMPACTION_MARK, _HANDOVER_MARK, _HandoverProvider
from tests.test_turn_join import AuthedClient

pytestmark = pytest.mark.unit

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_BROWSE = delegate_channel(WebBrowseAbility.NAME)
_CODE = delegate_channel(CodeAgentAbility.NAME)
_HIGH = ThinkingLevel.HIGH

Call = tuple[int, ThinkingLevel]


class _Answers(Protocol):
    def get_context_limit(self) -> int: ...
    def send(self, dto: ProviderApiRequest) -> ProviderResponse: ...


class _Lanes:
    """Builds a client per provider row and records, under the route
    *route_of* names, the row and thinking level each call went out on; the
    planning pre-pass is left out. *answers* answers every call."""

    def __init__(self, answers: _Answers, route_of: Callable[[ProviderApiRequest], str]) -> None:
        self._answers = answers
        self._route_of = route_of
        self._lock = threading.Lock()
        self._calls: list[tuple[str, int, ThinkingLevel]] = []

    def build(self, config: dict[str, object], *_a: object, **_k: object) -> _Client:
        return _Client(self, cast("int", config["id"]))

    def send(self, dto: ProviderApiRequest, provider_id: int) -> ProviderResponse:
        if not is_planning_request(dto):
            with self._lock:
                self._calls.append((self._route_of(dto), provider_id, dto.thinking_mode))
        return self._answers.send(dto)

    def get_context_limit(self) -> int:
        return self._answers.get_context_limit()

    def on(self, route: str) -> list[Call]:
        """Every call on *route*, in arrival order; fails on none, so a check
        over them can never pass vacuously."""
        with self._lock:
            found = [(provider_id, level) for r, provider_id, level in self._calls if r == route]
        assert found, f"no call on the {route} route"
        return found


class _Client:
    def __init__(self, lanes: _Lanes, provider_id: int) -> None:
        self._lanes = lanes
        self._provider_id = provider_id

    def get_context_limit(self) -> int:
        return self._lanes.get_context_limit()

    def abort(self) -> None:
        """Nothing in flight to abort: every call answers at once or on its gate."""

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        return self._lanes.send(dto, self._provider_id)


@contextmanager
def _serving(lanes: _Lanes, release: Callable[[], None] = lambda: None) -> Generator[None, None, None]:
    """Serve every provider call through *lanes* for the block; on the way out
    open every held call and wait for every turn to end."""
    with patch(BUILD_CLIENT, side_effect=lanes.build):
        try:
            yield
        finally:
            release()
            drain()


def _channel_lanes(provider: ChannelProvider) -> _Lanes:
    return _Lanes(cast("_Answers", provider), cast("Callable[[ProviderApiRequest], str]", provider.route))


def _chat(raw_input: str, turn_id: int = -1) -> MessageProcessor:
    """Start a chat turn — a new thread, or a reply into *turn_id*."""
    mp = MessageProcessor(UserConfig(), turn_id=turn_id, raw_input=raw_input)
    mp.begin()
    return mp


def _first_thread(client: FlaskClient, provider_id: int, level: str) -> int:
    """Run the chat's first thread, then set it to *provider_id* at *level*
    through the settings endpoint, so the next new thread copies them; return
    its turn id."""
    provider = ChannelProvider(chat=[text("Noted.")])
    with _serving(_channel_lanes(provider), provider.release_all):
        mp = _chat("remember that the plan is due on Friday")
        mp.result()
    turn_id = mp.turn_id
    _change(client, turn_id, provider_id, level)
    return turn_id


def _spawned(db: sqlite3.Connection, tool_name: str, kind: str) -> tuple[int | None, str]:
    """The saved settings of the one subagent thread the one *tool_name* call started."""
    [call] = calls_of(db, tool_name)
    started = child(db, kind, cast("int", call["id"]))
    return _stored(started.turn_id, started.channel)


def _the_coding_agent() -> Agent:
    [agent] = Agent.filter("type", CodeAgentAbility.NAME).get()
    return agent


def test_a_subtask_runs_on_its_caller_model_while_a_browse_keeps_the_delegate_model(
    authed_client: AuthedClient, models: tuple[int, int],
) -> None:
    """One step spawns a subtask and a browse from a thread on the second model
    at ``high``: the subtask's thread saves that model and level and runs on
    them; the browse's thread is not seeded from its caller and runs on the
    Delegate Provider's model — the default, with no delegate pinned — at its
    own level."""
    client, db, _store = authed_client
    default, other = models
    _first_thread(client, other, "high")
    provider = ChannelProvider(
        chat=[
            calls(
                tool(SubtaskAbility.NAME, action="start", instructions="count to three"),
                tool(WebBrowseAbility.NAME, instructions="read the title of the page"),
            ),
            text("Started both."),
        ],
        subtask=[text("One, two, three.")],
        web_browse=[text("The page is titled Home.")],
    )
    lanes = _channel_lanes(provider)

    with _serving(lanes, provider.release_all):
        _chat("count, and read the page").result()

    assert provider.unused() == {}
    assert set(lanes.on("chat")) == {(other, _HIGH)}
    assert _spawned(db, SubtaskAbility.NAME, _SUBTASK) == (other, "high")
    assert lanes.on("subtask") == [(other, _HIGH)]
    assert _spawned(db, WebBrowseAbility.NAME, _BROWSE) == (default, "auto")
    assert lanes.on("web_browse") == [(default, ThinkingLevel.LOW)]


def test_a_coding_agent_messaged_again_keeps_the_model_it_was_spawned_on(
    authed_client: AuthedClient, models: tuple[int, int],
) -> None:
    """The coding agent starts on its caller's model and level; messaged again
    once it has answered, its next turn opens a new thread on its own channel,
    which keeps them rather than the default."""
    client, db, _store = authed_client
    _default, other = models
    _first_thread(client, other, "high")

    def message_it() -> ProviderResponse:
        return calls(tool(MessageSubagentAbility.NAME, agent_id=_the_coding_agent().agent_id, message="now add a title"))

    provider = ChannelProvider(
        chat=[
            calls(tool(CodeAgentAbility.NAME, instructions="make a notes file")),
            text("Started."),
            _find_agent_tools(),
            message_it,
            text("Asked."),
            text("Noted."),
        ],
        code_agent=[text("I made the notes file."), text("I added a title.")],
    )
    lanes = _channel_lanes(provider)

    with _serving(lanes, provider.release_all):
        _chat("make me a notes file").result()

    assert provider.unused() == {}
    assert _spawned(db, CodeAgentAbility.NAME, _CODE) == (other, "high")
    assert _spawned(db, MessageSubagentAbility.NAME, _CODE) == (other, "high")
    assert lanes.on("code_agent") == [(other, _HIGH), (other, _HIGH)]


def _pass_of(dto: ProviderApiRequest) -> str:
    if _HANDOVER_MARK in dto.system:
        return "hand-over"
    if _COMPACTION_MARK in dto.system:
        return "compaction"
    return "turn"


def test_a_compaction_and_its_hand_over_run_on_the_model_of_the_turn_they_shrink(
    authed_client: AuthedClient, models: tuple[int, int],
) -> None:
    """A reply in a thread set to the second model overruns its context: the
    hand-over and the compaction that make room run on that model, not the
    default, and neither starts a thread of its own."""
    client, _db, _store = authed_client
    _default, other = models
    thread = _first_thread(client, other, "high")
    lanes = _Lanes(cast("_Answers", _HandoverProvider()), _pass_of)

    with _serving(lanes):
        _chat("a very long conversation", turn_id=thread).result()

    assert {provider_id for provider_id, _level in lanes.on("hand-over")} == {other}
    assert {provider_id for provider_id, _level in lanes.on("compaction")} == {other}
    assert Thread.filter("channel", Channel.COMPACTION.value).get() == []


def test_a_subtask_spawned_after_its_caller_model_was_deleted_heals_to_the_default(
    authed_client: AuthedClient, models: tuple[int, int],
) -> None:
    """The caller's model is deleted while its turn is mid-call; a subtask that
    turn then spawns keeps the caller's level, falls back to the default model,
    and stores the default on its thread so the fallback sticks."""
    client, db, _store = authed_client
    default, other = models
    _first_thread(client, other, "high")
    held = Held(calls(tool(SubtaskAbility.NAME, action="start", instructions="count to three")))
    provider = ChannelProvider(chat=[held, text("Started.")], subtask=[text("One, two, three.")])
    lanes = _channel_lanes(provider)

    with _serving(lanes, provider.release_all):
        mp = _chat("count to three")
        wait(held.entered, "the chat's first call")
        assert client.delete(f"/api/providers/{other}").status_code == 204
        held.gate.set()
        mp.result()

    assert held.opened
    assert provider.unused() == {}
    assert _spawned(db, SubtaskAbility.NAME, _SUBTASK) == (default, "high")
    assert lanes.on("subtask") == [(default, _HIGH)]
