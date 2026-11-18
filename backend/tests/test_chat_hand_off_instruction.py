"""Feature test: only the person's chat is told to hand longer work to a subagent.

A turn whose subagent calls are answered later — the person's chat, its main
exchange and a reply inside a thread alike — is sent a system prompt with a
section on keeping the chat moving, which names the five subagent tools
(``code_agent``, ``web_browse``, ``web_search``, ``pim``, ``subtask``). Every
other turn waits for its subagents, so none is sent that section: not a
subtask the chat starts, not the own turn of a subagent the chat starts, not a
scheduled task or the subtask it starts, not a discovery pass — even though a
subtask the chat starts and a discovery pass share the chat's prompt body.

What each model is sent is read off the provider requests of real turns,
driven through the real production entry point against the real,
fully-migrated SQLite DB. The only substitution is the LLM network boundary
(``services.provider_service.build_client``); ``pim`` is connected to a
mailbox through the product's own setup route, against an IMAP peer on
loopback.
"""

import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.pim import PimAbility
from abilities.recall import Recall
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels.discovery import DiscoveryConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from services.processor_config import ProcessorConfig
from tests._delegates import (
    USER,
    ChannelProvider,
    Entry,
    Held,
    assistant_texts,
    calls,
    connected_mail,
    run_turn,
    text,
    tool,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_HEADING = "## Keeping the chat moving"
_SUBAGENTS = (
    CodeAgentAbility.NAME, WebBrowseAbility.NAME, WebSearchAbility.NAME, PimAbility.NAME, SubtaskAbility.NAME,
)
_SCHEDULE_ID = 7
_QUESTION = "what is the capital of France?"
_INSTRUCTIONS = "find the capital of France"
_ANSWER = "Paris is the capital of France."
_STARTED = "I have started it."
_ANSWERED = "Here is what it found."


def _named_in_section(system: str) -> list[str]:
    """The subagent tools named — in backticks, as a tool name is written — in
    the hand-off section of *system*, the text from its heading up to the next
    heading, in ``_SUBAGENTS`` order; none when *system* has no such section."""
    start = system.find(_HEADING)
    if start < 0:
        return []
    end = system.find("\n## ", start + len(_HEADING))
    section = system[start:] if end < 0 else system[start:end]
    return [name for name in _SUBAGENTS if f"`{name}`" in section]


def _user_turns(db: sqlite3.Connection) -> list[int]:
    """Every turn the person's chat holds rows in."""
    return [r[0] for r in db.execute("SELECT DISTINCT turn_id FROM transcript WHERE channel = ? ORDER BY 1", (USER,))]


def _start(name: str) -> dict[str, object]:
    """The call that starts the subagent *name*."""
    if name == SubtaskAbility.NAME:
        return tool(name, action="start", instructions=_INSTRUCTIONS)
    return tool(name, instructions=_INSTRUCTIONS)


def _provider(caller: Callable[[], ProcessorConfig], starts: str | None) -> ChannelProvider:
    """The model of every turn one case runs. A caller that starts nothing
    answers at once. A caller that starts the subagent *starts* says so; the
    chat's subagent answers once the chat's turn has ended (held until then)
    and the chat answers its reply, while a scheduled task waits for the
    answer and closes on it. A subagent's own turn is answered from the route
    named after its tool."""
    if starts is None:
        return ChannelProvider(chat=[text("Nothing new.")])
    if caller is UserConfig:
        late: list[Entry] = [Held(text(_ANSWER))]
        return ChannelProvider(chat=[calls(_start(starts)), text(_STARTED), text(_ANSWERED)], **{starts: late})
    waited: list[Entry] = [text(_ANSWER)]
    return ChannelProvider(chat=[calls(_start(starts)), text("Noted.")], **{starts: waited})


def _mailbox(client: FlaskClient, starts: str | None) -> AbstractContextManager[None]:
    """A connected mailbox for ``pim``, which refuses without one; nothing for
    any other case."""
    return connected_mail(client) if starts == PimAbility.NAME else nullcontext()


@pytest.mark.parametrize("in_thread", [False, True], ids=["main_chat", "thread_reply"])
def test_the_persons_chat_is_told_to_hand_longer_work_to_a_subagent(db: sqlite3.Connection, in_thread: bool) -> None:
    """A message to the chat — one that opens a thread, or a reply inside a
    thread that has already been answered — takes a recall step, then
    answers. Both requests of that turn are sent the hand-off section once, naming
    every one of the five subagent tools."""
    opening: list[Entry] = [text("Paris.")] if in_thread else []
    provider = ChannelProvider(chat=[*opening, calls(tool(Recall.NAME, query="capital of Italy")), text("Rome.")])

    thread = -1
    if in_thread:
        opener, _frames = run_turn(provider, UserConfig(), _QUESTION)
        thread = opener.turn_id
    mp, _frames = run_turn(provider, UserConfig(), "and the capital of Italy?", turn_id=thread)

    assert provider.unused() == {}
    assert _user_turns(db) == [mp.turn_id]
    assert assistant_texts(db, USER, mp.turn_id)[-1] == "Rome."
    sent = provider.systems("chat")[len(opening):]
    assert len(sent) == 2, sent
    for system in sent:
        assert system.count(_HEADING) == 1, system[-2500:]
        assert _named_in_section(system) == list(_SUBAGENTS), system[-2500:]


@pytest.mark.parametrize(
    ("caller", "turn_id", "starts", "under_test"),
    [
        *(pytest.param(UserConfig, -1, name, (name,), id=f"chat_started_{name}") for name in _SUBAGENTS),
        pytest.param(
            ScheduledConfig, _SCHEDULE_ID, SubtaskAbility.NAME, ("chat", SubtaskAbility.NAME),
            id="scheduled_task_and_its_subtask",
        ),
        pytest.param(DiscoveryConfig, -1, None, ("chat",), id="discovery"),
    ],
)
def test_no_turn_but_the_chat_is_told_to_hand_work_to_a_subagent(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    caller: Callable[[], ProcessorConfig],
    turn_id: int,
    starts: str | None,
    under_test: tuple[str, ...],
) -> None:
    """The turns under test — a subtask or a subagent the chat started, a
    scheduled task and the subtask it started, a discovery pass — each make at
    least one request, and none of them is sent the hand-off section."""
    client, _db, _store = authed_client
    provider = _provider(caller, starts)

    with _mailbox(client, starts):
        run_turn(provider, caller(), _QUESTION, turn_id=turn_id)

    assert provider.unused() == {}
    for route in under_test:
        sent = provider.systems(route)
        assert sent, f"the {route} turn sent no request"
        told = [system for system in sent if _HEADING in system]
        assert told == [], f"the {route} turn was told: {told[0][-2500:]}"
