"""Feature test: the person's chat never plans and has every subagent ready;
the subagents plan instead.

A chat turn — the main chat and a reply inside a thread alike — goes straight
to work: no planning request reaches the model, no request it sends carries a
plan or an instruction to make one, and ``plan`` is never offered. Every
request of the turn offers the five subagents (``subtask``, ``code_agent``,
``web_browse``, ``web_search``, ``pim``) from the first step, with no
``find_tools`` look-up. A subtask the chat starts plans its task first,
offered only ``recall`` and ``plan``, then works it with the plan in view, and
is offered none of the five. Neither a discovery pass nor an outside agent's
message plans or is offered them.

The model stand-in plans whenever it is asked to, so a turn that wrongly plans
stores a plan the thread read-back would show.
"""

import json
import sqlite3
from collections.abc import Callable
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.pim import PimAbility
from abilities.plan import PlanAbility
from abilities.recall import Recall
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels.discovery import DiscoveryConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from models.plan_task import PlanTask
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.processor_config import ProcessorConfig
from services.prompt_service import _NO_PLAN, _PLAN_INSTRUCTION, _PLANNING_INSTRUCTION
from tests._delegates import (
    USER,
    ChannelProvider,
    Held,
    Request,
    assistant_texts,
    block,
    calls,
    calls_of,
    child,
    drain,
    messages,
    patched,
    replies,
    run_turn,
    text,
    tool,
)
from tests.helpers import is_planning_request
from tests.test_turn_join import _post, _result

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_SUBAGENTS = frozenset({
    SubtaskAbility.NAME, CodeAgentAbility.NAME, WebBrowseAbility.NAME, WebSearchAbility.NAME, PimAbility.NAME,
})
#: What the stand-in plans whenever it is asked to.
_STEP = "Sieve"
_STEP_DETAIL = "Cross out the multiples of each prime up to the limit"
_TASK = "list the primes below 6"
_PRIMES = "2, 3, 5"


class _Planner(ChannelProvider):
    """The shared stand-in as a model that plans whenever it is asked to: it
    answers a planning request by creating a one-task plan, and keeps each
    planning request it answers under the route ``planning``."""

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        if not is_planning_request(dto):
            return super().send(dto)
        names = frozenset(str(t["name"]) for t in dto.tools or [])
        with self._lock:
            self.requests.append(Request("planning", json.dumps(dto.messages, default=str), names))
        return calls(tool(PlanAbility.NAME, action="create", tasks=[{"title": _STEP, "task": _STEP_DETAIL}]))


def _plan_text_sent(provider: ChannelProvider) -> list[str]:
    """Every request, on any route, that showed the model a plan or asked it
    to make one."""
    routes = {r.route for r in provider.requests}
    return [
        read for route in sorted(routes) for read in provider.read(route)
        if any(marker in read for marker in (_PLANNING_INSTRUCTION, _NO_PLAN, _PLAN_INSTRUCTION, "[plan]\n"))
    ]


def test_a_chat_message_and_a_reply_in_its_thread_never_plan_and_offer_every_subagent(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """A message opens a thread and a reply follows inside it; each takes a
    recall step, then answers. No planning request is sent for either, no
    request carries plan text, every request offers all five subagents and
    never ``plan``, no plan is stored on either message, and the thread read
    back shows neither with a plan."""
    client, _db, _store = authed_client
    provider = _Planner(chat=[
        calls(tool(Recall.NAME, query="capital of France")),
        text("Paris."),
        calls(tool(Recall.NAME, query="capital of Italy")),
        text("Rome."),
    ])

    with patched(provider):
        turn_id = cast("int", _result(_post(client, -1, "what is the capital of France?"))["turn_id"])
        drain()
        _result(_post(client, turn_id, "and of Italy?"))
        drain()

    assert provider.unused() == {}
    assert [answer for answer in assistant_texts(db, USER, turn_id) if answer] == ["Paris.", "Rome."]
    assert provider.offered("planning") == []
    offered = provider.offered("chat")
    assert len(offered) == 4, offered
    for names in offered:
        assert _SUBAGENTS <= names, names
        assert PlanAbility.NAME not in names, names
    assert _plan_text_sent(provider) == []

    inputs = [cast("int", r["id"]) for r in Transcript.by_turn(USER, turn_id) if r["role"] == "user"]
    assert len(inputs) == 2
    for message_id in inputs:
        assert PlanTask.for_transcript(message_id) == []
        assert PlanAbility.NAME not in [c.tool_name for c in ToolCall.by_transcript(message_id)]
    users = [m for m in messages(block(client, f"/api/threads/{turn_id}")) if m["role"] == "user"]
    assert [m["content"] for m in users] == ["what is the capital of France?", "and of Italy?"]
    assert [m.get("plan") for m in users] == [None, None]


def test_a_subtask_the_chat_starts_plans_its_task_and_is_offered_no_subagent(db: sqlite3.Connection) -> None:
    """The chat starts a subtask and every chat request offers the subagents.
    The subtask's first request is its planning request, offering only
    ``recall`` and ``plan`` and carrying the task; its next request shows the
    plan, and it answers in the response that closes it. No request of the
    subtask offers a subagent. The plan is stored done on the task's row, and
    the answer reaches the chat."""
    held = Held(ProviderResponse(
        text=_PRIMES, model="m", tool_calls=[tool(PlanAbility.NAME, action="update", task=1, status="done")],
    ))
    provider = _Planner(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK)),
            text("Started it."),
            text("Got the reply."),
        ],
        subtask=[held],
    )

    run_turn(provider, UserConfig(), "find the primes below 6 in the background")

    assert held.opened
    assert provider.unused() == {}
    assert all(_SUBAGENTS <= names for names in provider.offered("chat")), provider.offered("chat")
    assert provider.offered("planning") == [frozenset({Recall.NAME, PlanAbility.NAME})]
    [planning] = provider.read("planning")
    assert _TASK in planning
    [step] = provider.read("subtask")
    assert f"1. [todo] {_STEP} — {_STEP_DETAIL}" in step
    [subtask_tools] = provider.offered("subtask")
    assert not subtask_tools & _SUBAGENTS, subtask_tools

    call_id = cast("int", calls_of(db, SubtaskAbility.NAME)[0]["id"])
    [task_row] = [
        cast("int", r["id"]) for r in Transcript.by_turn(*child(db, _SUBTASK, call_id)) if r["role"] == "user"
    ]
    assert [(t.title, t.status) for t in PlanTask.for_transcript(task_row)] == [(_STEP, "done")]
    [reply] = replies(db, call_id)
    assert reply["content"] == _PRIMES


@pytest.mark.parametrize(
    ("make_config", "ask"),
    [
        pytest.param(DiscoveryConfig, "run the research pass", id="discovery"),
        pytest.param(
            lambda: EAMPConfig("reviewer", "sample-project", False),
            "what did we decide about the release date?",
            id="outside_agent",
        ),
    ],
)
def test_discovery_and_an_outside_agent_never_plan_and_are_offered_no_subagent(
    db: sqlite3.Connection, make_config: Callable[[], ProcessorConfig], ask: str,
) -> None:
    """A discovery pass, and an outside agent's message, each take a recall
    step, then answer: no planning request, no plan text, no plan stored, and
    neither ``plan`` nor any subagent offered on any request."""
    provider = _Planner(chat=[calls(tool(Recall.NAME, query="what the person cares about")), text("Nothing new.")])

    mp, _frames = run_turn(provider, make_config(), ask)

    assert provider.unused() == {}
    assert provider.offered("planning") == []
    offered = provider.offered("chat")
    assert len(offered) == 2, offered
    for names in offered:
        assert not names & _SUBAGENTS, names
        assert PlanAbility.NAME not in names, names
    assert _plan_text_sent(provider) == []
    assert mp.uid is not None
    assert PlanTask.for_transcript(mp.uid) == []
