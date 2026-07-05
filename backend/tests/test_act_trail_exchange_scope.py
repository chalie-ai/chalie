"""Feature test for the exchange-scoped act-trail (the reply/fork boundary).

A turn can hold more than one exchange: a reply forks into an already-answered
turn, or an async result re-enters it — each a fresh MP recursion instance with
its own input row, all sharing one ``turn_id``. The rendered act-trail must show
only the CURRENT exchange's tool calls; a prior exchange's raw params/results
must leave context the moment that exchange synthesised (their answer carries
forward as the assistant synthesis in Previous Messages, not as a re-loaded raw
trail). This pins that boundary against the real ``PromptService.act_trail`` (via
``ToolCallService.by_exchange`` → ``ToolCall.by_exchange``) on the real DB.

Scenario mirrored from the design: turn 1 asks the weather (exchange 1, one
weather call), the user replies "similar temperature to us?" into the same turn
(exchange 2, a second weather call). Exchange 2's context must carry only its own
call.
"""

import json
import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _bare_mp(channel: str, turn_id: int, uid: int) -> "MessageProcessor":
    """A stand-in exercising the real ``PromptService.act_trail`` read path —
    ``channel``/``turn_id``/``uid`` are the only mp state it touches (via
    ``ToolCallService.by_exchange``). ``uid`` is the current exchange's input
    row: the fork boundary. Everything else the real ctor wires is irrelevant to
    the render and is skipped via ``__new__``."""
    from controllers.message_processor import MessageProcessor
    from services.tool_call_service import ToolCallService

    mp = object.__new__(MessageProcessor)
    mp.channel = channel
    mp.turn_id = turn_id
    mp.uid = uid
    mp.tool_call_service = ToolCallService(mp)
    return mp


def _input_row(channel: str, turn_id: int, content: str) -> int:
    from models.transcript import Transcript

    return cast("int", Transcript(
        channel=channel, role="user", content=content, turn_id=turn_id,
        settled=0, xml_migrated=1, deliberation_score=0.0,
    ).save().id)


def _assistant_row(channel: str, turn_id: int, content: str) -> int:
    from models.transcript import Transcript

    return cast("int", Transcript(
        channel=channel, role="assistant", content=content, turn_id=turn_id,
        settled=1, xml_migrated=1, deliberation_score=0.0,
    ).save().id)


def _tool_call(transcript_id: int, tool_name: str, params: object, result: str) -> None:
    from models.tool_call import ToolCall
    from services.time_utils import utc_now

    now = utc_now().isoformat()
    ToolCall(
        transcript_id=transcript_id, tool_name=tool_name, params=json.dumps(params),
        result=result, summary="", created_at=now, ended_at=now, state=ToolCall.DONE,
    ).save()


def test_reply_exchange_sees_only_its_own_tool_calls(db: sqlite3.Connection) -> None:
    """Step 6's context carries the reply's weather call, not exchange 1's."""
    ch = "user"
    turn = 1

    # Exchange 1 — "what's the weather?" (steps 1-3)
    _input_row(ch, turn, "hello, what's the weather?")            # step 1 (a prior MP instance's uid)
    a1 = _assistant_row(ch, turn, "Let me check.")
    _tool_call(a1, "weather", {"loc": "here"}, "24C here")        # step 2
    _assistant_row(ch, turn, "It's 24C here.")                    # step 3 synthesis

    # Exchange 2 — reply into the SAME turn (steps 4-6)
    r4 = _input_row(ch, turn, "Does Sicily have a similar temperature to us?")  # step 4 = this exchange's uid
    a2 = _assistant_row(ch, turn, "Checking Sicily.")
    _tool_call(a2, "weather", {"loc": "Sicily"}, "26C Sicily")    # step 5

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=r4)).act_trail()

    assert "Sicily" in rendered, "the reply's own tool call is missing from its context"
    assert "26C Sicily" in rendered
    assert '"loc": "here"' not in rendered and "24C here" not in rendered, (
        "exchange 1's raw tool call leaked into the reply exchange's context"
    )


def test_non_forked_turn_renders_whole_trail(db: sqlite3.Connection) -> None:
    """A single-exchange turn is unchanged: the exchange floor is its input row,
    the turn's lowest id, so every call still renders (no regression)."""
    ch = "user"
    turn = 1

    uid = _input_row(ch, turn, "research the news")
    step = _assistant_row(ch, turn, "checking sources")
    _tool_call(step, "web_search", {"q": "news"}, "first result")
    _tool_call(step, "web_browse", {"url": "https://x"}, "second result")

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=uid)).act_trail()

    assert "first result" in rendered
    assert "second result" in rendered
