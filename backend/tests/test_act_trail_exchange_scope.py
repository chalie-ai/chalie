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
    from services.transcript_service import TranscriptService

    mp = object.__new__(MessageProcessor)
    mp.channel = channel
    mp.turn_id = turn_id
    mp.uid = uid
    mp.transcript_service = TranscriptService(mp)
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


def test_seed_recall_renders_as_background_memory_block(db: sqlite3.Connection) -> None:
    """The turn-0 auto-seed recall (``"_auto": true`` params) renders as a
    ``[background_memory]`` context block — same envelope body, relabeled
    wrapper, no ``[memory] {params} → …`` tool-call row. An explicit
    model-invoked recall on the same trail still renders as a normal row."""
    ch = "user"
    turn = 1

    uid = _input_row(ch, turn, "what's on today?")
    step = _assistant_row(ch, turn, "checking")
    seed_env = '[recall(status=success, query=x, results=1)]\n{"results":[{"id":"residence"}]}\n[end:recall]'
    _tool_call(
        step, "recall",
        {"query": "what's on today?", "_auto": True},
        seed_env,
    )
    explicit_env = "[recall(status=error, code=no-results)]\nNo results found.\n[end:recall]"
    _tool_call(step, "recall", {"query": "wifi password"}, explicit_env)

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=uid)).act_trail()

    assert "[background_memory]" in rendered
    assert "[end:background_memory]" in rendered
    assert '{"results":[{"id":"residence"}]}' in rendered
    assert '"_auto"' not in rendered, "the seed rendered as a tool-call row (params leaked)"
    assert '[recall] {"query": "wifi password"}' in rendered, (
        "the explicit recall no longer renders as a normal tool-call row"
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


def test_interim_prose_interleaves_in_emission_order(db: sqlite3.Connection) -> None:
    """Before the first tool call anchored to each assistant row of the current
    exchange, ``act_trail`` renders ``[interim_response] <row content with
    newlines flattened>``. Calls anchored directly to the input row render
    without an interim line; an assistant row anchoring no calls (the final
    synthesis) never renders; a prior exchange's assistant prose never renders."""
    ch = "user"
    turn = 1

    uid = _input_row(ch, turn, "find me flights")
    _tool_call(uid, "weather", {"loc": "here"}, "24C")                          # uid-anchored call, no interim
    a1 = _assistant_row(ch, turn, "Checking flights now")
    _tool_call(a1, "web_search", {"q": "flights"}, "found flights")             # a1-anchored call
    a2 = _assistant_row(ch, turn, "Found two options.\nComparing prices.")
    _tool_call(a2, "web_browse", {"url": "https://x"}, "option A")              # a2-anchored call
    _assistant_row(ch, turn, "Here is your final answer")                       # synthesis row, no calls

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=uid)).act_trail()

    # uid-anchored call renders first; no interim line before it.
    assert rendered.index("[weather] {\"loc\": \"here\"} → 24C") == 0
    assert rendered.index("[interim_response]") > rendered.index("[weather] {\"loc\": \"here\"} → 24C")

    # Order: uid call → interim(a1) → a1 call → interim(a2) → a2 call.
    assert rendered.index("[interim_response] Checking flights now") < rendered.index("[web_search] {\"q\": \"flights\"} → found flights")
    assert rendered.index("[web_search] {\"q\": \"flights\"} → found flights") < rendered.index("[interim_response] Found two options. Comparing prices.")
    assert rendered.index("[interim_response] Found two options. Comparing prices.") < rendered.index("[web_browse] {\"url\": \"https://x\"} → option A")

    # Synthesis row (no calls) never renders.
    assert "Here is your final answer" not in rendered


def test_prior_exchange_interim_prose_does_not_leak(db: sqlite3.Connection) -> None:
    """An assistant row from a prior exchange — even one that anchored a call —
    must not surface when rendering the reply exchange. Only the reply's own
    interim prose appears as an ``[interim_response]`` line."""
    ch = "user"
    turn = 1

    # Exchange 1 — interim + call + synthesis.
    _input_row(ch, turn, "what's the weather?")
    a1 = _assistant_row(ch, turn, "Let me check.")
    _tool_call(a1, "weather", {"loc": "here"}, "24C here")
    _assistant_row(ch, turn, "It's 24C here.")                                  # exchange 1 synthesis

    # Exchange 2 — reply fork.
    r4 = _input_row(ch, turn, "Does Sicily have a similar temperature to us?")
    a2 = _assistant_row(ch, turn, "Checking Sicily.")
    _tool_call(a2, "weather", {"loc": "Sicily"}, "26C Sicily")

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=r4)).act_trail()

    # Exchange 1 content must not leak.
    assert "Let me check." not in rendered
    assert "It's 24C here." not in rendered
    assert "24C here" not in rendered

    # Exchange 2's own interim prose must render.
    assert "[interim_response] Checking Sicily." in rendered
    assert "Sicily" in rendered
    assert "26C Sicily" in rendered


def test_compaction_drops_preceding_interim_prose(db: sqlite3.Connection) -> None:
    """A chat_history_compactor call anchored to an assistant row drops that
    row's interim prose (and its ordinary calls); rows after the compactor
    marker still render with their interim prose."""
    ch = "user"
    turn = 1

    uid = _input_row(ch, turn, "do stuff")
    a_before = _assistant_row(ch, turn, "before compaction thought")
    _tool_call(a_before, "web_search", {"q": "x"}, "result x")                  # ordinary call before compactor
    _tool_call(a_before, "chat_history_compactor", {"reason": "mid-turn"}, "")  # compactor anchored to same row
    a_after = _assistant_row(ch, turn, "after compaction thought")
    _tool_call(a_after, "web_search", {"q": "y"}, "result y")                   # ordinary call after compactor

    from services.prompt_service import PromptService

    rendered = PromptService(_bare_mp(ch, turn, uid=uid)).act_trail()

    assert "before compaction thought" not in rendered
    assert "[interim_response] after compaction thought" in rendered
    assert "result y" in rendered
