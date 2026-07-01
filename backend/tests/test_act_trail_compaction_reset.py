"""Feature test for the mid-turn act-trail reset on compaction.

When tool results bloat the act-trail past the context window, the over-cap
handler fires ``chat_history_compactor`` and resumes. For the resume to actually
fit (instead of re-sending the same bloated trail and looping), the model's
rendered act-trail must drop every tool call at or before the turn's most recent
``chat_history_compactor`` marker — the compacted checkpoint carries prior
continuity, so the model re-fires what it still needs. This pins that reset
against the real ``_render_act_trail`` on the real DB.
"""

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _bare_mp(channel: str, turn_id: int) -> "MessageProcessor":
    """A MessageProcessor whose ``_render_act_trail`` alone is exercised.

    ``_render_act_trail`` reads only ``self._cfg.channel`` and ``self.turn_id``;
    the rest of the constructor (input-row allocation, providers, flashback) is
    irrelevant to the render and is skipped via ``__new__`` (precedent:
    test_ability_compactors._make_mp). The config is a real UserConfig so the
    channel is the real 'user'."""
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor

    mp = object.__new__(MessageProcessor)
    mp.config = UserConfig()
    mp.turn_id = turn_id
    return mp


def test_act_trail_resets_to_calls_after_the_latest_compaction_marker(
    db: sqlite3.Connection,
) -> None:
    """Only tool calls emitted AFTER the turn's most recent compaction marker render.

    Seeds a real user turn's trail with [search, compactor, browse] and asserts
    the render keeps the post-compaction browse and drops the pre-compaction
    search (and the compactor row itself). This is the reset that lets a
    bloated-trail over-cap recover instead of re-sending the same trail."""
    from services.act_trail import ActTrail
    from services.transcript_service import Transcript

    ch = "user"
    input_id = Transcript.write_input_row(ch, "user", "research the news")
    turn_id = Transcript.turn_id_of_row(input_id)
    step = Transcript.write_assistant_row(ch, "checking sources", turn_id=turn_id)

    ActTrail().record(
        tool_name="web_search", params={"q": "news"}, result="BEFORE search result",
        transcript_id=step, summary="searched",
    )
    ActTrail().record(
        tool_name="chat_history_compactor", params={}, result="compacted",
        transcript_id=step,
    )
    ActTrail().record(
        tool_name="web_browse", params={"url": "https://x"}, result="AFTER browse result",
        transcript_id=step, summary="browsed",
    )

    rendered = _bare_mp(ch, turn_id)._render_act_trail()

    assert "AFTER browse result" in rendered, "post-compaction tool result dropped"
    assert "BEFORE search result" not in rendered, (
        "pre-compaction tool result survived compaction — trail did not reset"
    )
    assert "chat_history_compactor" not in rendered


def test_act_trail_renders_the_whole_turn_when_no_compaction_fired(
    db: sqlite3.Connection,
) -> None:
    """With no compaction marker, the whole turn's tool calls render (byte-identical
    to pre-reset behaviour) — ordinary multi-step loops keep their tool-result
    continuity."""
    from services.act_trail import ActTrail
    from services.transcript_service import Transcript

    ch = "user"
    input_id = Transcript.write_input_row(ch, "user", "research the news")
    turn_id = Transcript.turn_id_of_row(input_id)
    step = Transcript.write_assistant_row(ch, "checking sources", turn_id=turn_id)

    ActTrail().record(
        tool_name="web_search", params={"q": "news"}, result="first result",
        transcript_id=step, summary="searched",
    )
    ActTrail().record(
        tool_name="web_browse", params={"url": "https://x"}, result="second result",
        transcript_id=step, summary="browsed",
    )

    rendered = _bare_mp(ch, turn_id)._render_act_trail()

    assert "first result" in rendered
    assert "second result" in rendered
