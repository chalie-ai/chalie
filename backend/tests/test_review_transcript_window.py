"""``review_transcript``'s real windowed-fetch behaviour.

Its "working twin" ``review_tool_calls`` (see ``test_follow_up_dynamic_
interpolation.py``) only has coverage for ``get_follow_up``'s dynamic
interpolation — neither review tool has direct ``run()``/``_fetch`` coverage
today. ``_fetch`` builds its own ``Transcript`` query (channel allowlist +
``created_at`` window) independent of ``ReviewWindowAbility``'s shared flow, so
a regression there (e.g. a query-builder method silently changing shape) has
no other test that would catch it. This drives the real ``ReviewTranscriptAbility``
entry point (``Ability`` takes no ``mp`` — direct construction is the
established pattern, see ``test_timer_ability.py``) against real seeded rows.
"""

import sqlite3

import pytest

from abilities.review_transcript import ReviewTranscriptAbility
from models.transcript import Transcript

pytestmark = pytest.mark.unit

_ANCHOR = "2026-04-07T14:30:00+00:00"


def _seed(channel: str, content: str, created_at: str) -> None:
    Transcript(
        channel=channel, role="user", content=content,
        created_at=created_at, xml_migrated=1,
    ).save()


def _seed_window_rows(db: sqlite3.Connection) -> None:
    _seed("user", "approved email draft", "2026-04-07T14:30:00+00:00")  # in window
    _seed("subagent", "subagent task result", "2026-04-07T14:31:00+00:00")  # in window
    _seed("user", "old unrelated chat", "2026-04-07T14:00:00+00:00")  # 30 min before — out of window
    _seed("other", "irrelevant channel row", "2026-04-07T14:32:00+00:00")  # in window, wrong channel


def test_returns_only_user_rows_in_window_by_default(db: sqlite3.Connection) -> None:
    """Default call (no include_subagent_transcripts): only the in-window
    'user' row comes back — out-of-window and non-user/non-subagent channel
    rows are excluded."""
    _seed_window_rows(db)

    result = ReviewTranscriptAbility().run({"date_time": _ANCHOR})

    assert result.status == "success"
    assert result.meta["count"] == 1
    assert isinstance(result.body, list)
    assert result.body[0]["content"] == "approved email draft"
    assert result.body[0]["channel"] == "user"


def test_include_subagent_transcripts_adds_subagent_channel_rows(db: sqlite3.Connection) -> None:
    """Should-fire path: with include_subagent_transcripts=True, the in-window
    'subagent' row joins the result — still excluding the out-of-window and
    unrelated-channel rows."""
    _seed_window_rows(db)

    result = ReviewTranscriptAbility().run(
        {"date_time": _ANCHOR, "include_subagent_transcripts": True},
    )

    assert result.status == "success"
    assert result.meta["count"] == 2
    assert isinstance(result.body, list)
    contents = {row["content"] for row in result.body}
    assert contents == {"approved email draft", "subagent task result"}
