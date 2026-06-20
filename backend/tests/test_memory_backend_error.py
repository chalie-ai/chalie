"""Feature test: a dead retrieval backend surfaces as a LOUD, stable error (TKT-886).

The audit finding this pins: ``memory.recall`` used to discard the backend error
status from its search lanes, so a dead store was indistinguishable from "no
memories". Under the ToolResult contract a backend failure MUST become
``ToolResult.err(code='memory-backend-error')``.
"""

import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig

pytestmark = pytest.mark.unit


class _MP:
    """Minimal real chat-channel mp — the dispatcher reads ``config`` (the policy
    channel) and ``uid`` (the act-trail anchor) off it."""

    def __init__(self, uid: int, config: object) -> None:
        self.config = config
        self.uid = uid


def _chat_mp(db: sqlite3.Connection) -> _MP:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("chat", "user", "what did we talk about at home"),
    )
    db.commit()
    return _MP(cast(int, cur.lastrowid), UserConfig({}))


def _kill_episodes_backend(db: sqlite3.Connection) -> None:
    db.execute("DROP TABLE IF EXISTS episodes")
    db.commit()


def test_dead_backend_recall_is_loud_error_not_zero_results(db: sqlite3.Connection) -> None:
    mp = _chat_mp(db)
    _kill_episodes_backend(db)

    out = ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "location": "home", "act_summary": "x"}
    )

    # Loud, stable, self-correcting error — the discriminator the model needs.
    assert "[memory(status=error, code=memory-backend-error" in out, (
        f"dead backend did not surface as a stable error: {out!r}"
    )
    # The silent-failure regression: it must NOT pretend nothing is stored.
    assert "status=success" not in out
    assert "results=0" not in out, f"dead backend masqueraded as empty: {out!r}"
    # The hint steers the model away from asserting "no record".
    assert "hint:" in out
    assert "infrastructure failure" in out


def test_dead_backend_error_is_recorded_on_the_act_trail(db: sqlite3.Connection) -> None:
    from services.act_trail import ActTrail

    mp = _chat_mp(db)
    _kill_episodes_backend(db)

    ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "location": "home", "act_summary": "x"}
    )

    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert trail, "no act-trail row recorded for the recall"
    assert "code=memory-backend-error" in cast(str, trail[-1]["result"]), (
        f"trail did not record the backend error: {trail[-1]['result']!r}"
    )
