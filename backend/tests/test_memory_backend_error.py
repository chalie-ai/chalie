"""Feature test: a dead retrieval backend surfaces as a LOUD, stable error — not
a silent ``results=0`` (TKT-886).

The audit finding this pins: ``memory.recall`` used to discard the backend error
status from its search lanes, so a dead store was indistinguishable from "no
memories". A weak model, told ``results=0``, then confidently asserts the user
never said something it simply could not look up. Under the ToolResult contract a
backend failure MUST become ``ToolResult.err(code='memory-backend-error')``.

This is driven the production way — through the real ``ToolDispatcher(mp).
dispatch('memory', …)`` chokepoint on the chat channel — against a genuinely
broken backend: the ``episodes`` table is DROPPED from the real test database, so
the real ``location`` recall lane raises a real ``sqlite3.OperationalError`` and
reports a backend error. Zero mocks: the failure is a real dead store, the error
classification is the real handler, and the rendered envelope is the real one the
model would receive.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig

pytestmark = pytest.mark.unit


class _MP:
    """Minimal real chat-channel mp — the dispatcher reads ``config`` (the policy
    channel) and ``uid`` (the act-trail anchor) off it."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


def _chat_mp(db) -> _MP:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("chat", "user", "what did we talk about at home"),
    )
    db.commit()
    return _MP(cur.lastrowid, UserConfig({}))


def _kill_episodes_backend(db) -> None:
    """Drop the episodes table so the recall lane hits a real dead backend."""
    db.execute("DROP TABLE IF EXISTS episodes")
    db.commit()


def test_dead_backend_recall_is_loud_error_not_zero_results(db):
    """A location recall against a dropped ``episodes`` table renders a
    ``status=error, code=memory-backend-error`` envelope — never ``results=0``."""
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


def test_dead_backend_error_is_recorded_on_the_act_trail(db):
    """The loud error is what gets recorded on the trail — so the model reading the
    trail sees the failure, not a phantom empty result."""
    from services.act_trail import ActTrail

    mp = _chat_mp(db)
    _kill_episodes_backend(db)

    ToolDispatcher(mp).dispatch(
        "memory", {"action": "recall", "location": "home", "act_summary": "x"}
    )

    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert trail, "no act-trail row recorded for the recall"
    assert "code=memory-backend-error" in trail[-1]["result"], (
        f"trail did not record the backend error: {trail[-1]['result']!r}"
    )
