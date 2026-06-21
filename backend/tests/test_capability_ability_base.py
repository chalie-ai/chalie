"""Feature tests for the CapabilityAbility base, driven through the REAL dispatcher.

``email`` is the exemplar that still runs the base's whole delegation flow: it
maps the model's ``action`` onto a mail-capability handler, refuses through the
base when the capability is not connected, and renders the structured error. (The
 exemplar, ``contacts``, was migrated to inline local-index reads in
 and no longer reaches the base's handler-dispatch path — so the base flow
is now exercised against ``email``, which keeps ``ACTION_HANDLERS`` and the full
``super().run()`` path.)

No mocks: in the test env the mail capability is genuinely not connected (no
credentials), so every call exercises the base's structured not-connected /
action-meta error paths for real.
"""

import sqlite3
import threading
from typing import cast

import pytest

from abilities._capability import CapabilityAbility
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection) -> int:
    """Insert a user message transcript row for test setup."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "check my email"),
    )
    db.commit()
    return cast(int, cur.lastrowid)


class _MP:

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_email_is_a_capability_ability() -> None:
    email = AbilityRegistry.get("email")
    assert isinstance(email, CapabilityAbility)
    assert email.CAPABILITY_KEY == "mail"
    assert email.ACTION_HANDLERS["search"] == "search_email"
    assert email.ACTION_HANDLERS["read"] == "read_email"


def test_not_connected_renders_structured_error_through_dispatcher(db: sqlite3.Connection) -> None:
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("email", {"action": "search"})

    assert isinstance(result, str)
    assert result.startswith("[email(status=error, code=not-connected")
    assert "not connected" in result.lower()
    # The hint names the integration the user must configure.
    assert "mail integration" in result.lower()
    assert result.endswith("[end:email]")

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["email"]
    assert "not connected" in cast(str, rows[0]["result"]).lower()


def test_action_flows_through_to_rendered_meta(db: sqlite3.Connection) -> None:
    """A mapped action threads through the base into the rendered envelope meta.

    Dispatching email.read (also seeded allow) proves the base echoes the
    selected action into the ToolResult meta, which the dispatcher renders into
    the wire envelope — the action survives the whole pipeline, not just the
    happy-path 'search'.
    """
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("email", {"action": "read", "uid": 1})

    assert isinstance(result, str)
    assert "action=read" in result
    assert "code=not-connected" in result
