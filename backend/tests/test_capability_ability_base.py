"""Feature tests for the CapabilityAbility base, driven through the REAL dispatcher.

``email`` is the exemplar that still runs the base's whole delegation flow: it
maps the model's ``action`` onto a mail-capability handler, refuses through the
base when the capability is not connected, and renders the structured error. (The
TKT-883 exemplar, ``contacts``, was migrated to inline local-index reads in
TKT-905 and no longer reaches the base's handler-dispatch path — so the base flow
is now exercised against ``email``, which keeps ``ACTION_HANDLERS`` and the full
``super().run()`` path.)

These tests exercise the base's delegation flow the way production does: a real
``ToolDispatcher`` resolves email from the real registry, passes the real
PolicyManager gate (email.search = allow on the subconscious channel), runs the
base ``run()``, and the dispatcher renders the ToolResult into the canonical wire
envelope which is also written to the real ActTrail.

No mocks: in the test env the mail capability is genuinely not connected (no
credentials), so every call exercises the base's structured not-connected /
action-meta error paths for real.
"""

import threading

import pytest

from abilities._capability import CapabilityAbility
from abilities._dispatcher import ToolDispatcher
from abilities._registry import AbilityRegistry
from configs.channels import DmnConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("dmn", "user", "check my email"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context.

    DmnConfig (subconscious channel) is used for the same reason
    test_tool_dispatcher.py uses it: broadcast_to is None, so the act-event
    emitter is a real no-op (no live WebSocket needed), and email.search/read is
    seeded ``allow`` on the subconscious channel so the real policy gate passes.
    """

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_email_is_a_capability_ability():
    """The email tool is wired on the shared base with its action→handler map."""
    email = AbilityRegistry.get("email")
    assert isinstance(email, CapabilityAbility)
    assert email.CAPABILITY_KEY == "mail"
    assert email.ACTION_HANDLERS["search"] == "search_email"
    assert email.ACTION_HANDLERS["read"] == "read_email"


def test_not_connected_renders_structured_error_through_dispatcher(db):
    """Base not-connected path → ToolResult.err(code=not-connected) → wire envelope.

    The dispatcher resolves email, passes the gate, runs the base, and renders
    the structured error. The not-connected hint names the integration to fix,
    and the whole outcome is recorded on the trail.
    """
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
    assert "not connected" in rows[0]["result"].lower()


def test_action_flows_through_to_rendered_meta(db):
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
