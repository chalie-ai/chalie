"""Feature tests for the CapabilityAbility base, driven through the REAL dispatcher.

contacts is the exemplar migrated onto ``abilities/_capability.CapabilityAbility``
in TKT-883. These tests exercise the base's whole delegation flow the way
production does: a real ``ToolDispatcher`` resolves contacts from the real
registry, passes the real PolicyManager gate (contacts.list = allow on chat),
runs the base ``run()``, and the dispatcher renders the ToolResult into the
canonical wire envelope which is also written to the real ActTrail.

No mocks: in the test env the mail capability is genuinely not connected (no
CardDAV credentials), so every call exercises the base's structured
not-connected / unknown-action error paths for real.
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
        ("dmn", "user", "find John's number"),
    )
    db.commit()
    return cur.lastrowid


class _MP:
    """Minimal real MP-shaped context.

    DmnConfig (subconscious channel) is used for the same reason
    test_tool_dispatcher.py uses it: broadcast_to is None, so the act-event
    emitter is a real no-op (no live WebSocket needed), and contacts.list/get is
    seeded ``allow`` on the subconscious channel so the real policy gate passes.
    """

    def __init__(self, uid: int):
        self.config = DmnConfig()
        self.uid = uid
        self.DISCOVERABLE: list[str] = []
        self.active_tools: list[str] = []
        self.cancel_event = threading.Event()


def test_contacts_is_a_capability_ability():
    """The migrated contacts tool is wired on the shared base, not bespoke code."""
    contacts = AbilityRegistry.get("contacts")
    assert isinstance(contacts, CapabilityAbility)
    assert contacts.CAPABILITY_KEY == "mail"
    assert contacts.ACTION_HANDLERS == {"list": "list_contacts", "get": "get_contact"}


def test_not_connected_renders_structured_error_through_dispatcher(db):
    """Base not-connected path → ToolResult.err(code=not-connected) → wire envelope.

    The dispatcher resolves contacts, passes the gate, runs the base, and renders
    the structured error. The not-connected hint names the integration to fix,
    and the whole outcome is recorded on the trail.
    """
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("contacts", {"action": "list"})

    assert isinstance(result, str)
    assert result.startswith("[contacts(status=error, code=not-connected")
    assert "not connected" in result.lower()
    # The hint names the integration the user must configure.
    assert "mail integration" in result.lower()
    assert result.endswith("[end:contacts]")

    rows = ActTrail().fetch_by_transcript_id(transcript_id)
    assert [r["tool_name"] for r in rows] == ["contacts"]
    assert "not connected" in rows[0]["result"].lower()


def test_action_flows_through_to_rendered_meta(db):
    """A mapped action threads through the base into the rendered envelope meta.

    Dispatching contacts.get (also seeded allow) proves the base echoes the
    selected action into the ToolResult meta, which the dispatcher renders into
    the wire envelope — the action survives the whole pipeline, not just the
    happy-path 'list'.
    """
    transcript_id = _seed_transcript(db)
    mp = _MP(transcript_id)

    result = ToolDispatcher(mp).dispatch("contacts", {"action": "get", "identifier": "John"})

    assert isinstance(result, str)
    assert "action=get" in result
    assert "code=not-connected" in result
