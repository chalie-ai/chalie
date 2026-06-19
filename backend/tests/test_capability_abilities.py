"""Feature tests for the three capability-backed Ability wrappers.

Real-world assertions only — no mocks of in-process production code.

What each test asserts:
  1. All three abilities are discovered by AbilityRegistry with the correct
     NAME, SUMMARY, and EXAMPLES count (6–8 enforced by _base.py metaclass).
  2. When the mail capability is not connected (no credentials in test env),
     each ability returns a structured ``ToolResult.err`` (code=not-connected)
     from the shared CapabilityAbility base — not an exception, not a JSON body.
  3. MailCapability scheduler-driven sync is the authoritative sync path
     (not the subconscious worker).

Context: MailCapability is discovered at import time but is_connected()
returns False in test environments (no credentials configured). This means
every execute() call on these abilities exercises the not-connected error
path without requiring real IMAP/CalDAV/CardDAV credentials.
"""

from typing import cast

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# 1. Ability registration
# ---------------------------------------------------------------------------


def test_email_calendar_contacts_are_registered() -> None:
    """All three new abilities are discoverable via AbilityRegistry."""
    from abilities._registry import AbilityRegistry
    from abilities._ability import Ability

    for name in ("email", "calendar", "contacts"):
        ability = AbilityRegistry.get(name)
        assert isinstance(ability, Ability), f"{name} is not an Ability subclass"
        assert ability.get_name() == name
        assert isinstance(ability.get_summary(), str) and ability.get_summary()
        assert isinstance(ability.get_parameters(), dict)
        # EXAMPLES count is enforced by __init_subclass__ (6–8).
        assert 6 <= len(ability.get_examples()) <= 8, (
            f"{name}.get_examples() has {len(ability.get_examples())} entries, expected 6–8"
        )


# ---------------------------------------------------------------------------
# 2. Not-connected error path — each ability returns structured error
# ---------------------------------------------------------------------------


def test_email_not_connected_returns_structured_error() -> None:
    """EmailAbility returns a ToolResult not-connected error when mail is not connected.

    email migrated onto CapabilityAbility in TKT-889, so the not-connected
    surface is the base class's ``ToolResult.err`` (status=error,
    code=not-connected, hint naming the integration) — the canonical contract
    form, no longer the legacy JSON ``{status: error}`` body. In the test
    environment no IMAP credentials are configured, so is_connected() returns
    False and ``search`` (a valid action needing no params) reaches the gate.
    """
    from abilities.email import EmailAbility

    result = EmailAbility().run({"action": "search"})
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()


def test_calendar_write_not_connected_returns_tool_result_error() -> None:
    """CalendarAbility write ops return a ToolResult not-connected error.

    calendar migrated onto CapabilityAbility in TKT-888, so the not-connected
    surface is the base class's ``ToolResult.err`` (status=error,
    code=not-connected, hint naming the integration) — the canonical contract
    form. Reads (list_events/get_event) query scheduled_items and are covered
    against a real DB in test_ability_calendar_tool_result.py; without a DB
    they raise, and the dispatcher owns unhandled-exception wrapping.
    """
    from abilities.calendar import CalendarAbility

    result = CalendarAbility().run(
        {"action": "update_event", "uid": "test-123", "summary": "New title"}
    )
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()


def test_contacts_not_connected_returns_structured_error() -> None:
    """ContactsAbility returns a structured ToolResult error when mail is not connected.

    contacts is the TKT-883 exemplar migrated onto CapabilityAbility, so its
    not-connected surface is now a first-class ``ToolResult.err`` (status=error,
    code=not-connected, hint naming the integration) rather than a JSON body —
    the canonical contract form. calendar followed in TKT-888 and email in
    TKT-889; all three capability abilities now share the base's surface.
    """
    from abilities.contacts import ContactsAbility

    result = ContactsAbility().run({"action": "list"})
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()


# ---------------------------------------------------------------------------
# 3. Scheduler-driven sync is the authoritative sync path
# ---------------------------------------------------------------------------


def test_subconscious_worker_owns_capability_sync() -> None:
    """The subconscious worker drives capability syncs via _step_capability_sync.

    The worker calls each connected capability's monitor() method on every
    tick. MailCapability._do_monitor() handles per-protocol cadence internally.
    The scheduler is NOT involved in triggering syncs.
    """
    from services.subconscious_worker import SubconsciousWorker

    assert hasattr(SubconsciousWorker, "_step_capability_sync")
    assert not hasattr(SubconsciousWorker, "_step_calendar_sync")
    assert not hasattr(SubconsciousWorker, "_step_contacts_sync")


def test_mail_capability_has_no_scheduler_sync_registration() -> None:
    """MailCapability must NOT register a scheduler handler for syncs.

    Syncs are driven by the subconscious worker, not the scheduler.
    The scheduler only stores calendar event data.
    """
    from capabilities.mail_capability.capability import MailCapability

    assert not hasattr(MailCapability, "_ensure_sync_registration") or not callable(getattr(MailCapability, "_ensure_sync_registration", None))
    cap = MailCapability()
    assert not hasattr(cap, "_sync_registered")
