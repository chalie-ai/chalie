"""Feature tests for the three capability-backed Ability wrappers.

Real-world assertions only — no mocks of in-process production code.

What each test asserts:
  1. All three abilities are discovered by AbilityRegistry with the correct
     NAME, SUMMARY, and EXAMPLES count (6–8 enforced by _base.py metaclass).
  2. When the mail capability is not connected (no credentials in test env),
     each ability returns a structured ``ToolResult.err`` (code=not-connected) —
     email/calendar via the shared CapabilityAbility base, contacts from its own
     ``_not_connected`` fallback — not an exception, not a JSON body.
  3. MailCapability scheduler-driven sync is the authoritative sync path
     (not the subconscious worker).

Context: MailCapability is discovered at import time but is_connected()
returns False in test environments (no credentials configured). This means
every execute() call on these abilities exercises the not-connected error
path without requiring real IMAP/CalDAV/CardDAV credentials.
"""

from typing import cast

import pytest
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# 1. Ability registration
# ---------------------------------------------------------------------------


def test_email_calendar_contacts_are_registered() -> None:
    """All three abilities are registered in AbilityRegistry (they are
    DISCOVERABLE=False — pim-delegate-exclusive — but still registered and
    reachable when pinned on PimConfig)."""
    from abilities._registry import AbilityRegistry
    from abilities._ability import Ability

    for name in ("email", "calendar", "contacts"):
        ability = AbilityRegistry.get(name)
        assert isinstance(ability, Ability), f"{name} is not an Ability subclass"
        assert ability.get_name() == name
        assert isinstance(ability.get_summary(), str) and ability.get_summary()
        assert isinstance(ability.get_parameters(), dict)
        # 6–8 keeps the find_tools embedding/FTS index balanced across abilities.
        assert 6 <= len(ability.get_examples()) <= 8, (
            f"{name}.get_examples() has {len(ability.get_examples())} entries, expected 6–8"
        )


# ---------------------------------------------------------------------------
# 2. Not-connected error path — each ability returns structured error
# ---------------------------------------------------------------------------


def test_email_not_connected_returns_structured_error() -> None:
    """EmailAbility returns a ToolResult not-connected error when mail is not connected.

    email migrated onto CapabilityAbility in , so the not-connected
    surface is the base class's ``ToolResult.err`` (status=error,
    code=not-connected, hint naming the integration) — the canonical contract
    form, no longer the legacy JSON ``{status: error}`` body. In the test
    environment no IMAP credentials are configured, so is_connected() returns
    False and ``search`` (a valid action needing no params) reaches the gate.
    """
    from abilities.email import EmailAbility
    from contracts.params.capability_params_bag import CapabilityParamsBag

    result = EmailAbility().run(built(CapabilityParamsBag.from_params({"action": "search"})))
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()


def test_calendar_write_not_connected_returns_tool_result_error() -> None:
    """CalendarAbility write ops return a ToolResult not-connected error.

    calendar migrated onto CapabilityAbility in , so the not-connected
    surface is the base class's ``ToolResult.err`` (status=error,
    code=not-connected, hint naming the integration) — the canonical contract
    form. Reads (list_events/get_event) query scheduled_items and are covered
    against a real DB in test_ability_calendar_tool_result.py; without a DB
    they raise, and the dispatcher owns unhandled-exception wrapping.
    """
    from abilities.calendar import CalendarAbility
    from contracts.params.capability_params_bag import CapabilityParamsBag

    result = CalendarAbility().run(
        built(CapabilityParamsBag.from_params(
            {"action": "update_event", "uid": "test-123", "summary": "New title"}
        ))
    )
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()


def test_contacts_not_connected_returns_structured_error() -> None:
    """ContactsAbility returns a structured ToolResult error when mail is not connected.

    contacts owns its not-connected surface directly (it no longer extends
    CapabilityAbility since the ParamBag migration): a first-class
    ``ToolResult.err`` (status=error, code=not-connected, hint naming the
    integration) — the same canonical contract form email and calendar get from
    the shared base. An empty index with mail unconnected is the remediation
    case its ``_not_connected`` fallback exists for.
    """
    from abilities.contacts import ContactsAbility
    from contracts.params.contacts_params_bag import ContactsParamsBag

    result = ContactsAbility().run(built(ContactsParamsBag.from_params({"action": "list"})))
    assert result.status == "error"
    assert result.code == "not-connected"
    assert "not connected" in cast(str, result.body).lower()
    assert "mail integration" in (result.hint or "").lower()
