"""Feature tests for the three capability-backed Ability wrappers.

Real-world assertions only — no mocks of in-process production code.

What each test asserts:
  1. All three abilities are discovered by AbilityRegistry with the correct
     NAME, SUMMARY, and EXAMPLES count (6–8 enforced by _base.py metaclass).
  2. When the mail capability is not connected (no credentials in test env),
     each ability returns a structured error dict — not an exception.
  3. MailCapability scheduler-driven sync is the authoritative sync path
     (not the subconscious worker).

Context: MailCapability is discovered at import time but is_connected()
returns False in test environments (no credentials configured). This means
every execute() call on these abilities exercises the not-connected error
path without requiring real IMAP/CalDAV/CardDAV credentials.
"""

import json

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(result: dict) -> dict:
    """Parse the JSON payload out of a tagged text result dict.

    The abilities return ``{"text": "[name(...)]\n<json>\n[end:name]"}``.
    This helper extracts and parses the JSON body.
    """
    text = result["text"]
    lines = text.split("\n")
    # First line is the tag opener, last is [end:name], middle is JSON.
    body = "\n".join(lines[1:-1])
    return json.loads(body)


# ---------------------------------------------------------------------------
# 1. Ability registration
# ---------------------------------------------------------------------------


def test_email_calendar_contacts_are_registered():
    """All three new abilities are discoverable via AbilityRegistry."""
    from abilities._registry import AbilityRegistry
    from abilities._base import Ability

    for name in ("email", "calendar", "contacts"):
        ability = AbilityRegistry.get(name)
        assert isinstance(ability, Ability), f"{name} is not an Ability subclass"
        assert ability.NAME == name
        assert isinstance(ability.SUMMARY, str) and ability.SUMMARY
        assert isinstance(ability.INPUT_SCHEMA, dict)
        # EXAMPLES count is enforced by _base.py's __init_subclass__ (6–8).
        assert 6 <= len(ability.EXAMPLES) <= 8, (
            f"{name}.EXAMPLES has {len(ability.EXAMPLES)} entries, expected 6–8"
        )


# ---------------------------------------------------------------------------
# 2. Not-connected error path — each ability returns structured error
# ---------------------------------------------------------------------------


def test_email_not_connected_returns_structured_error():
    """EmailAbility.execute returns {status: error} when mail is not connected.

    In the test environment no IMAP credentials are configured, so
    is_connected() returns False. The ability must return a structured error
    dict (not raise) so the ACT loop can surface it to the model.
    """
    from abilities.email import EmailAbility

    result = EmailAbility().execute(
        channel="test",
        params={"action": "search"},
        telemetry=None,
    )
    assert isinstance(result, dict), "execute() must return a dict"
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


def test_calendar_not_connected_returns_structured_error():
    """CalendarAbility.execute returns {status: error} when mail is not connected."""
    from abilities.calendar import CalendarAbility

    result = CalendarAbility().execute(
        channel="test",
        params={"action": "list_events"},
        telemetry=None,
    )
    assert isinstance(result, dict)
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


def test_contacts_not_connected_returns_structured_error():
    """ContactsAbility.execute returns {status: error} when mail is not connected."""
    from abilities.contacts import ContactsAbility

    result = ContactsAbility().execute(
        channel="test",
        params={"action": "list"},
        telemetry=None,
    )
    assert isinstance(result, dict)
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


# ---------------------------------------------------------------------------
# 3. Scheduler-driven sync is the authoritative sync path
# ---------------------------------------------------------------------------


def test_mail_capability_registers_sync_handler():
    """MailCapability._ensure_sync_registration registers a mail:sync handler.

    Verifies the method exists and the registration flag starts False.
    The actual registration requires a connected DB + scheduler, which the
    test env doesn't have — we just confirm the mechanism is wired.
    """
    from capabilities.mail_capability.capability import MailCapability

    cap = MailCapability()
    assert hasattr(cap, "_ensure_sync_registration")
    assert cap._sync_registered is False


def test_subconscious_worker_has_no_capability_sync_steps():
    """The subconscious worker must NOT have capability sync steps.

    Capability syncs are driven by the scheduler via MailCapability._do_monitor().
    The worker previously had broken _step_calendar_sync / _step_contacts_sync
    that called read-only abilities instead of actual server sync — those are
    now removed.
    """
    from services.subconscious_worker import SubconsciousWorker

    assert not hasattr(SubconsciousWorker, "_step_calendar_sync")
    assert not hasattr(SubconsciousWorker, "_step_contacts_sync")
    assert not hasattr(SubconsciousWorker, "_run_capability_syncs")
