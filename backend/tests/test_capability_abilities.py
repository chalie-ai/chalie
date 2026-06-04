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
    from abilities._ability import Ability

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

    result = EmailAbility().run({"action": "search"})
    assert isinstance(result, dict), "execute() must return a dict"
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


def test_calendar_read_returns_structured_error_without_db():
    """CalendarAbility.execute read ops return structured error when DB is unavailable.

    Read operations (list_events, get_event) query scheduled_items via the
    schedule ability's query_items — they do NOT require the mail capability
    to be connected. In test env the DB is absent, so we get a DB error
    wrapped in a structured dict.
    """
    from abilities.calendar import CalendarAbility

    result = CalendarAbility().run({"action": "list_events"})
    assert isinstance(result, dict)
    assert "text" in result
    payload = _extract_json(result)
    assert payload.get("status") == "error" or "error" in payload


def test_calendar_write_not_connected_returns_structured_error():
    """CalendarAbility.execute write ops return {status: error} when mail not connected."""
    from abilities.calendar import CalendarAbility

    result = CalendarAbility().run({"action": "update_event", "uid": "test-123", "summary": "New title"})
    assert isinstance(result, dict)
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


def test_contacts_not_connected_returns_structured_error():
    """ContactsAbility.execute returns {status: error} when mail is not connected."""
    from abilities.contacts import ContactsAbility

    result = ContactsAbility().run({"action": "list"})
    assert isinstance(result, dict)
    assert "text" in result
    payload = _extract_json(result)
    assert payload["status"] == "error"
    assert "not connected" in payload["error"].lower()


# ---------------------------------------------------------------------------
# 3. Scheduler-driven sync is the authoritative sync path
# ---------------------------------------------------------------------------


def test_subconscious_worker_owns_capability_sync():
    """The subconscious worker drives capability syncs via _step_capability_sync.

    The worker calls each connected capability's monitor() method on every
    tick. MailCapability._do_monitor() handles per-protocol cadence internally.
    The scheduler is NOT involved in triggering syncs.
    """
    from services.subconscious_worker import SubconsciousWorker

    assert hasattr(SubconsciousWorker, "_step_capability_sync")
    assert not hasattr(SubconsciousWorker, "_step_calendar_sync")
    assert not hasattr(SubconsciousWorker, "_step_contacts_sync")


def test_mail_capability_has_no_scheduler_sync_registration():
    """MailCapability must NOT register a scheduler handler for syncs.

    Syncs are driven by the subconscious worker, not the scheduler.
    The scheduler only stores calendar event data.
    """
    from capabilities.mail_capability.capability import MailCapability

    assert not hasattr(MailCapability, "_ensure_sync_registration") or not callable(getattr(MailCapability, "_ensure_sync_registration", None))
    cap = MailCapability()
    assert not hasattr(cap, "_sync_registered")
