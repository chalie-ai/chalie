"""Feature tests for the three capability-backed Ability wrappers and the
subconscious worker sync integration.

Real-world assertions only — no mocks of in-process production code.

What each test asserts:
  1. All three abilities are discovered by AbilityRegistry with the correct
     NAME, SUMMARY, and EXAMPLES count (6–8 enforced by _base.py metaclass).
  2. When the mail capability is not connected (no credentials in test env),
     each ability returns a structured error dict — not an exception.
  3. The subconscious sync cadence: calendar on every even tick, contacts on
     every 12th tick — verified by inspecting the keys returned by
     _run_capability_syncs() across N calls.
  4. The subconscious _step_calendar_sync / _step_contacts_sync survive
     gracefully when the registry has a capability that is not connected —
     returning "ok" (execute was called, returned structured error, no raise).

Context: MailCapability is discovered at import time but is_connected()
returns False in test environments (no credentials configured). This means
every execute() call on these abilities exercises the not-connected error
path without requiring real IMAP/CalDAV/CardDAV credentials.
"""

import json
import threading

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


def _make_worker_bare() -> object:
    """Construct a SubconsciousWorker with all state initialised but without
    touching the database or MemoryStore (skipping the hydrate call).

    Uses __new__ to bypass __init__'s _load_last_fired_from_storage() call so
    the test does not need a real DB. Only the fields touched by
    _run_capability_syncs() are set.
    """
    from services.subconscious_worker import SubconsciousWorker

    worker = SubconsciousWorker.__new__(SubconsciousWorker)
    worker._sync_tick_count = 0
    worker._lock = threading.Lock()
    worker._cached_last_fired = None
    worker._decay_engine = None
    return worker


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
# 3. Subconscious sync cadence
# ---------------------------------------------------------------------------


def test_sync_cadence_calendar_on_even_ticks_contacts_on_12th():
    """_run_capability_syncs fires calendar every 2nd tick and contacts every 12th.

    Drives 12 calls and checks that:
    - "calendar" key appears in the result for every even tick.
    - "contacts" key appears only on tick 12 (the first tick divisible by 12).
    - Odd ticks produce an empty result dict.
    """
    worker = _make_worker_bare()

    calendar_ticks = []
    contacts_ticks = []

    for call_number in range(1, 13):
        results = worker._run_capability_syncs()
        if "calendar" in results:
            calendar_ticks.append(call_number)
        if "contacts" in results:
            contacts_ticks.append(call_number)

    # Calendar should fire on ticks 2, 4, 6, 8, 10, 12
    assert calendar_ticks == [2, 4, 6, 8, 10, 12], (
        f"Calendar did not fire on every even tick. Fired on: {calendar_ticks}"
    )
    # Contacts fires only on tick 12 within this window
    assert contacts_ticks == [12], (
        f"Contacts did not fire only on tick 12. Fired on: {contacts_ticks}"
    )


def test_sync_odd_tick_produces_empty_result():
    """On odd ticks, _run_capability_syncs returns an empty dict (no syncs run)."""
    worker = _make_worker_bare()

    # First call increments to 1 — odd
    result = worker._run_capability_syncs()
    assert result == {}, (
        f"Odd tick must produce empty sync result, got: {result}"
    )


# ---------------------------------------------------------------------------
# 4. Subconscious step survives when capability is not connected
# ---------------------------------------------------------------------------


def test_step_calendar_sync_does_not_raise_when_not_connected():
    """_step_calendar_sync completes without raising when calendar is registered
    but the underlying mail account is not connected.

    The step calls calendar.execute() which returns a structured error dict.
    The step must return "ok" — it is fire-and-forget from the worker's
    perspective (the error is logged but does not abort the tick).
    """
    worker = _make_worker_bare()
    # Direct call — does not go through _safe_step so exceptions would propagate.
    result = worker._step_calendar_sync()
    assert result == "ok", (
        f"_step_calendar_sync must return 'ok' even when capability is not "
        f"connected. Got: {result!r}"
    )


def test_step_contacts_sync_does_not_raise_when_not_connected():
    """_step_contacts_sync completes without raising when contacts is registered
    but the underlying mail account is not connected.
    """
    worker = _make_worker_bare()
    result = worker._step_contacts_sync()
    assert result == "ok", (
        f"_step_contacts_sync must return 'ok' even when capability is not "
        f"connected. Got: {result!r}"
    )
