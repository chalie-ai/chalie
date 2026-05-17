"""
Regression tests for ClientContextService._persist_timezone().

Verifies that save() writes user_timezone to the settings table when the
timezone changes, skips the write on a duplicate call, and silently ignores
invalid IANA names — restoring the behaviour removed in commit 8088f09a.

Uses the real SQLite DB via the `db` fixture (no mocks for the integration
aspect) and a real MemoryStore via the `store` fixture (needed because
ClientContextService.__init__ opens a MemoryClientService connection).
"""

import json
import pytest


def _read_timezone_setting(db):
    """Return the stored user_timezone value, or None if absent."""
    row = db.execute(
        "SELECT value FROM settings WHERE key = 'user_timezone'"
    ).fetchone()
    return row[0] if row else None


def _read_timezone_updated_at(db):
    """Return the updated_at timestamp for user_timezone, or None."""
    row = db.execute(
        "SELECT updated_at FROM settings WHERE key = 'user_timezone'"
    ).fetchone()
    return row[0] if row else None


def _seed_telemetry(db, **kwargs):
    """Seed flat telemetry rows for the context get() call inside save()."""
    db.execute("DELETE FROM telemetry")
    for key, value in kwargs.items():
        db.execute(
            "INSERT INTO telemetry (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    db.commit()


@pytest.mark.unit
class TestPersistTimezone:
    """ClientContextService.save() writes timezone to settings table."""

    def test_save_writes_timezone_to_settings(self, db, store):
        """save() with a valid IANA timezone persists user_timezone to settings."""
        _seed_telemetry(db)  # empty — no prior context
        from services.client_context_service import ClientContextService
        svc = ClientContextService()
        svc.save({"timezone": "Asia/Tokyo"})
        assert _read_timezone_setting(db) == "Asia/Tokyo"

    def test_save_updates_timezone_when_changed(self, db, store):
        """A second save() with a different timezone updates the settings row."""
        _seed_telemetry(db)
        from services.client_context_service import ClientContextService
        svc = ClientContextService()
        svc.save({"timezone": "Asia/Tokyo"})
        assert _read_timezone_setting(db) == "Asia/Tokyo"

        # Simulate the telemetry table now reflecting the first save so
        # cached_ctx.get("timezone") returns "Asia/Tokyo" on the next call.
        _seed_telemetry(db, timezone="Asia/Tokyo")

        svc.save({"timezone": "Europe/Malta"})
        assert _read_timezone_setting(db) == "Europe/Malta"

    def test_invalid_timezone_not_written(self, db, store):
        """An invalid IANA name must not write a row and must not raise."""
        _seed_telemetry(db)
        from services.client_context_service import ClientContextService
        svc = ClientContextService()
        svc.save({"timezone": "Not/A/Zone"})
        assert _read_timezone_setting(db) is None

    def test_same_timezone_does_not_rewrite(self, db, store):
        """Calling save() twice with the same timezone skips the second write."""
        _seed_telemetry(db)
        from services.client_context_service import ClientContextService
        svc = ClientContextService()
        svc.save({"timezone": "Asia/Tokyo"})
        first_updated_at = _read_timezone_updated_at(db)

        # Reflect the saved timezone into telemetry so cached_ctx matches
        _seed_telemetry(db, timezone="Asia/Tokyo")

        svc.save({"timezone": "Asia/Tokyo"})
        second_updated_at = _read_timezone_updated_at(db)

        # updated_at must not have changed — no second write occurred
        assert first_updated_at == second_updated_at
