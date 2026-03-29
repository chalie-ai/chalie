"""Tests for life_snapshot innate skill."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

_UTC = "services.innate_skills.life_snapshot_skill.utc_now"
_CAP = "capabilities.load_capabilities"
_NOW = datetime(2026, 3, 29, 10, 0, 0, tzinfo=timezone.utc)


def _invoke(params=None):
    with patch(_UTC, return_value=_NOW):
        from services.innate_skills.life_snapshot_skill import handle_life_snapshot
        return handle_life_snapshot("t", params or {})


def _seed(db, summary, dtstart):
    db.execute(
        "INSERT INTO scheduled_items "
        "(id, message, due_at, status, source, item_type, metadata) "
        "VALUES (?, ?, ?, 'pending', 'caldav', 'event', ?)",
        (str(uuid.uuid4()), summary, dtstart.isoformat(),
         json.dumps({"dtstart": dtstart.isoformat(), "summary": summary})),
    )
    db.commit()


@pytest.mark.unit
class TestLifeSnapshot:

    def test_calendar_events_listed(self, db):
        _seed(db, "Standup", _NOW.replace(hour=9))
        r = _invoke()
        assert "[LIFE SNAPSHOT]" in r
        assert "Standup" in r

    def test_email_section(self, db):
        c = MagicMock(is_connected=MagicMock(return_value=True))
        c.get_tools.return_value = [{"name": "imap_search_email",
                                     "handler": lambda t, p: {"emails": [
                                         {"from_name": "A", "subject": "B"}]}}]
        with patch(_CAP, return_value={"imap": c}):
            assert "Unanswered" in _invoke()

    def test_tomorrow_opt_in(self, db):
        _seed(db, "Board", _NOW + timedelta(days=1))
        assert "Board" in _invoke({"include_tomorrow": True})
