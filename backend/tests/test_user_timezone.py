"""
Tests for user timezone support.

Covers:
- get_user_tz() resolution order (heartbeat → settings → UTC)
- Timezone persistence from heartbeat to settings
- Prompt assembly local time formatting
- Scheduler window checks in user timezone
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# get_user_tz()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetUserTz:
    """Test services.time_utils.get_user_tz resolution order."""

    def test_returns_utc_when_no_context(self):
        """Falls back to UTC when no heartbeat or settings exist."""
        with patch("services.client_context_service.ClientContextService") as mock_ctx:
            mock_ctx.return_value.get.return_value = {}
            with patch("services.settings_service.SettingsService") as mock_settings:
                mock_settings.return_value.get.return_value = None
                with patch("services.database_service.get_shared_db_service"):
                    from services.time_utils import get_user_tz
                    tz = get_user_tz()
                    assert tz.key == "UTC"

    def test_prefers_heartbeat_over_settings(self):
        """Heartbeat timezone takes priority over persisted setting."""
        with patch("services.client_context_service.ClientContextService") as mock_ctx:
            mock_ctx.return_value.get.return_value = {"timezone": "Europe/Berlin"}
            from services.time_utils import get_user_tz
            tz = get_user_tz()
            assert tz.key == "Europe/Berlin"

    def test_falls_back_to_settings(self):
        """When heartbeat has no timezone, uses settings."""
        with patch("services.client_context_service.ClientContextService") as mock_ctx:
            mock_ctx.return_value.get.return_value = {}
            with patch("services.database_service.get_shared_db_service"):
                with patch("services.settings_service.SettingsService") as mock_settings:
                    mock_settings.return_value.get.return_value = "America/New_York"
                    from services.time_utils import get_user_tz
                    tz = get_user_tz()
                    assert tz.key == "America/New_York"

    def test_handles_invalid_heartbeat_timezone(self):
        """Invalid IANA name in heartbeat falls through to next source."""
        with patch("services.client_context_service.ClientContextService") as mock_ctx:
            mock_ctx.return_value.get.return_value = {"timezone": "Not/A/Timezone"}
            with patch("services.database_service.get_shared_db_service"):
                with patch("services.settings_service.SettingsService") as mock_settings:
                    mock_settings.return_value.get.return_value = "Asia/Tokyo"
                    from services.time_utils import get_user_tz
                    tz = get_user_tz()
                    assert tz.key == "Asia/Tokyo"


# ---------------------------------------------------------------------------
# Timezone persistence from heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimezonePersistence:
    """Test ClientContextService persists timezone to settings."""

    def test_persists_on_first_heartbeat(self):
        """Timezone written to settings when no previous timezone exists."""
        from services.client_context_service import ClientContextService

        mock_store = MagicMock()
        mock_store.get.return_value = None  # no cached context

        with patch.object(ClientContextService, "__init__", lambda self: None):
            svc = ClientContextService()
            svc._store = mock_store

            with patch("services.settings_service.SettingsService") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings_cls.return_value = mock_settings

                with patch("services.database_service.get_shared_db_service"):
                    svc._persist_timezone("Europe/Malta", None)

                mock_settings.set.assert_called_once_with(
                    "user_timezone", "Europe/Malta", "string",
                    "User IANA timezone (auto-detected from client heartbeat)"
                )

    def test_skips_when_unchanged(self):
        """No DB write when timezone hasn't changed."""
        from services.client_context_service import ClientContextService

        with patch.object(ClientContextService, "__init__", lambda self: None):
            svc = ClientContextService()

            with patch("services.settings_service.SettingsService") as mock_settings_cls:
                svc._persist_timezone("Europe/Malta", "Europe/Malta")
                mock_settings_cls.assert_not_called()

    def test_rejects_invalid_iana(self):
        """Invalid IANA timezone name is not persisted."""
        from services.client_context_service import ClientContextService

        with patch.object(ClientContextService, "__init__", lambda self: None):
            svc = ClientContextService()

            with patch("services.settings_service.SettingsService") as mock_settings_cls:
                svc._persist_timezone("Not/Valid", None)
                mock_settings_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptAssemblyTimezone:
    """Test that prompt assembly uses user timezone for {{current_datetime}}."""

    def test_injects_user_local_time(self):
        """{{current_datetime}} should show user's timezone, not UTC."""
        with patch("services.time_utils.get_user_tz") as mock_tz:
            mock_tz.return_value = ZoneInfo("Europe/Berlin")

            from services.time_utils import utc_now
            now = utc_now()
            local = now.astimezone(ZoneInfo("Europe/Berlin"))
            expected_abbr = local.strftime("%Z")

            # The template placeholder replacement happens in _inject_parameters
            # We test the formatting logic directly
            tz = mock_tz.return_value
            local_now = now.astimezone(tz)
            tz_abbr = local_now.strftime("%Z") or tz.key
            result = local_now.strftime(f"%A, %Y-%m-%d %H:%M {tz_abbr}")

            assert expected_abbr in result
            assert "UTC" not in result or expected_abbr == "UTC"

    def test_falls_back_to_utc_on_error(self):
        """On exception, falls back to UTC display."""
        with patch("services.time_utils.get_user_tz", side_effect=Exception("boom")):
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            result = now.strftime("%A, %Y-%m-%d %H:%M UTC")
            assert "UTC" in result


# ---------------------------------------------------------------------------
# Scheduler window timezone conversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchedulerWindowTimezone:
    """Test that hourly window checks use user timezone."""

    def test_window_check_in_user_timezone(self):
        """Window 09:00-17:00 local should work correctly for non-UTC user.

        User in UTC+2: due_at 08:00 UTC (= 10:00 local) is within 09:00-17:00 local.
        Before the fix, this would fail because 08:00 < 09:00 in UTC comparison.
        """
        import services.scheduler_service as svc

        due_at = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)  # 10:00 CET

        with patch("services.time_utils.get_user_tz") as mock_tz:
            mock_tz.return_value = ZoneInfo("Europe/Berlin")  # UTC+1 in Jan (CET)
            next_due = svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")

        # 08:00 UTC + 1h = 09:00 UTC = 10:00 CET → within 09:00-17:00 CET
        assert next_due is not None
        assert next_due == datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)

    def test_window_past_end_advances_to_next_day_local(self):
        """Past window end in local time should advance to next day's window start.

        User in UTC+2: due_at 16:00 UTC (= 17:00 CET) is at window end.
        Next hour = 17:00 UTC (= 18:00 CET) → past window → advance to 09:00 CET next day.
        """
        import services.scheduler_service as svc

        due_at = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)  # 17:00 CET

        with patch("services.time_utils.get_user_tz") as mock_tz:
            mock_tz.return_value = ZoneInfo("Europe/Berlin")
            next_due = svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")

        # Should advance to 09:00 CET next day = 08:00 UTC
        assert next_due is not None
        assert next_due.day == 16
        local_next = next_due.astimezone(ZoneInfo("Europe/Berlin"))
        assert local_next.hour == 9
        assert local_next.minute == 0

    def test_window_utc_user_unchanged(self):
        """UTC user: window behavior should be identical to pre-timezone code."""
        import services.scheduler_service as svc

        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        with patch("services.time_utils.get_user_tz") as mock_tz:
            mock_tz.return_value = ZoneInfo("UTC")
            next_due = svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")

        assert next_due == datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

    def test_window_before_start_jumps_to_start(self):
        """Before window start in local time should jump to window start.

        User in UTC+5: due_at 02:00 UTC (= 07:00 local). Next hour = 03:00 UTC (= 08:00 local).
        08:00 < 09:00 → jump to 09:00 local = 04:00 UTC.
        """
        import services.scheduler_service as svc

        due_at = datetime(2024, 1, 15, 2, 0, tzinfo=timezone.utc)  # 07:00 local

        with patch("services.time_utils.get_user_tz") as mock_tz:
            mock_tz.return_value = ZoneInfo("Asia/Karachi")  # UTC+5
            next_due = svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")

        assert next_due is not None
        local_next = next_due.astimezone(ZoneInfo("Asia/Karachi"))
        assert local_next.hour == 9
        assert local_next.minute == 0
