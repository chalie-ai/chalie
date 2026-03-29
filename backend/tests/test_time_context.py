"""Tests for capabilities.time_context — canonical user-timezone helpers."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch


@pytest.mark.unit
class TestGetUserTz:
    """get_user_tz returns ZoneInfo when configured, UTC when not."""

    def test_returns_utc_when_no_context(self):
        from capabilities.time_context import get_user_tz

        with patch(
            "services.client_context_service.ClientContextService"
        ) as mock_cls:
            mock_cls.return_value.get.return_value = {}
            result = get_user_tz()
        assert result is timezone.utc

    def test_returns_zoneinfo_when_configured(self):
        from zoneinfo import ZoneInfo

        from capabilities.time_context import get_user_tz

        with patch(
            "services.client_context_service.ClientContextService"
        ) as mock_cls:
            mock_cls.return_value.get.return_value = {
                "timezone": "America/New_York"
            }
            result = get_user_tz()
        assert result == ZoneInfo("America/New_York")

    def test_returns_utc_on_exception(self):
        from capabilities.time_context import get_user_tz

        with patch(
            "services.client_context_service.ClientContextService",
            side_effect=RuntimeError("no service"),
        ):
            result = get_user_tz()
        assert result is timezone.utc


@pytest.mark.unit
class TestUserLocalNow:
    """user_local_now converts to local timezone correctly."""

    def test_converts_utc_to_local(self):
        from zoneinfo import ZoneInfo

        from capabilities.time_context import user_local_now

        utc_time = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)
        with patch(
            "services.client_context_service.ClientContextService"
        ) as mock_cls:
            mock_cls.return_value.get.return_value = {
                "timezone": "Asia/Tokyo"
            }
            local = user_local_now(utc_time)
        assert local.hour == 21  # UTC+9
        assert local.tzinfo == ZoneInfo("Asia/Tokyo")

    def test_falls_back_to_utc(self):
        from capabilities.time_context import user_local_now

        utc_time = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)
        with patch(
            "services.client_context_service.ClientContextService"
        ) as mock_cls:
            mock_cls.return_value.get.return_value = {}
            local = user_local_now(utc_time)
        assert local.hour == 12
        assert local.tzinfo is timezone.utc
