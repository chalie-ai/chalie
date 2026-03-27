"""Unit tests for EngagementSignalService — engagement score surfacing."""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestEngagementSignalServiceDefaults:
    """Tests for score values that fall in the unremarkable band (no items returned)."""

    def test_returns_empty_for_default_score(self):
        """Score of 0.7 (inside unremarkable band) should return an empty list."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.return_value = "0.7"

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert result == []

    def test_returns_empty_for_mid_range_score(self):
        """Score of 0.6 (inside unremarkable band) should return an empty list."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.return_value = "0.6"

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert result == []


class TestEngagementSignalServiceLowScore:
    """Tests for score values below the low threshold (< 0.35)."""

    def test_returns_item_for_low_score(self):
        """Score of 0.2 (below 0.35 threshold) should return a single advisory item."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.return_value = "0.2"

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert isinstance(result, list)
        assert len(result) == 1

        item = result[0]
        assert "type" in item
        assert "label" in item
        assert "salience" in item
        assert item["type"] == "engagement"
        assert isinstance(item["salience"], float)


class TestEngagementSignalServiceHighScore:
    """Tests for score values above the high threshold (> 0.88)."""

    def test_returns_item_for_high_score(self):
        """Score of 0.95 (above 0.88 threshold) should return a non-empty list."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.return_value = "0.95"

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert isinstance(result, list)
        assert len(result) > 0

        item = result[0]
        assert item["type"] == "engagement"
        assert isinstance(item["salience"], float)


class TestEngagementSignalServiceErrorHandling:
    """Tests for graceful degradation when the store is unavailable or returns nothing."""

    def test_returns_empty_on_store_exception(self):
        """A store.get exception should be caught and return an empty list (fail-open)."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.side_effect = Exception("Connection refused")

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert result == []

    def test_returns_empty_when_no_score(self):
        """store.get returning None (key absent) should return an empty list."""
        from services.engagement_signal_service import EngagementSignalService

        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch(
            "services.memory_client.MemoryClientService.create_connection",
            return_value=mock_store,
        ):
            svc = EngagementSignalService()
            result = svc.get_engagement_items()

        assert result == []


class TestEngagementSignalServiceImportability:
    """Smoke tests that the module and class are importable without side effects."""

    def test_service_importable(self):
        """EngagementSignalService should be importable with no infrastructure required."""
        from services.engagement_signal_service import EngagementSignalService  # noqa: F401

        svc = EngagementSignalService()
        assert svc is not None
        # Store must be lazily initialised — not yet connected at construction time
        assert svc._store is None
