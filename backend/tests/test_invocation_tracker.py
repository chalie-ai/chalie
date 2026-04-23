"""Unit tests for invocation_tracker.track() — channel forwarding."""
import pytest

from services.invocation_tracker import track
from services.database_service import get_shared_db_service

pytestmark = pytest.mark.unit


def _fetch_channel(exchange_id: str) -> str | None:
    """Re-open a fresh connection to query the patched shared db."""
    db = get_shared_db_service()
    rows = db.fetch_all(
        "SELECT channel FROM tool_performance_metrics WHERE exchange_id = ?",
        (exchange_id,)
    )
    return rows[0]["channel"] if rows else None


class TestTrackChannelForwarding:
    def test_channel_forwarded_to_metrics(self, db):
        track(name="weather", success=True, latency_ms=50.0, exchange_id="exch_t1", channel="dmn")
        assert _fetch_channel("exch_t1") == "dmn"

    def test_topic_alias_resolves_to_channel(self, db):
        track(name="memory", success=True, latency_ms=30.0, exchange_id="exch_t2", topic="goal_pursuit")
        assert _fetch_channel("exch_t2") == "goal_pursuit"

    def test_channel_overrides_topic_when_both_set(self, db):
        track(name="find_tools", success=True, latency_ms=20.0, exchange_id="exch_t3",
              topic="topic_value", channel="user")
        assert _fetch_channel("exch_t3") == "user"

    def test_no_channel_stores_empty_string(self, db):
        track(name="search", success=False, latency_ms=10.0, exchange_id="exch_t4")
        assert _fetch_channel("exch_t4") == ""
