"""
Unit tests for WorldStateService external signal handling.

Tests the notify_external_signal() and _get_external_signals() methods
that route external interface signals directly to world state (zero LLM).
Signals are grouped by source — each source gets one world state slot
showing only its most salient signal.
"""

import json
import math
import time

import pytest
from unittest.mock import MagicMock


from services.world_state_service import (
    WorldStateService,
    EXTERNAL_SIGNALS_KEY,
    MAX_EXTERNAL_SIGNALS,
    EXTERNAL_SIGNAL_DECAY_HOURS,
    SALIENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(store_data=None):
    """Create a WorldStateService with a mock MemoryStore."""
    store = MagicMock()
    stored_list = []

    def rpush(key, value):
        if key == EXTERNAL_SIGNALS_KEY:
            stored_list.append(value)

    def ltrim(key, start, end):
        nonlocal stored_list
        if key == EXTERNAL_SIGNALS_KEY and len(stored_list) > MAX_EXTERNAL_SIGNALS:
            stored_list = stored_list[-MAX_EXTERNAL_SIGNALS:]

    def lrange(key, start, end):
        if key == EXTERNAL_SIGNALS_KEY:
            return list(stored_list)
        return []

    store.rpush.side_effect = rpush
    store.ltrim.side_effect = ltrim
    store.lrange.side_effect = lrange

    svc = WorldStateService()
    svc._store = store
    svc._stored_list = stored_list
    return svc, store


# ---------------------------------------------------------------------------
# notify_external_signal
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNotifyExternalSignal:
    def test_writes_to_memorystore(self):
        svc, store = _make_service()
        svc.notify_external_signal(
            signal_type="stock_price",
            source="stock-exchange",
            content="AAPL at $185.50",
        )
        store.rpush.assert_called_once()
        assert store.rpush.call_args[0][0] == EXTERNAL_SIGNALS_KEY
        payload = json.loads(store.rpush.call_args[0][1])
        assert payload["signal_type"] == "stock_price"
        assert payload["source"] == "stock-exchange"
        assert payload["content"] == "AAPL at $185.50"
        assert payload["activation_energy"] == 0.5
        assert "timestamp" in payload

    def test_trims_list_after_write(self):
        svc, store = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src", content="data",
        )
        store.ltrim.assert_called_once_with(
            EXTERNAL_SIGNALS_KEY, -MAX_EXTERNAL_SIGNALS, -1
        )

    def test_preserves_all_fields(self):
        svc, store = _make_service()
        svc.notify_external_signal(
            signal_type="weather",
            source="weather-service",
            content="Rain expected",
            topic="weather",
            activation_energy=0.7,
            metadata={"city": "London"},
        )
        payload = json.loads(store.rpush.call_args[0][1])
        assert payload["topic"] == "weather"
        assert payload["activation_energy"] == 0.7
        assert payload["metadata"] == {"city": "London"}

    def test_defaults_optional_fields(self):
        svc, store = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src", content="data",
        )
        payload = json.loads(store.rpush.call_args[0][1])
        assert payload["topic"] is None
        assert payload["activation_energy"] == 0.5
        assert payload["metadata"] is None

    def test_fail_open_on_store_error(self):
        svc, store = _make_service()
        store.rpush.side_effect = Exception("connection lost")
        svc.notify_external_signal(
            signal_type="test", source="src", content="data",
        )


# ---------------------------------------------------------------------------
# _get_external_signals — grouped by source, most salient wins
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetExternalSignals:
    def test_empty_store_returns_empty(self):
        svc, _ = _make_service()
        assert svc._get_external_signals() == []

    def test_single_signal_shows_content(self):
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="alert",
            source="hospital",
            content="Emergency in wing B",
            activation_energy=0.8,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        assert items[0]["label"] == "[SIGNAL:hospital] Emergency in wing B"
        assert items[0]["salience"] > SALIENCE_THRESHOLD

    def test_multiple_signals_same_source_one_slot(self):
        """Multiple signals from one source produce exactly one item."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="AAPL $185", activation_energy=0.5,
        )
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="MSFT $420", activation_energy=0.7,
        )
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="GOOGL $178", activation_energy=0.4,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        # Most salient signal wins (MSFT has highest energy)
        assert items[0]["label"] == "[SIGNAL:exchange] MSFT $420"

    def test_different_sources_separate_items(self):
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="alert", source="hospital",
            content="Emergency in wing B", activation_energy=0.8,
        )
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="AAPL $185", activation_energy=0.5,
        )
        items = svc._get_external_signals()
        assert len(items) == 2
        labels = {item["label"] for item in items}
        assert "[SIGNAL:hospital] Emergency in wing B" in labels
        assert "[SIGNAL:exchange] AAPL $185" in labels

    def test_most_salient_signal_wins_per_source(self):
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="Minor update", activation_energy=0.2,
        )
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="CRITICAL: market crash", activation_energy=0.9,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        assert items[0]["label"] == "[SIGNAL:exchange] CRITICAL: market crash"

    def test_source_salience_is_best_signal_salience(self):
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="Low", activation_energy=0.3,
        )
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="High", activation_energy=0.9,
        )
        items = svc._get_external_signals()
        # Salience from the 0.9 energy signal, not averaged
        assert items[0]["salience"] == pytest.approx(
            min(1.0, 1.0 * 0.9 * 2), abs=0.05
        )

    def test_noisy_source_cannot_dominate(self):
        """50 signals from one source still produce one world state slot."""
        svc, _ = _make_service()
        for i in range(50):
            svc.notify_external_signal(
                signal_type="price", source="exchange",
                content=f"Tick {i}", activation_energy=0.5,
            )
        svc.notify_external_signal(
            signal_type="alert", source="hospital",
            content="Wing B closed", activation_energy=0.6,
        )
        items = svc._get_external_signals()
        assert len(items) == 2  # One per source, not 51

    def test_old_signal_decays_below_threshold(self):
        svc, _ = _make_service()
        svc._stored_list.append(json.dumps({
            "signal_type": "price", "source": "exchange",
            "content": "Old data", "activation_energy": 0.3,
            "timestamp": time.time() - (24 * 3600),
        }))
        assert svc._get_external_signals() == []

    def test_sorted_by_salience_descending(self):
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="a", source="low-src", content="Low",
            activation_energy=0.3,
        )
        svc.notify_external_signal(
            signal_type="b", source="high-src", content="High",
            activation_energy=0.9,
        )
        svc.notify_external_signal(
            signal_type="c", source="mid-src", content="Mid",
            activation_energy=0.5,
        )
        items = svc._get_external_signals()
        saliences = [item["salience"] for item in items]
        assert saliences == sorted(saliences, reverse=True)

    def test_fail_open_on_store_error(self):
        svc, store = _make_service()
        store.lrange.side_effect = Exception("connection lost")
        assert svc._get_external_signals() == []

    def test_malformed_json_skipped(self):
        svc, _ = _make_service()
        svc._stored_list.append("not valid json {{{")
        svc.notify_external_signal(
            signal_type="valid", source="src", content="Fine",
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        assert "Fine" in items[0]["label"]

    def test_temporal_decay_math(self):
        svc, _ = _make_service()
        svc._stored_list.append(json.dumps({
            "signal_type": "test", "source": "src",
            "content": "Half-life test", "activation_energy": 1.0,
            "timestamp": time.time() - (EXTERNAL_SIGNAL_DECAY_HOURS * 3600),
        }))
        items = svc._get_external_signals()
        assert len(items) == 1
        assert items[0]["salience"] == pytest.approx(1.0, abs=0.05)

    def test_stale_signals_excluded_from_source(self):
        """Stale signals below threshold don't appear — only fresh ones do."""
        svc, _ = _make_service()
        svc._stored_list.append(json.dumps({
            "signal_type": "price", "source": "exchange",
            "content": "Ancient data", "activation_energy": 0.2,
            "timestamp": time.time() - (48 * 3600),
        }))
        svc.notify_external_signal(
            signal_type="price", source="exchange",
            content="Fresh data", activation_energy=0.8,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        # Only the fresh signal — stale one below threshold
        assert items[0]["label"] == "[SIGNAL:exchange] Fresh data"
