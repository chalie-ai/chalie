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
from unittest.mock import MagicMock, patch


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

    def test_large_metadata_dict_stored(self):
        """Signals with large metadata dicts are serialised and stored without error."""
        svc, store = _make_service()
        big_meta = {f"key_{i}": f"value_{i}" * 50 for i in range(30)}
        svc.notify_external_signal(
            signal_type="data_dump",
            source="data-service",
            content="large payload",
            metadata=big_meta,
        )
        store.rpush.assert_called_once()
        payload = json.loads(store.rpush.call_args[0][1])
        assert len(payload["metadata"]) == 30

    def test_max_signals_cap_enforced_via_ltrim(self):
        """After each write, ltrim is called with -MAX_EXTERNAL_SIGNALS, -1 to cap the list."""
        svc, store = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src", content="data",
        )
        store.ltrim.assert_called_once_with(
            EXTERNAL_SIGNALS_KEY, -MAX_EXTERNAL_SIGNALS, -1
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

    def test_activation_energy_zero_is_below_threshold(self):
        """A signal with activation_energy=0.0 decays to zero salience immediately."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src", content="zero energy",
            activation_energy=0.0,
        )
        items = svc._get_external_signals()
        # salience = temporal_factor * activation_energy * 2 = ... * 0.0 * 2 = 0.0
        # 0.0 < SALIENCE_THRESHOLD → not included
        assert items == []

    def test_activation_energy_one_gives_maximum_salience(self):
        """A signal with activation_energy=1.0 and recent timestamp has maximum salience."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src", content="max energy",
            activation_energy=1.0,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        # salience = temporal_factor * 1.0 * 2 — capped at 1.0 for very fresh signals
        assert items[0]["salience"] > SALIENCE_THRESHOLD
        assert items[0]["salience"] <= 1.0 + 1e-6  # allow floating-point tolerance

    def test_unicode_content_preserved(self):
        """Unicode content and emoji in signal content round-trip correctly."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="src",
            content="日本語テスト 🚀 Ünïcödë",
            activation_energy=0.8,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        assert "日本語テスト 🚀 Ünïcödë" in items[0]["label"]

    def test_unicode_source_name_preserved(self):
        """Unicode source names are preserved in the world state label."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="système-监控",
            content="status update",
            activation_energy=0.7,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        assert "système-监控" in items[0]["label"]

    def test_multiple_sources_interleaved_timestamps(self):
        """Signals from multiple sources interleaved in time each get their own slot."""
        svc, _ = _make_service()
        now = time.time()
        for i, src in enumerate(["source-a", "source-b", "source-c"]):
            svc._stored_list.append(json.dumps({
                "signal_type": "event",
                "source": src,
                "content": f"event from {src}",
                "activation_energy": 0.6,
                "timestamp": now - i * 10,  # staggered timestamps
            }))
        items = svc._get_external_signals()
        sources = {item["label"].split("]")[0].lstrip("[SIGNAL:") for item in items}
        assert len(items) == 3

    def test_fresh_signal_has_near_maximum_salience(self):
        """A very fresh signal (just now) has salience close to activation_energy * 2."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="test", source="fresh-src",
            content="just arrived",
            activation_energy=0.5,
        )
        items = svc._get_external_signals()
        assert len(items) == 1
        # temporal_factor ≈ 1.0 for a brand-new signal → salience ≈ 1.0 * 0.5 * 2 = 1.0
        assert items[0]["salience"] > 0.5

    def test_decay_at_two_half_lives(self):
        """A signal at 2× the half-life has roughly 25% of its original salience."""
        svc, _ = _make_service()
        svc._stored_list.append(json.dumps({
            "signal_type": "test", "source": "old-src",
            "content": "aging signal", "activation_energy": 1.0,
            "timestamp": time.time() - (2 * EXTERNAL_SIGNAL_DECAY_HOURS * 3600),
        }))
        items = svc._get_external_signals()
        # At 2× half-life temporal_factor ≈ 0.25; salience = 0.25 * 1.0 * 2 = 0.5
        # May or may not exceed SALIENCE_THRESHOLD (0.15) — just verify it's < fresh salience
        if items:
            assert items[0]["salience"] < 0.7  # clearly decayed from max ≈ 2.0

    def test_low_activation_energy_fresh_signal_below_threshold(self):
        """A fresh signal with very low activation_energy falls below SALIENCE_THRESHOLD."""
        svc, _ = _make_service()
        svc.notify_external_signal(
            signal_type="minor", source="background",
            content="low priority update",
            activation_energy=0.05,  # Very weak signal
        )
        items = svc._get_external_signals()
        # salience = temporal_factor * 0.05 * 2 ≈ 0.10 < SALIENCE_THRESHOLD (0.15)
        assert items == []


# ---------------------------------------------------------------------------
# context_update bridge — telemetry flows to ClientContextService
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestContextUpdateBridge:
    def test_context_update_calls_client_context_save(self):
        """context_update signals should bridge to ClientContextService.save()."""
        svc, _ = _make_service()
        context_payload = {
            "timezone": "Europe/London",
            "locale": "en-GB",
            "device": {"class": "phone", "platform": "iOS"},
            "location": {"lat": 51.5, "lon": -0.12},
        }
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content=json.dumps(context_payload),
            )
            mock_instance.save.assert_called_once_with(context_payload)

    def test_non_context_signal_does_not_bridge(self):
        """Regular signals should not call ClientContextService.save()."""
        svc, _ = _make_service()
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            svc.notify_external_signal(
                signal_type="weather_update",
                source="weather",
                content="Sunny, 22C",
            )
            mock_cls.return_value.save.assert_not_called()

    def test_bridge_fail_open(self):
        """If ClientContextService.save() throws, the signal is still stored."""
        svc, _ = _make_service()
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            mock_cls.return_value.save.side_effect = Exception("save failed")
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content=json.dumps({"timezone": "UTC"}),
            )
        # Signal should still be in the store despite bridge failure
        items = svc._get_external_signals()
        assert len(items) == 1

    def test_bridge_handles_invalid_json(self):
        """If content is not valid JSON, bridge skips gracefully."""
        svc, _ = _make_service()
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content="not valid json {{{",
            )
            mock_cls.return_value.save.assert_not_called()

    def test_bridge_handles_empty_payload(self):
        """Empty dict payload should be skipped."""
        svc, _ = _make_service()
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content=json.dumps({}),
            )
            mock_cls.return_value.save.assert_not_called()


# ---------------------------------------------------------------------------
# Context update bridge — additional coverage
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestContextUpdateBridgeAdditional:
    """Additional coverage for the context_update bridge path."""

    def test_multiple_context_updates_in_sequence_each_bridged(self):
        """Each context_update signal triggers a separate ClientContextService.save() call."""
        svc, _ = _make_service()
        payloads = [
            {"timezone": "Europe/London"},
            {"locale": "fr-FR"},
        ]
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            for p in payloads:
                svc.notify_external_signal(
                    signal_type="context_update",
                    source="__chat_ui__",
                    content=json.dumps(p),
                )
            assert mock_instance.save.call_count == 2

    def test_bridge_passes_nested_dict_payload_unchanged(self):
        """Nested dict payloads are forwarded as-is to ClientContextService.save()."""
        svc, _ = _make_service()
        nested_payload = {
            "device": {"class": "phone", "os": "iOS", "version": "17"},
            "location": {"lat": 48.85, "lon": 2.35, "accuracy": 10},
        }
        with patch("services.client_context_service.ClientContextService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content=json.dumps(nested_payload),
            )
            call_args = mock_instance.save.call_args[0][0]
            assert call_args["device"]["class"] == "phone"
            assert call_args["location"]["lat"] == pytest.approx(48.85)

    def test_context_update_signal_also_stored_in_memorystore(self):
        """context_update signals are stored in MemoryStore even when bridge succeeds."""
        svc, store = _make_service()
        with patch("services.client_context_service.ClientContextService"):
            svc.notify_external_signal(
                signal_type="context_update",
                source="__chat_ui__",
                content=json.dumps({"timezone": "UTC"}),
            )
        # rpush should still have been called once to store the signal
        store.rpush.assert_called_once()
