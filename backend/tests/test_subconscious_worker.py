"""
Unit tests for SubconsciousWorker — gates, step exception isolation, persistence.

Per Dylan's directive: lean test surface for the worker shell. Sub-services
(SuperEpisodeEncoderProcessor, DecayEngineService, PatternExtractor,
UserSummaryProcessor) own their own behavioural tests; this file only verifies
the worker's gate logic, exception isolation, persistence, and backpressure.

Per feedback_test_philosophy.md: hot-path only, no mock theater.  Step methods
are monkeypatched to exercise the worker's orchestration without dragging the
full LLM stack into unit tests.
"""

import threading
from datetime import timedelta

import pytest

from services.subconscious_worker import SubconsciousWorker
from services.time_utils import utc_now
from services.world_state import Signal, world_state

pytestmark = pytest.mark.unit


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_world_state():
    """Reset the WorldState singleton's store so each test starts cold."""
    with world_state._lock:
        saved = dict(world_state._store)
        world_state._store = {}
    try:
        yield world_state
    finally:
        with world_state._lock:
            world_state._store = saved


@pytest.fixture
def stub_worker(monkeypatch):
    """SubconsciousWorker instance whose steps are no-ops + persistence is in-memory.

    Gives every test the same starting point: a worker that records which
    steps fired, with persistence stubbed so MemoryStore + DataGraphService
    are not required.
    """
    fired: list[str] = []

    persisted: dict = {"value": None}

    def _persist(when):
        persisted["value"] = when

    def _consolidate(_self):
        fired.append("consolidate")
        return "stub"

    def _decay(_self):
        fired.append("decay")
        return "ok"

    def _patterns(_self):
        fired.append("patterns")
        return "candidates_added=0 promotions=0"

    def _synthesis(_self):
        fired.append("synthesis")
        return "ok"

    monkeypatch.setattr(SubconsciousWorker, "_step_consolidate", _consolidate)
    monkeypatch.setattr(SubconsciousWorker, "_step_decay", _decay)
    monkeypatch.setattr(SubconsciousWorker, "_step_patterns", _patterns)
    monkeypatch.setattr(SubconsciousWorker, "_step_synthesis", _synthesis)
    monkeypatch.setattr(SubconsciousWorker, "_persist_last_fired",
                        lambda _self, when: _persist(when))
    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage",
                        lambda _self: None)
    monkeypatch.setattr(SubconsciousWorker, "_is_bg_llm_saturated",
                        lambda _self: False)

    worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
    worker._fired = fired           # type: ignore[attr-defined]
    worker._persisted = persisted   # type: ignore[attr-defined]
    return worker


# ── Gate tests ───────────────────────────────────────────────────────────────


def test_cold_boot_passes_both_gates(isolated_world_state, stub_worker):
    """No user message, no prior fire → run all four steps."""
    result = stub_worker.run_once()
    assert "skipped" not in result
    assert stub_worker._fired == ["consolidate", "decay", "patterns", "synthesis"]
    assert all(step["status"] == "ok" for step in result["steps"].values())
    assert result["last_fired_at"] is not None


def test_user_active_gate_skips(isolated_world_state, stub_worker):
    """Recent user message within idle window → skip with reason 'user_active'."""
    isolated_world_state.absorb(Signal(
        source="ws",
        kind="user_message",
        payload={},
        received_at=utc_now() - timedelta(seconds=10),
    ))
    result = stub_worker.run_once()
    assert result.get("skipped") == "user_active"
    assert stub_worker._fired == []


def test_idle_window_passed_runs(isolated_world_state, stub_worker):
    """User message older than idle window → run all steps."""
    isolated_world_state.absorb(Signal(
        source="ws",
        kind="user_message",
        payload={},
        received_at=utc_now() - timedelta(seconds=120),  # > idle_window_sec=60
    ))
    result = stub_worker.run_once()
    assert "skipped" not in result
    assert stub_worker._fired == ["consolidate", "decay", "patterns", "synthesis"]


def test_already_fired_gate_skips(isolated_world_state, stub_worker):
    """last_fired_at newer than last_user_message_at → skip 'already_fired'."""
    last_msg_at = utc_now() - timedelta(seconds=600)
    isolated_world_state.absorb(Signal(
        source="ws",
        kind="user_message",
        payload={},
        received_at=last_msg_at,
    ))
    # Simulate a prior tick having fired AFTER that message.
    stub_worker._cached_last_fired = last_msg_at + timedelta(seconds=60)
    result = stub_worker.run_once()
    assert result.get("skipped") == "already_fired"
    assert stub_worker._fired == []


def test_fresh_message_resets_already_fired_gate(isolated_world_state, stub_worker):
    """A new user message after last_fired_at lets the next idle window fire again."""
    # Older fire timestamp.
    stub_worker._cached_last_fired = utc_now() - timedelta(minutes=10)
    # Newer user message — but still outside the idle window.
    isolated_world_state.absorb(Signal(
        source="ws",
        kind="user_message",
        payload={},
        received_at=utc_now() - timedelta(seconds=120),
    ))
    result = stub_worker.run_once()
    assert "skipped" not in result
    assert stub_worker._fired == ["consolidate", "decay", "patterns", "synthesis"]


# ── Step exception isolation ────────────────────────────────────────────────


def test_step_exception_does_not_block_others(isolated_world_state, monkeypatch):
    """A raise in step 2 must not prevent steps 3 + 4."""
    def _ok(name):
        return lambda _self: name

    def _boom(_self):
        raise RuntimeError("decay blew up")

    monkeypatch.setattr(SubconsciousWorker, "_step_consolidate", _ok("consolidate"))
    monkeypatch.setattr(SubconsciousWorker, "_step_decay", _boom)
    monkeypatch.setattr(SubconsciousWorker, "_step_patterns", _ok("patterns"))
    monkeypatch.setattr(SubconsciousWorker, "_step_synthesis", _ok("synthesis"))
    monkeypatch.setattr(SubconsciousWorker, "_persist_last_fired", lambda _self, _when: None)
    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage", lambda _self: None)
    monkeypatch.setattr(SubconsciousWorker, "_is_bg_llm_saturated", lambda _self: False)

    worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
    result = worker.run_once()

    assert result["steps"]["consolidate"]["status"] == "ok"
    assert result["steps"]["decay"]["status"] == "error"
    assert "decay blew up" in result["steps"]["decay"]["error"]
    assert result["steps"]["patterns"]["status"] == "ok"
    assert result["steps"]["synthesis"]["status"] == "ok"


# ── Backpressure ─────────────────────────────────────────────────────────────


def test_backpressure_skips_steps_3_and_4(isolated_world_state, monkeypatch):
    """When the bg LLM queue is saturated, only steps 1 + 2 run."""
    fired: list[str] = []
    monkeypatch.setattr(SubconsciousWorker, "_step_consolidate",
                        lambda _self: fired.append("consolidate") or "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_decay",
                        lambda _self: fired.append("decay") or "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_patterns",
                        lambda _self: fired.append("patterns") or "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_synthesis",
                        lambda _self: fired.append("synthesis") or "ok")
    monkeypatch.setattr(SubconsciousWorker, "_persist_last_fired", lambda _self, _when: None)
    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage", lambda _self: None)
    monkeypatch.setattr(SubconsciousWorker, "_is_bg_llm_saturated", lambda _self: True)

    worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
    result = worker.run_once()

    assert fired == ["consolidate", "decay"]
    assert result["steps"]["patterns"]["status"] == "skipped"
    assert result["steps"]["patterns"]["detail"] == "bg_llm_saturated"
    assert result["steps"]["synthesis"]["status"] == "skipped"
    assert result["steps"]["synthesis"]["detail"] == "bg_llm_saturated"


# ── Re-entrancy ──────────────────────────────────────────────────────────────


def test_concurrent_tick_returns_skipped_re_entrant(isolated_world_state, monkeypatch):
    """A second run_once() invoked while one is in flight returns skipped='re_entrant'."""
    enter = threading.Event()
    leave = threading.Event()

    def _slow_consolidate(_self):
        enter.set()
        leave.wait(timeout=2.0)
        return "slow"

    monkeypatch.setattr(SubconsciousWorker, "_step_consolidate", _slow_consolidate)
    monkeypatch.setattr(SubconsciousWorker, "_step_decay", lambda _self: "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_patterns", lambda _self: "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_synthesis", lambda _self: "ok")
    monkeypatch.setattr(SubconsciousWorker, "_persist_last_fired", lambda _self, _when: None)
    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage", lambda _self: None)
    monkeypatch.setattr(SubconsciousWorker, "_is_bg_llm_saturated", lambda _self: False)

    worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
    holder = {}

    t = threading.Thread(target=lambda: holder.setdefault("first", worker.run_once()))
    t.start()
    assert enter.wait(timeout=2.0), "first tick never entered consolidate"

    second = worker.run_once()
    leave.set()
    t.join(timeout=3.0)

    assert second.get("skipped") == "re_entrant"
    assert "skipped" not in holder["first"]


# ── Persistence (durable hydrate path) ──────────────────────────────────────


def test_hydrate_loads_cached_last_fired(monkeypatch):
    """When the durable store has a value, the new worker hydrates from it.

    Simulates a process restart: a fresh SubconsciousWorker built with the
    durable store populated must surface the persisted timestamp via its
    cached property.
    """
    target = utc_now() - timedelta(minutes=15)

    def _load(_self):
        return target

    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage", _load)

    worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
    assert worker.last_fired_at == target


def test_persist_called_with_recent_timestamp(isolated_world_state, stub_worker):
    """After a successful tick, last_fired_at is bumped to ~now."""
    before = utc_now()
    result = stub_worker.run_once()
    after = utc_now()
    persisted = stub_worker._persisted["value"]
    assert persisted is not None
    assert before <= persisted <= after
    assert result["last_fired_at"] is not None


# ── Backpressure default (P0 gap) ────────────────────────────────────────────


def test_bg_saturated_check_returns_false_on_exception(isolated_world_state, monkeypatch):
    """_is_bg_llm_saturated must return False when the queue check itself raises.

    The production defensive contract: a broken MemoryStore connection or missing
    import must never flip the default to True and start suppressing steps 3+4.
    Calling the real method with a poisoned import confirms the fallback.
    """
    import sys

    # Poison the MemoryClientService import so the queue-depth path raises.
    original = sys.modules.get("services.memory_client")
    broken = type(sys)("services.memory_client")

    class _BrokenService:
        @staticmethod
        def create_connection():
            raise RuntimeError("connection refused")

    broken.MemoryClientService = _BrokenService
    sys.modules["services.memory_client"] = broken

    try:
        worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
        result = worker._is_bg_llm_saturated()
    finally:
        if original is None:
            sys.modules.pop("services.memory_client", None)
        else:
            sys.modules["services.memory_client"] = original

    assert result is False, (
        "_is_bg_llm_saturated must default to False on exception "
        "so that steps 3+4 are not silently suppressed when the queue check breaks"
    )


# ── Synthesis result propagation (P0 gap) ────────────────────────────────────


def test_synthesis_skipped_result_propagates_to_step_detail(isolated_world_state, monkeypatch):
    """_step_synthesis returns a 'skipped' string when UserSummaryProcessor returns falsy.

    If the branch `return "ok" if result else "no new traits/patterns; skipped"` is
    accidentally collapsed to always return "ok", the step dict will carry the wrong
    detail and observability breaks.
    """
    monkeypatch.setattr(SubconsciousWorker, "_step_consolidate", lambda _self: "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_decay", lambda _self: "ok")
    monkeypatch.setattr(SubconsciousWorker, "_step_patterns",
                        lambda _self: "candidates_added=0 promotions=0")
    monkeypatch.setattr(SubconsciousWorker, "_persist_last_fired", lambda _self, _when: None)
    monkeypatch.setattr(SubconsciousWorker, "_load_last_fired_from_storage", lambda _self: None)
    monkeypatch.setattr(SubconsciousWorker, "_is_bg_llm_saturated", lambda _self: False)

    # Patch UserSummaryProcessor inside the method's import scope so .send() returns "".
    import types as _types
    fake_usp_mod = _types.ModuleType("services.user_summary_processor")

    class _SkippingProcessor:
        def __init__(self):
            pass

        def send(self):
            return ""  # falsy → synthesis self-gated

    fake_usp_mod.UserSummaryProcessor = _SkippingProcessor
    import sys
    original_usp = sys.modules.get("services.user_summary_processor")
    sys.modules["services.user_summary_processor"] = fake_usp_mod

    try:
        worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60, bg_queue_threshold=5)
        result = worker.run_once()
    finally:
        if original_usp is None:
            sys.modules.pop("services.user_summary_processor", None)
        else:
            sys.modules["services.user_summary_processor"] = original_usp

    synthesis_step = result["steps"]["synthesis"]
    assert synthesis_step["status"] == "ok"
    assert "skipped" in synthesis_step["detail"], (
        "When UserSummaryProcessor.send() returns falsy, step detail must signal "
        "that synthesis was self-gated, not claim 'ok'"
    )
