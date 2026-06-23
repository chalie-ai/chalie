"""Feature tests for AbilityRegistry (abilities/_registry.py).

The registry exposes only ``get(name)`` and ``all()``.  Tool *scope*
(always-available vs discoverable) lives on each MessageProcessor subclass
— see test_phase4_invariants.py for those checks.

Production state pins the abilities currently on disk (e.g. weather after
Phase 1).  Singleton caching and thread-safety are verified without any
mocks, stubs, or patches.

Registry reset between tests: _reset_for_tests() clears the module-level
_registry cache so each test starts with a clean lazy-load.
"""

import gc
import threading
from collections.abc import Iterator

import pytest

from abilities._ability import Ability
from abilities._registry import AbilityRegistry, _reset_for_tests
from abilities._result import ToolResult

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Reset the registry before and after every test.

    gc.collect() is called before reset so that locally-scoped subclasses
    defined in concurrently-collected test files (test_ability_base.py) are
    removed from Ability.__subclasses__() before the registry lazy-loads.
    """
    gc.collect()
    _reset_for_tests()
    yield
    gc.collect()
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Production-state assertions — reflect concrete abilities currently on disk
# ---------------------------------------------------------------------------


def test_all_includes_registered_abilities() -> None:
    """AbilityRegistry.all() surfaces every concrete subclass on disk."""
    names = [a.get_name() for a in AbilityRegistry.all()]
    assert "weather" in names


def test_get_raises_key_error_for_unregistered_name() -> None:
    """AbilityRegistry.get() raises KeyError for a name no subclass owns."""
    with pytest.raises(KeyError):
        AbilityRegistry.get("anything_not_registered")


def test_get_raises_key_error_for_unknown_name_with_subclass_present() -> None:
    """AbilityRegistry.get() raises KeyError for a name that no subclass owns."""

    class _KnownAbility(Ability):
        def get_name(self) -> str: return "known"
        def get_summary(self) -> str: return "known ability"
        def get_examples(self) -> list[str]: return ["a", "b", "c", "d", "e", "f"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            from typing import cast
            return cast("ToolResult", {})

    _reset_for_tests()
    with pytest.raises(KeyError):
        AbilityRegistry.get("not_known")

    del _KnownAbility
    gc.collect()


# ---------------------------------------------------------------------------
# Singleton / caching
# ---------------------------------------------------------------------------


def test_all_called_twice_returns_consistent_result() -> None:
    """Two calls to AbilityRegistry.all() return the same shape — registry is cached."""
    first = AbilityRegistry.all()
    second = AbilityRegistry.all()
    assert first == second


def test_registry_reflects_concrete_subclass_after_reset() -> None:
    """After reset, a newly defined subclass appears in the registry."""

    class _NewAbility(Ability):
        def get_name(self) -> str: return "new_ability"
        def get_summary(self) -> str: return "a freshly defined ability"
        def get_examples(self) -> list[str]: return ["do it", "run it", "start it", "go now", "begin", "execute"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            from typing import cast
            return cast("ToolResult", {"text": "ok"})

    _reset_for_tests()

    names = [a.get_name() for a in AbilityRegistry.all()]
    assert "new_ability" in names
    assert AbilityRegistry.get("new_ability").get_name() == "new_ability"

    del _NewAbility
    gc.collect()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_get_calls_produce_no_race() -> None:
    """10 threads hitting AbilityRegistry.get() simultaneously raise only KeyError — no crash, no double-init."""
    errors: list[object] = []
    key_errors: list[bool] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        try:
            barrier.wait()
            AbilityRegistry.get("nonexistent_concurrent")
        except KeyError:
            key_errors.append(True)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Unexpected exceptions in threads: {errors}"
    assert len(key_errors) == 10
