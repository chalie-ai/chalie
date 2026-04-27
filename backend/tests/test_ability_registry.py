"""Feature tests for AbilityRegistry (abilities/_registry.py).

Production state pins the abilities currently on disk (e.g. weather after
Phase 1).  Singleton caching and thread-safety are verified without any
mocks, stubs, or patches.

Registry reset between tests: _reset_for_tests() clears the module-level
_registry cache so each test starts with a clean lazy-load.
"""

import gc
import threading

import pytest

from abilities._base import Ability
from abilities._registry import AbilityRegistry, _reset_for_tests

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the registry before and after every test.

    gc.collect() is called before reset so that locally-scoped subclasses
    defined in concurrently-collected test files (test_ability_base.py) are
    removed from Ability.__subclasses__() before the registry lazy-loads.
    """
    import gc
    gc.collect()
    _reset_for_tests()
    yield
    gc.collect()
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Production-state assertions — reflect concrete abilities currently on disk
# ---------------------------------------------------------------------------


def test_all_includes_registered_abilities():
    """AbilityRegistry.all() surfaces every concrete subclass on disk."""
    names = [a.NAME for a in AbilityRegistry.all()]
    assert "weather" in names


def test_always_available_names_excludes_discoverable_abilities():
    """always_available_names() omits abilities with ALWAYS_AVAILABLE=False."""
    assert "weather" not in AbilityRegistry.always_available_names()


def test_discoverable_includes_non_always_available_abilities():
    """discoverable() includes abilities marked ALWAYS_AVAILABLE=False."""
    names = [a.NAME for a in AbilityRegistry.discoverable()]
    assert "weather" in names


def test_get_raises_key_error_for_unregistered_name():
    """AbilityRegistry.get() raises KeyError for a name no subclass owns."""
    with pytest.raises(KeyError):
        AbilityRegistry.get("anything_not_registered")


def test_get_raises_key_error_for_unknown_name_with_subclass_present():
    """AbilityRegistry.get() raises KeyError for a name that no subclass owns."""

    class _KnownAbility(Ability):
        NAME = "known"
        SUMMARY = "known ability"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    _reset_for_tests()
    with pytest.raises(KeyError):
        AbilityRegistry.get("not_known")

    del _KnownAbility
    gc.collect()


# ---------------------------------------------------------------------------
# Singleton / caching
# ---------------------------------------------------------------------------


def test_all_called_twice_returns_consistent_result():
    """Two calls to AbilityRegistry.all() return the same shape — registry is cached."""
    first = AbilityRegistry.all()
    second = AbilityRegistry.all()
    assert first == second


def test_registry_reflects_concrete_subclass_after_reset():
    """After reset, a newly defined subclass appears in the registry."""

    class _NewAbility(Ability):
        NAME = "new_ability"
        SUMMARY = "a freshly defined ability"
        EXAMPLES = ["do it", "run it", "start it", "go now", "begin", "execute"]
        INPUT_SCHEMA = {}
        ALWAYS_AVAILABLE = True

        def execute(self, channel, params, telemetry):
            return {"text": "ok"}

    _reset_for_tests()

    names = [a.NAME for a in AbilityRegistry.all()]
    assert "new_ability" in names
    assert AbilityRegistry.get("new_ability").NAME == "new_ability"
    assert "new_ability" in AbilityRegistry.always_available_names()
    discoverable_names = [a.NAME for a in AbilityRegistry.discoverable()]
    assert "new_ability" not in discoverable_names

    del _NewAbility
    gc.collect()


def test_discoverable_excludes_always_available():
    """Discoverable list excludes abilities with ALWAYS_AVAILABLE=True."""

    class _AlwaysOn(Ability):
        NAME = "always_on_disc"
        SUMMARY = "always on ability"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}
        ALWAYS_AVAILABLE = True

        def execute(self, channel, params, telemetry):
            return {}

    class _Discoverable(Ability):
        NAME = "discoverable_disc"
        SUMMARY = "discoverable ability"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}
        ALWAYS_AVAILABLE = False

        def execute(self, channel, params, telemetry):
            return {}

    _reset_for_tests()

    discoverable_names = [a.NAME for a in AbilityRegistry.discoverable()]
    always_names = AbilityRegistry.always_available_names()

    assert "always_on_disc" not in discoverable_names
    assert "discoverable_disc" in discoverable_names
    assert "always_on_disc" in always_names
    assert "discoverable_disc" not in always_names

    del _AlwaysOn, _Discoverable
    gc.collect()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_get_calls_produce_no_race():
    """10 threads hitting AbilityRegistry.get() simultaneously raise only KeyError — no crash, no double-init."""
    errors = []
    key_errors = []
    barrier = threading.Barrier(10)

    def worker():
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
