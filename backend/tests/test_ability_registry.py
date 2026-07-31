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
from configs.enums.ability_category import AbilityCategory

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
    names = [a.NAME for a in AbilityRegistry.all()]
    assert "weather" in names


def test_get_raises_key_error_for_unregistered_name() -> None:
    """AbilityRegistry.get() raises KeyError for a name no subclass owns."""
    with pytest.raises(KeyError):
        AbilityRegistry.get("anything_not_registered")


# ---------------------------------------------------------------------------
# Singleton / caching
# ---------------------------------------------------------------------------


def test_registry_reflects_concrete_subclass_after_reset() -> None:
    """After reset, a newly defined subclass appears in the registry."""

    class _NewAbility(Ability):
        NAME = "new_ability"
        CATEGORY = AbilityCategory.SYSTEM
        def get_summary(self) -> str: return "a freshly defined ability"
        def get_examples(self) -> list[str]: return ["do it", "run it", "start it", "go now", "begin", "execute"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            from typing import cast
            return cast("ToolResult", {"text": "ok"})

    _reset_for_tests()

    names = [a.NAME for a in AbilityRegistry.all()]
    assert "new_ability" in names
    assert AbilityRegistry.get("new_ability").NAME == "new_ability"

    del _NewAbility
    gc.collect()


def test_missing_name_fails_load_with_loud_value_error() -> None:
    """A concrete subclass that never declares NAME must fail registry load with
    the crafted ValueError naming the class — not a raw AttributeError."""

    class _NamelessAbility(Ability):
        def get_summary(self) -> str: return "an ability that forgot its NAME"
        def get_examples(self) -> list[str]: return ["a", "b", "c", "d", "e", "f"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            from typing import cast
            return cast("ToolResult", {"text": "ok"})

    _reset_for_tests()
    with pytest.raises(ValueError, match="_NamelessAbility"):
        AbilityRegistry.all()

    del _NamelessAbility
    gc.collect()
    _reset_for_tests()

    # The registry rebuilds cleanly once the offender is gone.
    assert AbilityRegistry.get("weather").NAME == "weather"


def test_discoverable_without_category_fails_load_with_loud_value_error() -> None:
    """A DISCOVERABLE ability with no CATEGORY has no heading to render under in
    the find_tools menu — it would be activatable by name yet listed nowhere. The
    registry must refuse to load it, naming the ability and the class."""

    class _UncategorisedAbility(Ability):
        NAME = "uncategorised_ability"
        def get_summary(self) -> str: return "discoverable but has no category"
        def get_examples(self) -> list[str]: return ["a", "b", "c", "d", "e", "f"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("ok")

    _reset_for_tests()
    with pytest.raises(ValueError, match="_UncategorisedAbility"):
        AbilityRegistry.all()

    del _UncategorisedAbility
    gc.collect()
    _reset_for_tests()

    assert AbilityRegistry.get("weather").NAME == "weather"


def test_non_discoverable_without_category_loads_fine() -> None:
    """The CATEGORY requirement is scoped to the menu: a DISCOVERABLE=False
    ability never appears there, so leaving CATEGORY None must NOT block load."""

    class _HiddenAbility(Ability):
        NAME = "hidden_ability"
        DISCOVERABLE = False
        def get_summary(self) -> str: return "pinned only, never discovered"
        def get_examples(self) -> list[str]: return ["a", "b", "c", "d", "e", "f"]
        def get_search_tooltip(self) -> str: return ""
        def get_parameters(self) -> dict[str, object]: return {}

        def run(self, params: dict[str, object]) -> ToolResult:
            return ToolResult.ok("ok")

    _reset_for_tests()
    assert AbilityRegistry.get("hidden_ability").CATEGORY is None
    assert "hidden_ability" not in AbilityRegistry.discoverable_names()

    del _HiddenAbility
    gc.collect()
    _reset_for_tests()


def test_every_shipped_discoverable_ability_declares_a_category() -> None:
    """The gate above proves the rule is enforced; this proves the shipped roster
    actually satisfies it, so every tool renders under a heading."""
    for name in AbilityRegistry.discoverable_names():
        assert AbilityRegistry.get(name).CATEGORY is not None, (
            f"discoverable ability '{name}' declares no CATEGORY"
        )


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
