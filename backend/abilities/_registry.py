import importlib
import threading
from pathlib import Path

from abilities._base import Ability

_lock = threading.RLock()
_registry: dict[str, Ability] | None = None


def _load() -> dict[str, Ability]:
    """Walk backend/abilities/ and import every non-underscore .py module.

    Concrete Ability subclasses self-register via __init_subclass__; we collect
    them after the walk by inspecting all subclasses of Ability.
    """
    abilities_dir = Path(__file__).resolve().parent
    for path in sorted(abilities_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"abilities.{path.stem}"
        importlib.import_module(module_name)

    result: dict[str, Ability] = {}
    for subclass in _all_concrete_subclasses(Ability):
        instance = subclass()
        result[instance.NAME] = instance
    return result


def _all_concrete_subclasses(cls: type) -> list[type]:
    """Recursively collect unique concrete (non-abstract) subclasses of *cls*."""
    seen: set[type] = set()
    out: list[type] = []

    def _walk(node: type) -> None:
        for sub in node.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            if not getattr(sub, "__abstractmethods__", None):
                out.append(sub)
            _walk(sub)

    _walk(cls)
    return out


def _get_registry() -> dict[str, Ability]:
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is None:
            _registry = _load()
    return _registry


class AbilityRegistry:
    """Registry of every dispatchable Ability subclass.

    Tool *scope* (always-available vs discoverable) is owned by each
    MessageProcessor subclass — the registry just provides the dispatch
    lookup and the full inventory.
    """

    @staticmethod
    def get(name: str) -> Ability:
        """Return the Ability instance for *name*; raises KeyError on miss."""
        return _get_registry()[name]

    @staticmethod
    def all() -> list[Ability]:
        """Return every registered Ability instance."""
        return list(_get_registry().values())


def _reset_for_tests() -> None:
    """Reset the module-level registry cache.  Test-only — never call in production."""
    global _registry
    _registry = None
