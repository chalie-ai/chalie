import importlib
import logging
import threading

from abilities._ability import Ability
from abilities._mcp_ability import _MCPAbility
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_registry: dict[str, Ability] | None = None


def _load() -> dict[str, Ability]:
    """Walk backend/abilities/ and import every non-underscore .py module.

    Concrete Ability subclasses self-register via __init_subclass__; we collect
    them after the walk by inspecting all subclasses of Ability.
    """
    abilities_dir = FileMapperService.get_abilities_path()
    for path in sorted(abilities_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"abilities.{path.stem}"
        importlib.import_module(module_name)

    result: dict[str, Ability] = {}
    for subclass in _all_concrete_subclasses(Ability):
        instance = subclass()
        result[instance.get_name()] = instance
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
            if getattr(sub, "_SYNTHETIC", False):
                continue
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

    @staticmethod
    def build_tools(mp: "object") -> list[dict]:
        """Resolve ``mp.active_tools`` to native tool schemas for this ACT turn.

        ``active_tools`` is the live list of tool NAMES available this turn:
        seeded with ``config.always_available`` by ``_setup`` and appended to by
        ``find_tools``. Each name resolves to a FRESH per-turn Ability instance
        bound to *mp* (native → registry template copy; ``_mcp_*`` → synthetic
        ``_MCPAbility`` proxy) and its schema is assembled by the one ``final``
        ``get_input_schema()`` — which injects ``act_summary`` (always) and
        ``async`` (iff this channel backgrounds). Binding *mp* lets a getter
        enrich for the live request (e.g. bash's cwd, find_tools' index). First-
        seen wins on dupes; unknown names are logged and skipped. Returns ``[]``
        when no active_tools are bound (compaction / encoder paths, or pre-_setup).
        """
        active = list(getattr(mp, "active_tools", None) or [])
        registry = _get_registry()

        seen: set[str] = set()
        result: list[dict] = []
        for name in active:
            if name in seen:
                continue
            seen.add(name)

            if name.startswith("_mcp_"):
                ability = _MCPAbility(name, mp=mp)
                if ability.remote_schema() is None:
                    logger.warning(
                        "[AbilityRegistry.build_tools] No MCP schema for '%s'", name
                    )
                    continue
            else:
                template = registry.get(name)
                if template is None:
                    logger.warning(
                        "[AbilityRegistry.build_tools] No ability registered for '%s'",
                        name,
                    )
                    continue
                # Fresh per-turn instance bound to mp — getters may read mp to
                # enrich (cwd / index) and the singleton template stays mp=None.
                ability = type(template)(mp=mp)

            result.append(ability.get_input_schema())

        return result


def _reset_for_tests() -> None:
    """Reset the module-level registry cache.  Test-only — never call in production."""
    global _registry
    _registry = None
