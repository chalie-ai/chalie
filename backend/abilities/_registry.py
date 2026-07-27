import importlib
import logging
import re
import threading
from typing import TYPE_CHECKING, cast

from abilities._ability import Ability
from abilities._mcp_ability import _MCPAbility
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_registry: dict[str, Ability] | None = None


def _load() -> dict[str, Ability]:
    """Importing every ability module makes its class object exist; we collect the
    concrete ones afterwards by walking ``Ability.__subclasses__()``.
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
        name = getattr(instance, "NAME", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Ability {subclass.__name__} has no valid NAME; "
                "must be a non-empty string"
            )
        if name in result:
            raise ValueError(
                f"Duplicate ability NAME '{name}' in {subclass.__name__} "
                f"and {result[name].__class__.__name__}"
            )
        result[name] = instance
    return result


def _all_concrete_subclasses(cls: type) -> list[type]:
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

    Tool *scope* is two flags: an ability is pinned directly on a
    MessageProcessor (``config.always_available``) or — iff its
    ``Ability.DISCOVERABLE`` is True — reachable through ``find_tools``. The
    registry owns the full inventory, the dispatch lookup, and the single global
    roster of discoverable names that ``find_tools`` searches over.
    """

    @staticmethod
    def get(name: str) -> Ability:
        """Raises KeyError on miss."""
        return _get_registry()[name]

    @staticmethod
    def all() -> list[Ability]:
        return list(_get_registry().values())

    @staticmethod
    def discoverable_names() -> set[str]:
        """The global set of ability names ``find_tools`` may surface.

        This is the single source of truth for discovery scope: every ability
        whose ``DISCOVERABLE`` flag is True. ``find_tools`` lists exactly this
        set in its menu and resolves names against it via ``discovery_aliases``.
        MCP proxies (``_mcp_*``) are not in the registry — they are surfaced by
        ``mcp_tools``.
        """
        return {name for name, a in _get_registry().items() if a.DISCOVERABLE}

    @classmethod
    def discovery_aliases(cls) -> dict[str, str]:
        """Map every normalized alias → canonical name for discoverable abilities.

        For each DISCOVERABLE ability the canonical normalized name AND each
        SEARCHABLE_AS entry is mapped to the canonical name. If two abilities
        claim the same alias, a RuntimeError is raised naming both abilities
        and the alias so misconfiguration fails loud at build time.
        """
        out: dict[str, str] = {}
        for ability in _get_registry().values():
            if not ability.DISCOVERABLE:
                continue
            for alias in (ability.NAME, *ability.SEARCHABLE_AS):
                norm = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
                if not norm:
                    continue
                existing = out.get(norm)
                if existing is not None and existing != ability.NAME:
                    raise RuntimeError(
                        f"Ability alias collision: abilities '{existing}' and "
                        f"'{ability.NAME}' both normalize to alias "
                        f"'{norm}'"
                    )
                out[norm] = ability.NAME
        return out

    @staticmethod
    def build_tools(mp: "MessageProcessor | None") -> list[dict[str, object]]:
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
        result: list[dict[str, object]] = []
        for name in active:
            if name in seen:
                continue
            seen.add(name)

            if name.startswith("_mcp_"):
                ability: Ability = _MCPAbility(name, mp=mp)
                if cast("_MCPAbility", ability).remote_schema() is None:
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
