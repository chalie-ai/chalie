import copy
import importlib
import logging
import threading

from abilities._ability import Ability
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_registry: dict[str, Ability] | None = None

# Injected into every tool's input_schema by ``build_tools`` and popped by
# ``ToolDispatcher.dispatch()`` before the ability sees it (spec §6, message-
# processing.md L1166). Carries the user-facing tooltip for the call.
_ACT_SUMMARY_PROPERTY: dict = {
    'type': 'string',
    'description': (
        'A ~3-10 word summary of what this specific tool call does, shown to'
        ' the user as a tooltip (e.g. "Searching for laptops in Malta",'
        ' "Looking up the weather in London").'
    ),
}


def _with_act_summary(schema: dict) -> dict:
    """Return a copy of *schema* with ``act_summary`` injected into input_schema.

    Deep-copies input_schema so the originating Ability's ClassVar dict is
    never mutated. ``act_summary`` is made a required property so the model
    always supplies a tooltip for each tool call.
    """
    input_schema = copy.deepcopy(schema.get('input_schema') or {})
    properties = input_schema.setdefault('properties', {})
    properties['act_summary'] = dict(_ACT_SUMMARY_PROPERTY)
    required = input_schema.setdefault('required', [])
    if 'act_summary' not in required:
        required.append('act_summary')
    return {
        'name': schema['name'],
        'description': schema['description'],
        'input_schema': input_schema,
    }


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
        ``find_tools``. Native names resolve via the registry; ``_mcp_*`` names
        via ``McpClientService().get_tool_schema``. First-seen wins on dupes;
        unknown names are logged and skipped. Every schema gets ``act_summary``
        injected (spec §6); ``ToolDispatcher.dispatch`` pops it back out. Returns ``[]`` when
        no active_tools are bound (compaction / encoder paths, or pre-_setup).
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
                from services.mcp_client_service import McpClientService  # noqa: PLC0415
                schema = McpClientService().get_tool_schema(name)
                if schema is None:
                    logger.warning(
                        "[AbilityRegistry.build_tools] No MCP schema for '%s'", name
                    )
                    continue
            else:
                ability = registry.get(name)
                if ability is None:
                    logger.warning(
                        "[AbilityRegistry.build_tools] No ability registered for '%s'",
                        name,
                    )
                    continue
                schema = {
                    'name': ability.NAME,
                    'description': ability.SUMMARY,
                    'input_schema': ability.get_input_schema(mp),
                }

            result.append(_with_act_summary(schema))

        return result


def _reset_for_tests() -> None:
    """Reset the module-level registry cache.  Test-only — never call in production."""
    global _registry
    _registry = None
