import copy
import importlib
import logging
import threading

from abilities._base import Ability
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_registry: dict[str, Ability] | None = None

# Injected into every tool's input_schema by ``build_tools`` and popped by
# ``Ability.dispatch()`` before the ability sees it (spec §6, message-
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
        """Return the full native tool list for the current ACT iteration.

        Resolution order (first-seen wins on duplicates):
        1. ``config.always_available`` — the innate tier pinned in every LLM
           call for this channel (resolved via the registry).
        2. ``mp.discovered_tools`` — abilities surfaced this turn via
           ``find_tools`` (gated by ``DISCOVERABLE`` inside find_tools itself).

        Each schema is ``{name, description, input_schema}`` pulled from the
        Ability's ``NAME`` / ``SUMMARY`` / ``get_input_schema()``. Every entry
        gets ``act_summary`` injected (spec §6) before it reaches the model;
        ``Ability.dispatch()`` pops it back out. Falls back to an empty list
        when no config is bound (compaction / encoder paths whose
        ``always_available`` is empty by design).
        """
        config = getattr(mp, "config", None)
        always = list(getattr(config, "always_available", None) or [])

        registry = _get_registry()
        native: list[dict] = []
        for tool_name in always:
            ability = registry.get(tool_name)
            if ability is None:
                logger.warning(
                    "[AbilityRegistry.build_tools] No ability registered for '%s'",
                    tool_name,
                )
                continue
            native.append({
                'name': ability.NAME,
                'description': ability.SUMMARY,
                'input_schema': ability.get_input_schema(),
            })

        dynamic = list(getattr(mp, "discovered_tools", None) or [])

        seen: set[str] = set()
        result: list[dict] = []
        for schema in native + dynamic:
            name = schema.get('name')
            if name and name not in seen:
                seen.add(name)
                result.append(_with_act_summary(schema))

        return result

    @staticmethod
    def policy_visible() -> list[Ability]:
        """Return abilities that should appear in the policy UI.

        Excludes SYSTEM and actionless ALWAYS_AVAILABLE meta-tools
        (find_tools, find_skills) whose denial would break routing.
        """
        from services.message_processor import MessageProcessor
        always_available = set(MessageProcessor.ALWAYS_AVAILABLE)
        return [
            a for a in _get_registry().values()
            if not getattr(a, "SYSTEM", False)
            and not (
                a.NAME in always_available
                and not a.INPUT_SCHEMA.get("properties", {}).get("action")
            )
        ]


def _reset_for_tests() -> None:
    """Reset the module-level registry cache.  Test-only — never call in production."""
    global _registry
    _registry = None
