"""
Tool Schema Service — builds native tool definitions for skills and external tools.

Converts innate skill TOOL_SCHEMA dicts and external tool manifests into the
provider-agnostic format expected by the LLM service layer's `tools` parameter.
Each provider then converts to its native format internally.

The normalized tool schema format (matches Anthropic's native format):
{
    "name": "tool_or_skill_name",
    "description": "What it does...",
    "input_schema": {
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
"""

import logging
import inspect

logger = logging.getLogger(__name__)

# Cache to avoid repeated module introspection
_schema_cache: dict = {}


def get_skill_schemas(skill_names: list = None) -> list:
    """Build native tool schema definitions for the given innate skills.

    Args:
        skill_names: List of skill names. If None, returns schemas for ALL
                     registered innate skills.

    Returns:
        List of tool schema dicts ready for the LLM `tools` parameter.
    """
    if skill_names is None:
        from services.innate_skills.registry import ALL_SKILL_NAMES
        skill_names = list(ALL_SKILL_NAMES)

    from services.innate_skills import get_skill_handler

    schemas = []
    for name in skill_names:
        # Check cache first
        if name in _schema_cache:
            schemas.append(_schema_cache[name])
            continue

        handler = get_skill_handler(name)
        if not handler:
            logger.warning(f"[TOOL SCHEMA] No handler for skill '{name}'")
            continue

        module = inspect.getmodule(handler)
        schema = getattr(module, 'TOOL_SCHEMA', None)
        if not schema:
            logger.warning(f"[TOOL SCHEMA] No TOOL_SCHEMA for skill '{name}'")
            continue

        # Validate minimal structure
        if 'name' not in schema or 'input_schema' not in schema:
            logger.warning(f"[TOOL SCHEMA] Invalid TOOL_SCHEMA for skill '{name}': missing name or input_schema")
            continue

        _schema_cache[name] = schema
        schemas.append(schema)

    return schemas


def get_external_tool_schemas(tool_names: list) -> list:
    """Build native tool schemas from external tool manifests.

    Converts the manifest parameter format (dict of name→{type, required, ...})
    into the Anthropic-style input_schema format used by the LLM tools parameter.

    Args:
        tool_names: List of registered external tool names.

    Returns:
        List of tool schema dicts ready for the LLM ``tools`` parameter.
        Tools that are not found in the registry are silently skipped.
    """
    from services.tool_registry_service import ToolRegistryService
    registry = ToolRegistryService()

    schemas = []
    for name in tool_names:
        tool_data = registry.tools.get(name)
        if not tool_data:
            logger.debug(f"[TOOL SCHEMA] External tool '{name}' not in registry, skipping")
            continue

        manifest = tool_data.get('manifest', {})
        schema = _manifest_to_schema(manifest)
        if schema:
            schemas.append(schema)

    return schemas


def _manifest_to_schema(manifest: dict) -> dict | None:
    """Convert an external tool manifest to native tool schema format.

    Fast path: if the manifest already has an ``input_schema`` key (first-party
    built-in tools), return it directly without conversion.  The old
    ``parameters`` conversion path is kept for backwards compatibility with
    interface tools that still use the legacy format.

    Args:
        manifest: Tool manifest dict with name, description, and either
                  ``input_schema`` (preferred) or ``parameters`` (legacy).

    Returns:
        Schema dict or None if the manifest is missing a name.
    """
    name = manifest.get('name')
    if not name:
        return None

    description = manifest.get('description', '')

    # Fast path: manifest already uses input_schema (built-in tools)
    if 'input_schema' in manifest:
        return {
            "name": name,
            "description": description,
            "input_schema": manifest['input_schema'],
        }

    # Legacy path: convert parameters dict for interface tools
    params = manifest.get('parameters', {})

    properties = {}
    required = []
    for pname, pdef in params.items():
        prop = {"type": pdef.get("type", "string")}
        if pdef.get("description"):
            prop["description"] = pdef["description"]
        if pdef.get("default") is not None:
            prop["default"] = pdef["default"]
        if pdef.get("enum"):
            prop["enum"] = pdef["enum"]
        if pdef.get("items"):
            prop["items"] = pdef["items"]
        properties[pname] = prop
        if pdef.get("required", False):
            required.append(pname)

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def clear_cache():
    """Clear the schema cache (for testing)."""
    _schema_cache.clear()
