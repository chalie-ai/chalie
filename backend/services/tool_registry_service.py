"""
Tool Registry Service — First-party tool loading and dispatch.

Singleton. First-party tools are declared in ToolLibraryService (metadata
and handlers in Python code — no manifest.json, no subprocess, no runner.py).
Interface tools are registered dynamically via register_interface_tool().

First-party tools are invoked directly in-process. Interface tools are
invoked via HTTP to the paired interface.

Tool output is sanitized and wrapped in [TOOL:name]...[/TOOL] markers.
Cost metadata is appended to every result.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance
_instance = None


class ToolRegistryService:
    """
    Registry for first-party tools and interface-provided tools.

    Singleton — created at startup. First-party tools are declared in
    ToolLibraryService and invoked directly in-process. Interface tools
    register dynamically via register_interface_tool() and are invoked
    via HTTP to the paired interface.
    """

    # Delegated to shared utilities (tool_output_utils.py)
    from services.tool_output_utils import MAX_OUTPUT_CHARS

    def __new__(cls, *args, **kwargs):
        """Return the existing singleton instance, creating it on first call.

        Stores the sole instance in the module-level ``_instance`` variable and
        sets ``_initialized = False`` so that ``__init__`` can gate its one-time
        setup logic.

        Returns:
            The singleton ``ToolRegistryService`` instance.
        """
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(self):
        """Initialise the singleton registry and load first-party tools.

        Idempotent — subsequent calls on the already-initialised singleton are
        no-ops (guarded by ``_initialized``).

        Reads ``configs/frontal-cortex.json`` to check the ``tools_enabled``
        kill-switch.  When disabled, loading is skipped entirely.
        """
        if self._initialized:
            return
        self._initialized = True

        self.tools: Dict[str, dict] = {}  # name -> {manifest, source_type, ...}
        self._enabled = True
        self._lock = threading.Lock()

        try:
            from services.config_service import ConfigService
            fc_config = ConfigService.get_agent_config("frontal-cortex")
            self._enabled = fc_config.get("tools_enabled", True)
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] config load failed, defaulting to enabled: {e}")
            self._enabled = True

        if not self._enabled:
            logger.info("[TOOL REGISTRY] Tools disabled via kill switch (tools_enabled=false)")
            return

        self._load_tools()

    def _load_tools(self):
        """Load first-party tools from ToolLibraryService."""
        try:
            from services.tool_library_service import TOOL_METADATA, get_all_tool_names
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] Failed to import ToolLibraryService: {e}")
            return

        # Filter out DB-disabled tools
        disabled = set()
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            config_svc = ToolConfigService(get_shared_db_service())
            all_names = get_all_tool_names()
            for name in all_names:
                if not config_svc.is_tool_enabled(name):
                    logger.info(f"[TOOL REGISTRY] Skipping disabled tool '{name}'")
                    disabled.add(name)
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] Could not check disabled status: {e}")
            all_names = get_all_tool_names()

        for name in all_names:
            if name in disabled:
                continue
            metadata = TOOL_METADATA.get(name, {})
            with self._lock:
                self.tools[name] = {
                    "manifest": metadata,
                    "source_type": "library",
                }

        if self.tools:
            names = ", ".join(sorted(self.tools.keys()))
            logger.info(f"[TOOL REGISTRY] Loaded {len(self.tools)} tools: {names}")
        else:
            logger.info("[TOOL REGISTRY] No tools loaded")

        # Kick off profile enrichment in background (skip built-in tools — they are seeded at startup)
        from services.tool_library_service import BUILTIN_TOOL_PROFILES
        for name in list(self.tools.keys()):
            if name in BUILTIN_TOOL_PROFILES:
                continue
            _name = name
            _metadata = dict(self.tools[name]["manifest"])

            def _build_profile(n=_name, m=_metadata):
                try:
                    from services.tool_profile_service import ToolProfileService
                    ToolProfileService().build_profile(n, m)
                except Exception as _e:
                    logger.warning(f"[TOOL REGISTRY] Profile build failed for '{n}': {_e}")

            threading.Thread(target=_build_profile, daemon=True,
                             name=f"profile-build-{name}").start()

    def _refresh_oauth_token(self, tool_name: str, manifest: dict, settings: dict) -> dict:
        """Refresh OAuth token if manifest declares auth.type == 'oauth2'.

        Returns the settings dict with a fresh _oauth_access_token if refreshed.
        """
        auth = manifest.get("auth", {})
        if auth.get("type") != "oauth2":
            return settings
        if not settings.get("_oauth_access_token"):
            return settings
        try:
            from services.oauth_service import OAuthService
            fresh_token = OAuthService().refresh_if_needed(tool_name, auth)
            if fresh_token:
                settings["_oauth_access_token"] = fresh_token
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] OAuth refresh failed for '{tool_name}': {e}")
        return settings

    def _build_telemetry(self, raw_telemetry: dict) -> dict:
        """Flatten telemetry from client context into contract format."""
        from services.tool_output_utils import build_tool_telemetry
        return build_tool_telemetry(raw_telemetry)

    def _invoke_interface_tool(self, tool_name: str, tool: dict, params: dict) -> str:
        """Invoke a tool on an external interface via HTTP.

        Uses InterfaceRegistryService.execute_capability() which does:
        POST interface_host:port/execute {capability, params}
        """
        from services.invocation_tracker import track

        interface_id = tool.get("interface_id")
        if not interface_id:
            return f"[TOOL:{tool_name}] No interface_id for interface tool [/TOOL]"

        start_time = time.time()
        try:
            from services.interface_registry_service import InterfaceRegistryService
            result = InterfaceRegistryService().execute_capability(interface_id, tool_name, params)
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            track(name=tool_name, success=False, latency_ms=elapsed_ms, failure_class='external')
            return f"[TOOL:{tool_name}] Interface execution error: {e} [/TOOL]"

        elapsed_ms = int((time.time() - start_time) * 1000)

        if not isinstance(result, dict):
            track(name=tool_name, success=False, latency_ms=elapsed_ms, failure_class='external')
            return f"[TOOL:{tool_name}] {result} [/TOOL]"

        if result.get("error"):
            track(name=tool_name, success=False, latency_ms=elapsed_ms, failure_class='external')
            return f"[TOOL:{tool_name}] Error: {result['error']} [/TOOL]"

        # Format result like any other tool
        text = result.get("text", "")
        html = result.get("html")

        output_parts = []
        if text:
            output_parts.append(text)
        if html:
            output_parts.append(html)

        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Apply standard sanitization and length limits
        if len(output) > self.MAX_OUTPUT_CHARS:
            output = output[:self.MAX_OUTPUT_CHARS] + "\n...(truncated)"

        track(name=tool_name, success=True, latency_ms=elapsed_ms)
        return f"[TOOL:{tool_name}] {output} [/TOOL]"

    def invoke(self, tool_name: str, channel: str, params: dict, exchange_id: str = '') -> str:
        """
        Invoke a tool by name.

        First-party tools are called directly in-process via their handler.
        Interface tools are routed via HTTP to the paired interface.

        Returns:
            Formatted result string: [TOOL:name] ... [/TOOL]
        """
        if not self._enabled:
            return f"[TOOL:{tool_name}] Tools are disabled. [/TOOL]"

        tool = self.tools.get(tool_name)
        if not tool:
            return f"[TOOL:{tool_name}] Unknown tool: {tool_name} [/TOOL]"

        # Interface tool — route via HTTP to the interface
        if tool.get("source_type") == "interface":
            return self._invoke_interface_tool(tool_name, tool, params)

        manifest = tool["manifest"]
        if "input_schema" in manifest:
            schema_props = manifest["input_schema"].get("properties", {})
            schema_required = manifest["input_schema"].get("required", [])
            validated_params = self._validate_params_from_schema(params, schema_props, schema_required)
        else:
            validated_params = self._validate_params(params, manifest.get("parameters", {}))

        # Fetch DB-stored config
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            settings = ToolConfigService(get_shared_db_service()).get_tool_config(tool_name)
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to load tool config for '{tool_name}': {e}", exc_info=True)
            settings = {}

        # OAuth token refresh
        settings = self._refresh_oauth_token(tool_name, manifest, settings)

        # Build telemetry
        raw_telemetry = {}
        try:
            from services.client_context_service import ClientContextService
            raw_telemetry = ClientContextService().get()
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to get client telemetry for '{tool_name}': {e}", exc_info=True)
        flattened_telemetry = self._build_telemetry(raw_telemetry)

        # Resolve handler from library
        from services.tool_library_service import get_handler
        handler = get_handler(tool_name)
        if not handler:
            return f"[TOOL:{tool_name}] No handler registered for '{tool_name}' [/TOOL]"

        start_time = time.time()
        success = False
        try:
            result = handler(
                topic=channel,
                params=validated_params,
                config=settings,
                telemetry=flattened_telemetry,
            )
            success = True
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[TOOL REGISTRY] Tool '{tool_name}' failed: {e}")
            self._log_outcome(tool_name, False, channel, elapsed_ms, failure_class="internal")
            return (
                f"[TOOL:{tool_name}] Error: {str(e)[:200]} "
                f"(cost: {elapsed_ms}ms) [/TOOL]"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Extract text and html from formalized contract output
        result_text = ""
        result_html = None
        result_error = None

        if isinstance(result, dict):
            result_text = result.get("text", "")
            result_html = result.get("html")
            result_error = result.get("error")
            # Fallback: if runner didn't set "text", convert dict to readable text
            if not result_text:
                result_text = self._format_result(result)
        else:
            # Fallback for non-dict responses
            result_text = str(result) if result else ""

        # Handle error path — tool self-declares failure_class ('internal'/'external')
        if result_error:
            output = f"[TOOL:{tool_name}] Error: {result_error} (cost: {elapsed_ms}ms) [/TOOL]"
            tool_failure_class = result.get("failure_class", "internal") if isinstance(result, dict) else "internal"
            self._log_outcome(tool_name, False, channel, elapsed_ms, failure_class=tool_failure_class)
            return output

        # Clean tool output via the shared text extraction pipeline (same service used by doc processor).
        # If result_text is empty but HTML was returned, derive clean text from the HTML.
        # If result_text contains actual HTML tags, extract plain text from it.
        # Plain text passes through unchanged. The regex check avoids false positives
        # on comparison operators (e.g., "Rust < C++") in user-generated content.
        import re
        from services.text_extractor import extract_html as _extract_html
        if not result_text and result_html:
            result_text = _extract_html(result_html)
        elif result_text and re.search(r'<[a-zA-Z/]', result_text):
            result_text = _extract_html(result_text)
        if len(result_text) > self.MAX_OUTPUT_CHARS:
            result_text = result_text[:self.MAX_OUTPUT_CHARS] + "\n... (truncated)"

        # Return text response (possibly synthesized by frontal cortex)
        token_estimate = len(result_text) // 4
        output = (
            f"[TOOL:{tool_name}] {result_text}\n"
            f"(cost: {elapsed_ms}ms, ~{token_estimate} tokens)"
            f" [/TOOL]"
        )

        self._log_outcome(tool_name, success, channel, elapsed_ms)
        return output

    def _validate_params(self, params: dict, schema: dict) -> dict:
        """Validate and coerce parameters against manifest schema."""
        validated = {}
        for param_name, param_def in schema.items():
            required = param_def.get("required", False)
            default = param_def.get("default")
            param_type = param_def.get("type", "string")

            if param_name in params:
                value = params[param_name]
                try:
                    if param_type == "integer":
                        value = int(value)
                    elif param_type == "float":
                        value = float(value)
                    elif param_type == "boolean":
                        if isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes")
                        else:
                            value = bool(value)
                    elif param_type == "string":
                        value = str(value)
                except (ValueError, TypeError) as e:
                    logger.debug(f"[TOOL REGISTRY] Param coercion failed for '{param_name}': {e}", exc_info=True)
                validated[param_name] = value
            elif required:
                raise ValueError(f"Missing required parameter: {param_name}")
            elif default is not None:
                validated[param_name] = default

        return validated

    def _validate_params_from_schema(self, params: dict, properties: dict, required: list) -> dict:
        """Validate and coerce parameters against an input_schema properties dict."""
        validated = {}
        for param_name, param_def in properties.items():
            is_required = param_name in required
            default = param_def.get("default")
            param_type = param_def.get("type", "string")

            if param_name in params:
                value = params[param_name]
                try:
                    if param_type == "integer":
                        value = int(value)
                    elif param_type == "float":
                        value = float(value)
                    elif param_type == "boolean":
                        if isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes")
                        else:
                            value = bool(value)
                    elif param_type == "string":
                        value = str(value)
                except (ValueError, TypeError) as e:
                    logger.debug(f"[TOOL REGISTRY] Param coercion failed for '{param_name}': {e}", exc_info=True)
                # Enforce enum constraints
                allowed = param_def.get("enum")
                if allowed is not None and value not in allowed:
                    raise ValueError(f"Parameter '{param_name}': value {value!r} not in allowed values {allowed}")
                validated[param_name] = value
            elif is_required:
                raise ValueError(f"Missing required parameter: {param_name}")
            elif default is not None:
                validated[param_name] = default

        return validated

    def _format_result(self, result: Any) -> str:
        """Convert result dict to plain text (not JSON)."""
        from services.tool_output_utils import format_tool_result
        return format_tool_result(result)

    def _log_outcome(self, tool_name: str, success: bool, channel: str, elapsed_ms: int, failure_class: str = None, exchange_id: str = ''):
        """Log tool invocation outcome via unified tracker."""
        from services.invocation_tracker import track
        track(
            name=tool_name,
            success=success,
            latency_ms=elapsed_ms,
            channel=channel,
            exchange_id=exchange_id,
            failure_class=failure_class,
        )

    def unregister_tool(self, tool_name: str):
        """Unregister a tool from the registry (e.g., when disabling it)."""
        with self._lock:
            self.tools.pop(tool_name, None)
        logger.info(f"[TOOL REGISTRY] Unregistered tool '{tool_name}'")

    def register_interface_tool(self, interface_id: str, manifest: dict):
        """Register a tool from an external interface.

        Interface tools are stored in self.tools with source_type='interface'
        and interface_id set. They appear in prompts like any other on_demand tool.
        """
        name = manifest.get("name", "")
        if not name:
            return

        # Reject names that collide with innate skills or existing local tools
        with self._lock:
            existing = self.tools.get(name)
            if existing and existing.get("source_type") != "interface":
                logger.warning(
                    "[TOOL REGISTRY] Interface tool '%s' rejected: collides with local tool",
                    name,
                )
                return

        # Build a tool entry that looks like a local tool but with interface metadata
        tool_entry = {
            "manifest": {
                "name": name,
                "description": manifest.get("description", ""),
                "documentation": manifest.get("documentation", ""),
                "trigger": {"type": "on_demand"},
                "parameters": {
                    p["name"]: {
                        "type": p.get("type", "string"),
                        "required": p.get("required", False),
                        "description": p.get("description", ""),
                    }
                    for p in manifest.get("parameters", [])
                    if p.get("name")
                },
                "returns": manifest.get("returns", {}),
            },
            "image": None,
            "dir": None,
            "sandbox": {},
            "trust": "trusted",
            "runner_path": None,
            "source_type": "interface",
            "interface_id": interface_id,
        }

        with self._lock:
            self.tools[name] = tool_entry

        logger.info(f"[TOOL REGISTRY] Registered interface tool: {name} (interface={interface_id})")

        # Immediately kick off profile enrichment so the tool is discoverable
        # via find_tools as soon as the interface is detected — no waiting for the 6h cycle.
        _iface_name = name
        _iface_manifest = dict(tool_entry['manifest'])

        def _build_iface_profile():
            try:
                from services.tool_profile_service import ToolProfileService
                ToolProfileService().build_profile(_iface_name, _iface_manifest)
                logger.info(f"[TOOL REGISTRY] Profile built for interface tool '{_iface_name}'")
            except Exception as _e:
                logger.warning(f"[TOOL REGISTRY] Profile build failed for interface tool '{_iface_name}': {_e}")

        threading.Thread(target=_build_iface_profile, daemon=True,
                         name=f"profile-build-{name}").start()

    def remove_interface_tools(self, interface_id: str):
        """Remove all tools that belong to a specific interface.

        Cleans up both the in-memory registry and tool_capability_profiles
        so removed interfaces don't linger in find_tools results.
        """
        with self._lock:
            to_remove = [
                name for name, tool in self.tools.items()
                if tool.get("source_type") == "interface" and tool.get("interface_id") == interface_id
            ]
            for name in to_remove:
                self.tools.pop(name, None)

        # Remove profiles so find_tools stops returning them
        if to_remove:
            try:
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()
                with db.connection() as conn:
                    cursor = conn.cursor()
                    for name in to_remove:
                        # Delete vec entry first (FK on rowid)
                        row = cursor.execute(
                            "SELECT rowid FROM tool_capability_profiles WHERE tool_name = ?",
                            (name,),
                        ).fetchone()
                        if row:
                            cursor.execute(
                                "DELETE FROM tool_capability_profiles_vec WHERE rowid = ?",
                                (row[0],),
                            )
                        cursor.execute(
                            "DELETE FROM tool_capability_profiles WHERE tool_name = ?",
                            (name,),
                        )
                    cursor.close()
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] Failed to remove profiles: {e}")

            logger.info(f"[TOOL REGISTRY] Removed {len(to_remove)} interface tools for interface {interface_id}")

    # ── Public API ──────────────────────────────────────────────────

    def _is_ready(self, name: str, tool: dict) -> bool:
        """Check if a tool is ready for invocation."""
        # Library tools are always ready (deps in main requirements.txt)
        if tool.get("source_type") == "library":
            return True
        # Interface tools are ready once registered
        return True

    def _is_interface_online(self, tool: dict) -> bool:
        """Check if an interface-sourced tool's interface is online."""
        if tool.get("source_type") != "interface":
            return True  # not an interface tool — always considered online
        try:
            from services.interface_registry_service import InterfaceRegistryService
            iface = InterfaceRegistryService().get_interface(tool.get("interface_id", ""))
            return iface is not None and iface.get("status") == "online"
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Interface online check failed for '{tool.get('interface_id', '')}': {e}", exc_info=True)
            return False

    def get_tool_names(self) -> List[str]:
        """Return the names of all successfully registered tools.

        Returns:
            List of tool name strings drawn from the loaded manifest ``name``
            fields.  Order reflects insertion order (Python 3.7+ dict).
        """
        return [name for name, tool in self.tools.items()
                if self._is_ready(name, tool)
                and self._is_interface_online(tool)]

    def get_on_demand_tools(self) -> List[str]:
        """Return names of registered tools whose trigger type is ``on_demand``.

        On-demand tools are invoked explicitly by the LLM during a conversation
        turn, as opposed to webhook tools (HTTP-push). Scheduled work uses the
        ``schedule`` innate skill via ``ScheduledMessageProcessor``.

        Returns:
            List of tool name strings filtered to ``trigger.type == "on_demand"``.
        """
        return [
            name for name, tool in self.tools.items()
            if tool["manifest"].get("trigger", {}).get("type") == "on_demand"
            and self._is_ready(name, tool)
            and self._is_interface_online(tool)
        ]

    def get_ambient_tools(self) -> List[dict]:
        """
        Return on-demand tools eligible for ambient/proactive invocation.

        All on-demand tools are ambient-eligible by default. Tools opt OUT via:
          "ambient": {"enabled": false}

        Returns:
            List of {"name": str, "manifest": dict}
        """
        result = []
        for name, tool in self.tools.items():
            trigger_type = tool["manifest"].get("trigger", {}).get("type")
            if trigger_type != "on_demand":
                continue
            if not self._is_ready(name, tool):
                continue

            ambient = tool["manifest"].get("ambient", {})
            if not ambient.get("enabled", True):
                continue

            result.append({
                "name": name,
                "manifest": tool["manifest"],
            })
        return result

    def get_notification_tools(self) -> List[dict]:
        """
        Return tools that declare notification support with default_enabled=true.

        Returns:
            List of {"name": str, "manifest": dict}
        """
        result = []
        for name, tool in self.tools.items():
            notification = tool["manifest"].get("notification", {})
            if notification.get("default_enabled", False):
                result.append({"name": name, "manifest": tool["manifest"]})
        return result

    def get_tool_config_schema(self, tool_name: str) -> dict:
        """Return config_schema from a tool's manifest, or empty dict.

        Handles both dict and array formats:
        - Dict: {"field_name": {schema}} (normal)
        - Array: [{"key": "field_name", ...}] (legacy, converted to dict)
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {}

        schema = tool["manifest"].get("config_schema", {})

        # Convert array format to dict for backward compatibility
        if isinstance(schema, list):
            result = {}
            for item in schema:
                if isinstance(item, dict) and "key" in item:
                    key = item["key"]
                    result[key] = item
            return result

        return schema if isinstance(schema, dict) else {}

    def get_tool_prompt_summaries(self) -> str:
        """
        Generate SHORT prompt text for ACT prompt injection (~30 tokens per tool).
        Excludes notification tools (internal routing).
        """
        if not self._enabled or not self.tools:
            return ""

        lines = []
        for name in sorted(self.tools.keys()):
            tool = self.tools[name]
            manifest = tool["manifest"]
            trigger = manifest.get("trigger", {})

            if trigger.get("type") != "on_demand":
                continue
            if "notification" in manifest:
                continue
            if not self._is_ready(name, tool):
                continue
            if not self._is_interface_online(tool):
                continue

            desc = manifest.get("description", "")

            param_parts = []
            if "input_schema" in manifest:
                schema_props = manifest["input_schema"].get("properties", {})
                schema_required = manifest["input_schema"].get("required", [])
                for pname in schema_props:
                    param_parts.append(pname if pname in schema_required else f"{pname}?")
            else:
                params = manifest.get("parameters", {})
                for pname, pdef in params.items():
                    required = pdef.get("required", False)
                    param_parts.append(pname if required else f"{pname}?")
            param_str = ", ".join(param_parts)

            lines.append(f"- `{name}({param_str})` — {desc}")

        return "\n".join(lines)

    def get_tool_full_description(self, tool_name: str) -> Optional[dict]:
        """Get full manifest details for a tool (used by introspect)."""
        tool = self.tools.get(tool_name)
        return tool["manifest"] if tool else None

    def invoke_webhook(self, tool_name: str, webhook_body: dict, dialog_callback=None) -> dict:
        """
        Invoke a webhook-triggered tool.

        Loads state from MemoryStore, builds payload with _webhook key, runs subprocess
        interactively (so "tool" output dialogs work), persists returned state.

        Args:
            tool_name: Registered tool name (must have trigger.type == "webhook").
            webhook_body: Parsed JSON body from the webhook POST.
            dialog_callback: Optional callable(result_dict) -> str for "tool" output.
                If None, "tool" output is treated as silent.

        Returns:
            Final result dict from subprocess.

        Raises:
            ValueError: If tool not found or not a webhook trigger.
        """
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        manifest = tool["manifest"]
        trigger = manifest.get("trigger", {})
        if trigger.get("type") != "webhook":
            raise ValueError(
                f"Tool '{tool_name}' is not a webhook tool (type={trigger.get('type')})"
            )

        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            settings = ToolConfigService(get_shared_db_service()).get_tool_config(tool_name)
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to load webhook tool config for '{tool_name}': {e}", exc_info=True)
            settings = {}

        # OAuth token refresh
        settings = self._refresh_oauth_token(tool_name, manifest, settings)

        raw_telemetry = {}
        try:
            from services.client_context_service import ClientContextService
            raw_telemetry = ClientContextService().get()
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to get client telemetry for webhook '{tool_name}': {e}", exc_info=True)
        flattened_telemetry = self._build_telemetry(raw_telemetry)

        # Load persisted tool state from MemoryStore (webhook tools share state across calls)
        tool_state = {}
        state_key = f"tool_state:{tool_name}"
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            state_json = store.get(state_key)
            if state_json:
                tool_state = json.loads(state_json)
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to load persisted webhook tool state for '{tool_name}': {e}", exc_info=True)

        payload = {
            "params": {"_webhook": webhook_body, "_state": tool_state},
            "settings": settings,
            "telemetry": flattened_telemetry,
        }

        timeout = manifest.get("constraints", {}).get("timeout_seconds", 120)
        from services.tool_subprocess_service import ToolSubprocessService
        result = ToolSubprocessService().run_interactive(
            tool["runner_path"], payload,
            timeout=timeout,
            on_tool_output=dialog_callback,
        )

        # Persist returned state
        if isinstance(result, dict) and "_state" in result:
            try:
                from services.memory_client import MemoryClientService
                store = MemoryClientService.create_connection()
                store.setex(state_key, 7 * 24 * 3600, json.dumps(result.pop("_state")))
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] {tool_name}: failed to persist webhook state: {e}")

        return result

