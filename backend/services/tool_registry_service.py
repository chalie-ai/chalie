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
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance
_instance = None


class _CronToolWorker:
    """Picklable callable for cron-triggered tool service processes.

    Defined at module level so Python's spawn start method can pickle it.
    ToolRegistryService.create_cron_worker() returns an instance of this class
    instead of a local closure.
    """

    MAX_OUTPUT_CHARS = 3000

    def __init__(self, tool_config: dict):
        """Initialise a cron worker from a resolved tool configuration dict.

        Unpacks all fields required for scheduled execution: scheduling interval,
        prompt template, runner path, OAuth auth config, and per-invocation
        timeout.

        Args:
            tool_config: Resolved tool config dict produced by
                ``ToolRegistryService.get_cron_tools()``.  Required keys are
                ``name``, ``schedule``, ``prompt``, and ``dir``.
                Optional keys include ``runner_path`` and ``manifest``.
        """
        self.tool_name = tool_config["name"]
        self.schedule = tool_config["schedule"]
        self.prompt_template = tool_config["prompt"]
        self.runner_path = tool_config.get("runner_path")
        manifest = tool_config.get("manifest", {})
        self._auth = manifest.get("auth", {})
        self.tool_dir = tool_config["dir"]
        self.timeout = manifest.get("constraints", {}).get("timeout_seconds", 9)

    def __call__(self, shared_state=None):
        """Run the cron loop for this tool in a long-lived worker process.

        Sleeps for the interval parsed from ``self.schedule``, then:

        1. Backs off if the prompt-queue depth exceeds 5.
        2. Loads per-tool settings from the DB via ``ToolConfigService``.
        3. Refreshes OAuth tokens when required.
        4. Collects client telemetry and persisted tool state from MemoryStore.
        5. Dispatches the tool via subprocess.
        6. Persists any returned ``_state`` back to MemoryStore with a 7-day TTL.
        7. Routes output according to the formalized output-type contract
           (``prompt`` or ``tool``); falls back to legacy routing when no
           ``output`` field is present.

        The loop runs until a ``KeyboardInterrupt`` is received.  Any other
        exception is logged and the loop retries after a 60-second cooldown.

        Args:
            shared_state: Unused placeholder kept for multiprocessing API
                compatibility.  Defaults to ``None``.
        """
        _log = logging.getLogger(__name__)
        _log.info(f"[TOOL CRON] {self.tool_name} worker started (schedule: {self.schedule})")
        interval = self._parse_cron_interval(self.schedule)

        while True:
            try:
                time.sleep(interval)

                try:
                    from services.memory_store import get_shared_store
                    store = get_shared_store()
                    queue_depth = store.llen("prompt-queue")
                    if queue_depth > 5:
                        _log.info(
                            f"[TOOL CRON] {self.tool_name} deferred: "
                            f"prompt-queue depth={queue_depth}"
                        )
                        continue
                except Exception as e:
                    _log.debug(f"[TOOL CRON] {self.tool_name}: queue depth check failed: {e}", exc_info=True)

                try:
                    from services.tool_config_service import ToolConfigService
                    from services.database_service import get_shared_db_service
                    settings = ToolConfigService(get_shared_db_service()).get_tool_config(self.tool_name)
                except Exception as e:
                    _log.debug(f"[TOOL CRON] {self.tool_name}: failed to load tool config: {e}", exc_info=True)
                    settings = {}

                # OAuth token refresh for cron tools
                auth = self._auth
                if auth.get("type") == "oauth2" and settings.get("_oauth_access_token"):
                    try:
                        from services.oauth_service import OAuthService
                        fresh = OAuthService().refresh_if_needed(self.tool_name, auth)
                        if fresh:
                            settings["_oauth_access_token"] = fresh
                    except Exception as e:
                        _log.warning(f"[TOOL CRON] OAuth refresh failed for '{self.tool_name}': {e}")

                raw_telemetry = {}
                try:
                    from services.client_context_service import ClientContextService
                    raw_telemetry = ClientContextService().get()
                except Exception as e:
                    _log.debug(f"[TOOL CRON] {self.tool_name}: failed to get client telemetry: {e}", exc_info=True)

                # Flatten telemetry using same logic as ToolRegistryService
                loc = raw_telemetry.get("location") or {}
                loc_name = raw_telemetry.get("location_name", "")
                city, country = "", ""
                if "," in loc_name:
                    city, country = [p.strip() for p in loc_name.split(",", 1)]
                flattened_telemetry = {
                    "lat": loc.get("lat"),
                    "lon": loc.get("lon"),
                    "city": city,
                    "country": country,
                    "time": raw_telemetry.get("local_time", ""),
                    "locale": raw_telemetry.get("locale", ""),
                    "language": raw_telemetry.get("language", ""),
                }

                # Load persisted tool state from MemoryStore (survives process restarts)
                tool_state = {}
                state_key = f"tool_state:{self.tool_name}"
                old_state_key = f"tool_cron_state:{self.tool_name}"
                try:
                    from services.memory_store import get_shared_store
                    _store = get_shared_store()
                    # Migration: copy old key to new key on first access
                    if not _store.exists(state_key) and _store.exists(old_state_key):
                        old_val = _store.get(old_state_key)
                        if old_val:
                            _store.setex(state_key, 7 * 24 * 3600, old_val)
                    state_json = _store.get(state_key)
                    if state_json:
                        tool_state = json.loads(state_json)
                except Exception as e:
                    _log.debug(f"[TOOL CRON] {self.tool_name}: failed to load persisted tool state: {e}", exc_info=True)

                payload = {"params": {"_state": tool_state}, "settings": settings, "telemetry": flattened_telemetry}

                # Track interactive turns for final-turn memory storage
                dialog_turns = []

                def _on_tool_output(dialog_result):
                    from workers.digest_worker import process_tool_dialog
                    request_text = dialog_result.get("text", "")
                    response = process_tool_dialog(
                        text=request_text,
                        tool_name=self.tool_name,
                        trigger_prompt=self.prompt_template,
                    )
                    dialog_turns.append({"request": request_text, "response": response})
                    return response

                # All tools run as trusted subprocesses
                if not self.runner_path:
                    _log.warning(f"[TOOL CRON] {self.tool_name}: no runner_path, skipping")
                    continue
                from services.tool_subprocess_service import ToolSubprocessService
                result = ToolSubprocessService().run_interactive(
                    self.runner_path, payload,
                    timeout=self.timeout, on_tool_output=_on_tool_output,
                )

                # Persist returned state back to MemoryStore (7-day TTL)
                if isinstance(result, dict) and "_state" in result:
                    try:
                        from services.memory_store import get_shared_store
                        _store = get_shared_store()
                        _store.setex(state_key, 7 * 24 * 3600, json.dumps(result.pop("_state")))
                    except Exception as e:
                        _log.warning(f"[TOOL CRON] {self.tool_name}: failed to persist state: {e}")

                # Store final-turn dialog memory if interactive turns occurred
                if dialog_turns:
                    try:
                        from workers.digest_worker import store_tool_dialog_memory
                        store_tool_dialog_memory(self.tool_name, dialog_turns)
                    except Exception as e:
                        _log.warning(f"[TOOL CRON] {self.tool_name}: failed to store dialog memory: {e}")

                # --- Formalized output routing ---
                output_type = result.get("output") if isinstance(result, dict) else None

                if output_type is not None:
                    # New contract: route by output field
                    if output_type == "prompt":
                        from services.text_extractor import extract_html as _extract_html
                        result_text = result.get("text", "")
                        result_html_cron = result.get("html")
                        if not result_text and result_html_cron:
                            result_text = _extract_html(result_html_cron)
                        elif result_text and "<" in result_text:
                            result_text = _extract_html(result_text)
                        if len(result_text) > self.MAX_OUTPUT_CHARS:
                            result_text = result_text[:self.MAX_OUTPUT_CHARS]
                        if result_text:
                            full_prompt = f"{self.prompt_template}\n\n--- Tool Data ---\n{result_text}"
                            from services.prompt_queue import PromptQueue
                            from workers.digest_worker import digest_worker
                            queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
                            queue.enqueue(full_prompt, {
                                "source": f"cron_tool:{self.tool_name}",
                                "tool_name": self.tool_name,
                                "destination": "web",
                                "priority": result.get("priority", "normal"),
                            })
                    # output_type == "tool": already resolved via run_interactive callback
                    # output_type is null or anything else: silent
                    _log.info(f"[TOOL CRON] {self.tool_name} executed (output={output_type!r})")
                    continue

                # --- Legacy output routing (backward compat: no "output" field) ---
                result_text = ""
                result_html = None
                result_title = None

                if isinstance(result, dict):
                    result_text = result.get("text", "")
                    result_html = result.get("html")
                    result_title = result.get("title")
                    if not result.get("notify", True):
                        _log.debug(f"[TOOL CRON] {self.tool_name}: notify=false, skipping enqueue")
                        continue
                else:
                    result_text = str(result) if result else ""

                from services.text_extractor import extract_html as _extract_html
                if result_text and "<" in result_text:
                    result_text = _extract_html(result_text)
                if len(result_text) > self.MAX_OUTPUT_CHARS:
                    result_text = result_text[:self.MAX_OUTPUT_CHARS]

                full_prompt = f"{self.prompt_template}\n\n--- Tool Data ---\n{result_text}"

                from services.prompt_queue import PromptQueue
                from workers.digest_worker import digest_worker
                queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
                queue.enqueue(full_prompt, {
                    "source": f"cron_tool:{self.tool_name}",
                    "tool_name": self.tool_name,
                    "destination": "web",
                    "priority": result.get("priority", "normal"),
                })

                _log.info(f"[TOOL CRON] {self.tool_name} executed and enqueued (priority={result.get('priority', 'normal')})")

            except KeyboardInterrupt:
                _log.info(f"[TOOL CRON] {self.tool_name} shutting down")
                break
            except Exception as e:
                _log.error(f"[TOOL CRON] {self.tool_name} error: {e}")
                time.sleep(60)

    def _parse_cron_interval(self, schedule: str) -> int:
        parts = schedule.strip().split()
        if len(parts) >= 1 and parts[0].startswith("*/"):
            try:
                return int(parts[0][2:]) * 60
            except ValueError as e:
                logger.debug(f"[TOOL CRON] Failed to parse cron interval from '{schedule}': {e}", exc_info=True)
        return 1800

    def _format_result(self, result) -> str:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return str(result)
        lines = []
        if "results" in result and isinstance(result["results"], list):
            results = result["results"]
            if not results:
                lines.append(result.get("message", "No results found."))
            else:
                for i, r in enumerate(results, 1):
                    if isinstance(r, dict):
                        title = r.get("title", "")
                        snippet = r.get("snippet", "")
                        url = r.get("url", "")
                        lines.append(f"{i}. {title}")
                        if snippet:
                            lines.append(f"   {snippet}")
                        if url:
                            lines.append(f"   {url}")
                    else:
                        lines.append(f"{i}. {r}")
            if result.get("count") is not None and result["count"] > 0:
                lines.append(f"\n{result['count']} results returned.")
            return "\n".join(lines)
        if "content" in result and isinstance(result["content"], str):
            content = result["content"]
            if result.get("error"):
                return f"Error: {result['error']}"
            if not content:
                return "No content extracted from page."
            parts = [content]
            if result.get("truncated"):
                parts.append(f"(truncated to {result.get('char_count', '?')} chars)")
            return "\n".join(parts)
        for key, value in result.items():
            if key in ("budget_remaining",):
                continue
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: {json.dumps(value, default=str)[:500]}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


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
            from services.tool_library_service import TOOL_METADATA, ALL_TOOL_NAMES
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] Failed to import ToolLibraryService: {e}")
            return

        # Filter out DB-disabled tools
        disabled = set()
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            config_svc = ToolConfigService(get_shared_db_service())
            for name in ALL_TOOL_NAMES:
                if not config_svc.is_tool_enabled(name):
                    logger.info(f"[TOOL REGISTRY] Skipping disabled tool '{name}'")
                    disabled.add(name)
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] Could not check disabled status: {e}")

        for name in ALL_TOOL_NAMES:
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

        # Kick off profile enrichment in background
        for name in list(self.tools.keys()):
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

    def invoke(self, tool_name: str, topic: str, params: dict, exchange_id: str = '') -> str:
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
                topic=topic,
                params=validated_params,
                config=settings,
                telemetry=flattened_telemetry,
            )
            success = True
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[TOOL REGISTRY] Tool '{tool_name}' failed: {e}")
            self._log_outcome(tool_name, False, topic, elapsed_ms, failure_class="internal")
            return (
                f"[TOOL:{tool_name}] Error: {str(e)[:200]} "
                f"(cost: {elapsed_ms}ms) [/TOOL]"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Extract text and html from formalized contract output
        result_text = ""
        result_html = None
        result_title = None
        result_error = None

        if isinstance(result, dict):
            result_text = result.get("text", "")
            result_html = result.get("html")
            result_title = result.get("title")
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
            self._log_outcome(tool_name, False, topic, elapsed_ms, failure_class=tool_failure_class)
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

        self._log_outcome(tool_name, success, topic, elapsed_ms)
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

    def _format_result(self, result: Any) -> str:
        """Convert result dict to plain text (not JSON)."""
        from services.tool_output_utils import format_tool_result
        return format_tool_result(result)

    def _log_outcome(self, tool_name: str, success: bool, topic: str, elapsed_ms: int, failure_class: str = None, exchange_id: str = ''):
        """Log tool invocation outcome via unified tracker."""
        from services.invocation_tracker import track
        track(
            name=tool_name,
            success=success,
            latency_ms=elapsed_ms,
            topic=topic,
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

    def get_interface_id_for_tool(self, tool_name: str) -> str | None:
        """Return the interface_id for an interface-sourced tool, or None."""
        tool = self.tools.get(tool_name)
        if tool and tool.get("source_type") == "interface":
            return tool.get("interface_id")
        return None

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
        turn, as opposed to cron tools (scheduled) or webhook tools (HTTP-push).

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

    def get_cron_tools(self) -> List[dict]:
        """
        Return cron tools with schedule, prompt, runner path, and tool directory.

        Returns:
            List of {
                "name": str,
                "schedule": str,
                "prompt": str,
                "dir": str | None,
                "manifest": dict,
                "runner_path": str | None,
            }
        """
        cron_tools = []
        for name, tool in self.tools.items():
            trigger = tool["manifest"].get("trigger", {})
            if trigger.get("type") == "cron":
                cron_tools.append({
                    "name": name,
                    "schedule": trigger["schedule"],
                    "prompt": trigger["prompt"],
                    "dir": tool.get("dir"),
                    "manifest": tool["manifest"],
                    "runner_path": tool.get("runner_path"),
                })
        return cron_tools

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
            params = manifest.get("parameters", {})

            param_parts = []
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

    def create_cron_worker(self, tool_config: dict):
        """
        Create a worker callable for run.py service registration.

        Returns a _CronToolWorker instance (picklable with spawn start method).
        """
        return _CronToolWorker(tool_config)

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

        # Load persisted state (shared key with cron)
        tool_state = {}
        state_key = f"tool_state:{tool_name}"
        try:
            from services.memory_store import get_shared_store
            store = get_shared_store()
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
                from services.memory_store import get_shared_store
                store = get_shared_store()
                store.setex(state_key, 7 * 24 * 3600, json.dumps(result.pop("_state")))
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] {tool_name}: failed to persist webhook state: {e}")

        return result

    def _parse_cron_interval(self, schedule: str) -> int:
        """Parse simple cron expression to sleep interval in seconds. Defaults 30min."""
        parts = schedule.strip().split()
        if len(parts) >= 1 and parts[0].startswith("*/"):
            try:
                return int(parts[0][2:]) * 60
            except ValueError as e:
                logger.debug(f"[TOOL REGISTRY] Failed to parse cron interval from '{schedule}': {e}", exc_info=True)
        return 1800
