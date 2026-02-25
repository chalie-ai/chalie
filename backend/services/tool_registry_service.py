"""
Tool Registry Service — Dynamic plugin discovery, Docker image building, and dispatch.

Singleton. Discovers tools from the tools/ folder, validates manifests,
builds Docker images at startup, and dispatches invocations via containers.

IPC contract (formalized):
  Input:  base64-encoded JSON as container command arg: {"params", "settings", "telemetry"}
  Output: JSON on stdout: {"text"?, "html"?, "title"?, "error"?}
  Error:  non-zero exit code + error text on stderr

Tool output is sanitized and wrapped in [TOOL:name]...[/TOOL] markers.
Cost metadata is appended to every result.
"""

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    STRIP_PATTERNS = [
        re.compile(r'\{[^}]*\}'),
        re.compile(
            r'\b(recall|memorize|associate|introspect)\s*\(',
            re.IGNORECASE
        ),
        re.compile(r'ACTION\s*:', re.IGNORECASE),
    ]

    def __init__(self, tool_config: dict):
        self.tool_name = tool_config["name"]
        self.schedule = tool_config["schedule"]
        self.prompt_template = tool_config["prompt"]
        self.image = tool_config["image"]
        self.sandbox = tool_config.get("sandbox", {})
        manifest = tool_config.get("manifest", {})
        self._auth = manifest.get("auth", {})
        output_config = manifest.get("output", {})
        self.card_enabled = output_config.get("card", {}).get("enabled", False)
        self.card_config = output_config.get("card", {}) if self.card_enabled else None
        self.tool_dir = tool_config["dir"]
        self.timeout = manifest.get("constraints", {}).get("timeout_seconds", 9)

    def __call__(self, shared_state=None):
        _log = logging.getLogger(__name__)
        _log.info(f"[TOOL CRON] {self.tool_name} worker started (schedule: {self.schedule})")
        interval = self._parse_cron_interval(self.schedule)

        while True:
            try:
                time.sleep(interval)

                try:
                    from services.redis_client import RedisClientService
                    redis = RedisClientService.create_connection()
                    queue_depth = redis.llen("rq:queue:prompt-queue")
                    if queue_depth > 5:
                        _log.info(
                            f"[TOOL CRON] {self.tool_name} deferred: "
                            f"prompt-queue depth={queue_depth}"
                        )
                        continue
                except Exception:
                    pass

                try:
                    from services.tool_config_service import ToolConfigService
                    from services.database_service import get_shared_db_service
                    settings = ToolConfigService(get_shared_db_service()).get_tool_config(self.tool_name)
                except Exception:
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

                from services.tool_container_service import ToolContainerService
                raw_telemetry = {}
                try:
                    from services.client_context_service import ClientContextService
                    raw_telemetry = ClientContextService().get()
                except Exception:
                    pass

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

                # Load persisted tool state from Redis (survives container restarts)
                tool_state = {}
                state_key = f"tool_state:{self.tool_name}"
                old_state_key = f"tool_cron_state:{self.tool_name}"
                try:
                    from services.redis_client import RedisClientService as _RCS
                    _state_redis = _RCS.create_connection()
                    # Migration: copy old key to new key on first access
                    if not _state_redis.exists(state_key) and _state_redis.exists(old_state_key):
                        old_val = _state_redis.get(old_state_key)
                        if old_val:
                            _state_redis.setex(state_key, 7 * 24 * 3600, old_val)
                    state_json = _state_redis.get(state_key)
                    if state_json:
                        tool_state = json.loads(state_json)
                except Exception:
                    pass

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

                result = ToolContainerService().run_interactive(
                    self.image, payload, sandbox_config=self.sandbox,
                    timeout=self.timeout, on_tool_output=_on_tool_output,
                )

                # Persist returned state back to Redis (7-day TTL)
                if isinstance(result, dict) and "_state" in result:
                    try:
                        from services.redis_client import RedisClientService as _RCS
                        _state_redis = _RCS.create_connection()
                        _state_redis.setex(state_key, 7 * 24 * 3600, json.dumps(result.pop("_state")))
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
                    if output_type == "card":
                        result_html = result.get("html")
                        result_title = result.get("title")
                        if result_html:
                            try:
                                from services.card_renderer_service import CardRendererService
                                from services.output_service import OutputService
                                card_cfg = result.get("card_config") or {}
                                card_data = CardRendererService().render_tool_html(
                                    self.tool_name, result_html,
                                    result_title or card_cfg.get("title", self.tool_name), card_cfg
                                )
                                if card_data:
                                    OutputService().enqueue_card("cron", card_data, {})
                            except Exception as e:
                                _log.warning(f"[TOOL CRON] {self.tool_name}: card render failed: {e}")
                    elif output_type == "prompt":
                        result_text = self._sanitize_output(result.get("text", ""))
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

                skip_text_followup = False
                if self.card_enabled and result_html:
                    try:
                        from services.card_renderer_service import CardRendererService
                        from services.output_service import OutputService
                        card_data = CardRendererService().render_tool_html(
                            self.tool_name, result_html, result_title or self.card_config.get("title", self.tool_name),
                            self.card_config
                        )
                        if card_data:
                            OutputService().enqueue_card("cron", card_data, {})
                            # Check if synthesize is False → skip text followup
                            if not self.card_config.get("synthesize", True):
                                skip_text_followup = True
                    except Exception as e:
                        _log.warning(f"[TOOL CRON] Card render failed for {self.tool_name}: {e}")

                if skip_text_followup:
                    _log.info(f"[TOOL CRON] {self.tool_name} executed with card (response suppressed)")
                    continue
                result_text = self._sanitize_output(result_text)
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
            except ValueError:
                pass
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

    def _sanitize_output(self, text: str) -> str:
        for pattern in self.STRIP_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class ToolRegistryService:
    """
    Plugin registry for sandboxed external tools.

    Singleton — one instance created at startup, cached. Images are built once
    and reused across all invocations.
    """

    REQUIRED_MANIFEST_FIELDS = {"name", "description", "trigger", "parameters", "returns"}
    MAX_OUTPUT_CHARS = 3000

    STRIP_PATTERNS = [
        re.compile(r'\{[^}]*\}'),
        re.compile(
            r'\b(recall|memorize|associate|introspect)\s*\(',
            re.IGNORECASE
        ),
        re.compile(r'ACTION\s*:', re.IGNORECASE),
    ]

    def __new__(cls, *args, **kwargs):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(self, tools_dir: str = None):
        if self._initialized:
            return
        self._initialized = True

        if tools_dir:
            self.tools_dir = Path(tools_dir)
        else:
            self.tools_dir = Path(__file__).parent.parent / "tools"

        self.tools: Dict[str, dict] = {}  # name -> {manifest, image, dir, sandbox}
        self._enabled = True

        # Lifecycle management
        self._build_status: Dict[str, dict] = {}  # name -> {status, error}
        self._install_locks: Set[str] = set()      # names currently being installed
        self._lock = threading.Lock()              # protects build_status, install_locks, and tools mutations
        self._on_tool_registered = None  # Optional[callable] set by consumer for cron worker spawning

        try:
            from services.config_service import ConfigService
            fc_config = ConfigService.get_agent_config("frontal-cortex")
            self._enabled = fc_config.get("tools_enabled", True)
        except Exception:
            self._enabled = True

        if not self._enabled:
            logger.info("[TOOL REGISTRY] Tools disabled via kill switch (tools_enabled=false)")
            return

        self._discover_and_load()

    def _discover_and_load(self):
        """Scan tools/ folder for subdirectories with manifest.json + Dockerfile. Build images in parallel."""
        if not self.tools_dir.exists():
            logger.info(f"[TOOL REGISTRY] Tools directory not found: {self.tools_dir}")
            return

        candidates = []
        for entry in sorted(self.tools_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue

            manifest_path = entry / "manifest.json"
            dockerfile_path = entry / "Dockerfile"

            if not manifest_path.exists():
                logger.warning(f"[TOOL REGISTRY] Skipping {entry.name}: missing manifest.json")
                continue
            if not dockerfile_path.exists():
                logger.warning(f"[TOOL REGISTRY] Skipping {entry.name}: missing Dockerfile (required for sandboxing)")
                continue

            candidates.append((entry, manifest_path))

        if not candidates:
            logger.info("[TOOL REGISTRY] No tools found")
            return

        # Filter out DB-disabled tools before building
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            config_svc = ToolConfigService(get_shared_db_service())
            filtered = []
            for (entry, manifest_path) in candidates:
                try:
                    with open(manifest_path) as f:
                        tool_name = json.load(f).get("name", entry.name)
                    if not config_svc.is_tool_enabled(tool_name):
                        logger.info(f"[TOOL REGISTRY] Skipping disabled tool '{tool_name}'")
                        continue
                except Exception:
                    pass
                filtered.append((entry, manifest_path))
            candidates = filtered
        except Exception as e:
            logger.warning(f"[TOOL REGISTRY] Could not check disabled status: {e}")

        # Build images in parallel (up to 4 concurrent)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(self._load_tool, d, m): d.name for d, m in candidates}
            for future in as_completed(futures):
                dir_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"[TOOL REGISTRY] Failed to load tool '{dir_name}': {e}")
                    # Track failure in build_status so it appears with error status in list_tools()
                    with self._lock:
                        self._build_status[dir_name] = {
                            "status": "error",
                            "error": str(e)
                        }

        if self.tools:
            names = ", ".join(sorted(self.tools.keys()))
            logger.info(f"[TOOL REGISTRY] Loaded {len(self.tools)} tools: {names}")
        else:
            logger.info("[TOOL REGISTRY] No tools loaded")

    def _compute_tool_hash(self, tool_dir: Path) -> str:
        """Compute MD5 of all source files in a tool directory for staleness detection."""
        h = hashlib.md5()
        for filepath in sorted(tool_dir.rglob("*")):
            if filepath.is_file() and not any(p.startswith(".") for p in filepath.parts):
                h.update(filepath.name.encode())
                h.update(filepath.read_bytes())
        return h.hexdigest()

    def _load_tool(self, tool_dir: Path, manifest_path: Path):
        """Load, validate, and build Docker image for a single tool."""
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self._validate_manifest(manifest, tool_dir.name)

        tool_name = manifest["name"]
        version = manifest.get("version", "latest")
        image_tag = f"chalie-tool-{tool_name}:{version}"

        from services.tool_container_service import ToolContainerService
        container_svc = ToolContainerService()

        source_hash = self._compute_tool_hash(tool_dir)
        existing_hash = container_svc.get_image_source_hash(image_tag) if container_svc.image_exists(image_tag) else None

        if existing_hash == source_hash:
            logger.info(f"[TOOL REGISTRY] Image {image_tag} is up to date, skipping build")
        else:
            if existing_hash is not None:
                logger.info(f"[TOOL REGISTRY] Tool '{tool_name}' source changed, rebuilding {image_tag}...")
            else:
                logger.info(f"[TOOL REGISTRY] Building image {image_tag}...")
            if not container_svc.build_image(str(tool_dir), image_tag, source_hash=source_hash):
                raise RuntimeError(f"Failed to build image for tool '{tool_name}'")

        # Thread-safe mutation of self.tools
        with self._lock:
            self.tools[tool_name] = {
                "manifest": manifest,
                "image": image_tag,
                "dir": str(tool_dir),
                "sandbox": manifest.get("sandbox", {}),
            }

        trigger_type = manifest.get("trigger", {}).get("type", "unknown")
        logger.info(f"[TOOL REGISTRY] Loaded tool '{tool_name}' (trigger={trigger_type}, image={image_tag})")

    def _validate_manifest(self, manifest: dict, dir_name: str):
        """Validate manifest has required fields and correct structure."""
        missing = self.REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
        if missing:
            raise ValueError(f"Manifest missing required fields: {missing}")

        trigger = manifest.get("trigger", {})
        if "type" not in trigger:
            raise ValueError("trigger must have a 'type' field")
        if trigger["type"] not in ("on_demand", "cron", "webhook"):
            raise ValueError(f"Unknown trigger type: {trigger['type']}")

        if trigger["type"] == "cron":
            if "schedule" not in trigger:
                raise ValueError("Cron trigger must have a 'schedule' field")
            if "prompt" not in trigger:
                raise ValueError("Cron trigger must have a 'prompt' field")

        # Validate optional auth block
        auth = manifest.get("auth")
        if auth:
            if auth.get("type") not in ("oauth2",):
                raise ValueError(f"Unknown auth type: {auth.get('type')}")
            if auth["type"] == "oauth2":
                required_auth = {"authorization_url", "token_url", "scopes"}
                missing_auth = required_auth - set(auth.keys())
                if missing_auth:
                    raise ValueError(f"OAuth2 auth missing fields: {missing_auth}")

        if manifest.get("name") != dir_name:
            logger.warning(
                f"[TOOL REGISTRY] Tool name '{manifest.get('name')}' "
                f"doesn't match directory '{dir_name}' — using manifest name"
            )

        # Warn (not error) if documentation field is missing
        if not manifest.get('documentation'):
            logger.warning(
                f"[TOOL REGISTRY] Tool '{manifest.get('name', dir_name)}' has no 'documentation' field. "
                f"Add 'documentation' for richer capability profiles."
            )

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
        loc = raw_telemetry.get("location") or {}
        loc_name = raw_telemetry.get("location_name", "")
        city, country = "", ""
        if "," in loc_name:
            city, country = [p.strip() for p in loc_name.split(",", 1)]

        device = raw_telemetry.get("device") or {}

        result = {
            "lat": loc.get("lat"),
            "lon": loc.get("lon"),
            "location_name": raw_telemetry.get("location_name", ""),
            "city": city,
            "country": country,
            "time": raw_telemetry.get("local_time", ""),
            "locale": raw_telemetry.get("locale", ""),
            "language": raw_telemetry.get("language", ""),
        }

        # Device context — so tools can tailor output to user's device
        if device_class := device.get("class"):
            result["device_class"] = device_class
        if platform := device.get("platform"):
            result["platform"] = platform
        if "pwa" in device:
            result["pwa"] = device["pwa"]

        return result

    def invoke(self, tool_name: str, topic: str, params: dict) -> str:
        """
        Invoke a tool by name via Docker container.

        Validates params, fetches DB config, runs container, sanitizes output,
        appends cost metadata, logs outcome to procedural memory.

        Returns:
            Formatted result string: [TOOL:name] ... [/TOOL]
        """
        if not self._enabled:
            return f"[TOOL:{tool_name}] Tools are disabled. [/TOOL]"

        tool = self.tools.get(tool_name)
        if not tool:
            return f"[TOOL:{tool_name}] Unknown tool: {tool_name} [/TOOL]"

        manifest = tool["manifest"]
        validated_params = self._validate_params(params, manifest.get("parameters", {}))

        # Fetch DB-stored config and inject into payload (renamed to 'settings')
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            settings = ToolConfigService(get_shared_db_service()).get_tool_config(tool_name)
        except Exception:
            settings = {}

        # OAuth token refresh — if manifest declares auth.type == "oauth2"
        settings = self._refresh_oauth_token(tool_name, manifest, settings)

        # Get raw telemetry and flatten for tool payload
        raw_telemetry = {}
        try:
            from services.client_context_service import ClientContextService
            raw_telemetry = ClientContextService().get()
        except Exception:
            pass

        flattened_telemetry = self._build_telemetry(raw_telemetry)

        # Formalized contract: three keys only, no topic at top level
        payload = {
            "params": validated_params,
            "settings": settings,
            "telemetry": flattened_telemetry,
        }
        timeout = manifest.get("constraints", {}).get("timeout_seconds", 9)

        start_time = time.time()
        success = False
        try:
            from services.tool_container_service import ToolContainerService
            result = ToolContainerService().run(
                tool["image"],
                payload,
                sandbox_config=tool.get("sandbox", {}),
                timeout=timeout,
            )
            success = True
        except TimeoutError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[TOOL REGISTRY] Tool '{tool_name}' timed out: {e}")
            self._log_outcome(tool_name, False, topic, elapsed_ms)
            return f"[TOOL:{tool_name}] Error: timed out after {timeout}s (cost: {elapsed_ms}ms) [/TOOL]"
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[TOOL REGISTRY] Tool '{tool_name}' failed: {e}")
            self._log_outcome(tool_name, False, topic, elapsed_ms)
            return (
                f"[TOOL:{tool_name}] Error: {str(e)[:200]} "
                f"(cost: {elapsed_ms}ms) [/TOOL]"
            )

        # Cache raw result for card rendering (5-min TTL)
        if isinstance(result, dict):
            try:
                from services.redis_client import RedisClientService
                redis = RedisClientService.create_connection()
                redis.rpush(
                    f"tool_raw_cache:{topic}",
                    json.dumps({"tool": tool_name, "data": result})
                )
                redis.expire(f"tool_raw_cache:{topic}", 300)
            except Exception as e:
                logger.debug(f"[TOOL REGISTRY] Raw result cache failed for {tool_name}: {e}")

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
        else:
            # Fallback for non-dict responses
            result_text = str(result) if result else ""

        # Handle error path
        if result_error:
            output = f"[TOOL:{tool_name}] Error: {result_error} (cost: {elapsed_ms}ms) [/TOOL]"
            self._log_outcome(tool_name, False, topic, elapsed_ms)
            return output

        # Sanitize text result
        result_text = self._sanitize_output(result_text)
        if len(result_text) > self.MAX_OUTPUT_CHARS:
            result_text = result_text[:self.MAX_OUTPUT_CHARS] + "\n... (truncated)"

        # Read manifest output config
        output_config = manifest.get("output", {})
        synthesize = output_config.get("synthesize", True)
        card_config = output_config.get("card", {})
        card_enabled = card_config.get("enabled", False)

        # If card is enabled, render and enqueue it.
        # Path A: tool returned inline HTML → render_tool_html()
        # Path B: tool returned a dict with no HTML → template-based render()
        if card_enabled and result_html:
            try:
                from services.card_renderer_service import CardRendererService
                from services.output_service import OutputService
                card_data = CardRendererService().render_tool_html(
                    tool_name, result_html, result_title or card_config.get("title", tool_name), card_config
                )
                if card_data:
                    OutputService().enqueue_card(topic, card_data, {})
                    # If synthesize is false, suppress text response
                    if not synthesize:
                        output = f"[TOOL:{tool_name}] (card displayed, cost: {elapsed_ms}ms) [/TOOL]"
                        self._log_outcome(tool_name, success, topic, elapsed_ms)
                        return output
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] Card render failed for {tool_name}: {e}")
        elif card_enabled and not result_html and isinstance(result, dict):
            # Template-based rendering: loads card/template.html + card/styles.css
            # and compiles Mustache against the raw result dict.
            try:
                from services.card_renderer_service import CardRendererService
                from services.output_service import OutputService
                card_data = CardRendererService().render(
                    tool_name, result, card_config, tool["dir"]
                )
                if card_data:
                    OutputService().enqueue_card(topic, card_data, {})
                    if not synthesize:
                        output = f"[TOOL:{tool_name}] (card displayed, cost: {elapsed_ms}ms) [/TOOL]"
                        self._log_outcome(tool_name, success, topic, elapsed_ms)
                        return output
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] Template card render failed for {tool_name}: {e}")

        # Otherwise, return text response (possibly synthesized by frontal cortex)
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
                except (ValueError, TypeError):
                    pass
                validated[param_name] = value
            elif required:
                raise ValueError(f"Missing required parameter: {param_name}")
            elif default is not None:
                validated[param_name] = default

        return validated

    def _format_result(self, result: Any) -> str:
        """Convert result dict to plain text (not JSON)."""
        if isinstance(result, str):
            return result

        if not isinstance(result, dict):
            return str(result)

        lines = []

        if "results" in result and isinstance(result["results"], list):
            results = result["results"]
            if not results:
                msg = result.get("message", "No results found.")
                lines.append(msg)
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

    def _sanitize_output(self, text: str) -> str:
        """Strip action-like patterns from tool output."""
        for pattern in self.STRIP_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _log_outcome(self, tool_name: str, success: bool, topic: str, elapsed_ms: int):
        """Log tool invocation outcome to procedural memory."""
        try:
            from services.procedural_memory_service import ProceduralMemoryService
            from services.database_service import get_shared_db_service
            db_service = get_shared_db_service()
            service = ProceduralMemoryService(db_service)
            reward = 0.3 if success else -0.2
            service.record_action_outcome(tool_name, success, reward, topic)
        except Exception as e:
            logger.debug(f"[TOOL REGISTRY] Failed to log outcome: {e}")

    def register_tool_async(self, tool_dir: Path) -> bool:
        """
        Register a tool from a directory asynchronously.

        Spawns a background thread to load and build the Docker image.
        Returns False if the tool is already being installed.
        Returns True if installation started successfully.

        Args:
            tool_dir: Path to tool directory (must contain manifest.json)

        Returns:
            bool: True if build thread started, False if already installing
        """
        manifest_path = tool_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"manifest.json not found in {tool_dir}")

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            tool_name = manifest.get("name")
            if not tool_name:
                raise ValueError("Manifest missing 'name' field")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to read manifest: {e}")

        with self._lock:
            # Check if already installing
            if tool_name in self._install_locks:
                return False

            # Mark as installing
            self._install_locks.add(tool_name)
            self._build_status[tool_name] = {"status": "building", "error": None}

        # Spawn build worker in background thread
        thread = threading.Thread(
            target=self._build_worker,
            args=(tool_dir, tool_name),
            daemon=True,
        )
        thread.start()
        logger.info(f"[TOOL REGISTRY] Started async build for '{tool_name}'")
        return True

    def _build_worker(self, tool_dir: Path, tool_name: str):
        """
        Private worker function for background tool building (runs in thread).

        Attempts to load and build the tool. Updates _build_status and clears _install_locks.
        """
        try:
            # Check if directory still exists (might have been disabled)
            if not tool_dir.exists():
                logger.info(f"[TOOL REGISTRY] Tool directory removed during build: {tool_name}")
                with self._lock:
                    self._install_locks.discard(tool_name)
                return

            manifest_path = tool_dir / "manifest.json"
            # Build the tool (this calls _load_tool internally)
            self._load_tool(tool_dir, manifest_path)

            # Success: clear build status
            with self._lock:
                self._build_status.pop(tool_name, None)
                self._install_locks.discard(tool_name)

            logger.info(f"[TOOL REGISTRY] Async build completed for '{tool_name}'")

            # Build capability profile for the newly registered tool
            try:
                from services.tool_profile_service import ToolProfileService
                profile_service = ToolProfileService()
                manifest = self.tools[tool_name]['manifest']
                profile_service.build_profile(tool_name, manifest)
                logger.info(f"[TOOL REGISTRY] Built capability profile for {tool_name}")
            except Exception as profile_err:
                logger.warning(f"[TOOL REGISTRY] Profile build failed for {tool_name}: {profile_err}")

            # Notify consumer so it can spawn cron workers for newly registered tools
            if self._on_tool_registered:
                try:
                    self._on_tool_registered(tool_name)
                except Exception as cb_err:
                    logger.warning(f"[TOOL REGISTRY] on_tool_registered callback failed: {cb_err}")

        except Exception as e:
            logger.error(f"[TOOL REGISTRY] Async build failed for '{tool_name}': {e}")
            with self._lock:
                self._build_status[tool_name] = {
                    "status": "error",
                    "error": str(e)
                }
                self._install_locks.discard(tool_name)

    def unregister_tool(self, tool_name: str):
        """
        Unregister a tool from the registry (e.g., when disabling it).

        Removes the tool from self.tools, clears build status, and removes any install locks.

        Args:
            tool_name: Name of the tool to unregister
        """
        with self._lock:
            self.tools.pop(tool_name, None)
            self._build_status.pop(tool_name, None)
            self._install_locks.discard(tool_name)
        logger.info(f"[TOOL REGISTRY] Unregistered tool '{tool_name}'")

    def get_all_build_statuses(self) -> dict:
        """
        Get a shallow copy of all tool build statuses.

        Returns:
            dict: {tool_name: {"status": str, "error": str|None}, ...}
        """
        with self._lock:
            return dict(self._build_status)

    # ── Public API ──────────────────────────────────────────────────

    def set_on_tool_registered(self, callback):
        """Called by consumer to register a hook for post-build cron worker spawning."""
        self._on_tool_registered = callback

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_on_demand_tools(self) -> List[str]:
        return [
            name for name, tool in self.tools.items()
            if tool["manifest"].get("trigger", {}).get("type") == "on_demand"
        ]

    def get_cron_tools(self) -> List[dict]:
        """
        Return cron tools with schedule, prompt, image, sandbox config, and tool directory.

        Returns:
            List of {
                "name": str,
                "schedule": str,
                "prompt": str,
                "image": str,
                "sandbox": dict,
                "dir": str,
                "manifest": dict,
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
                    "image": tool["image"],
                    "sandbox": tool.get("sandbox", {}),
                    "dir": tool["dir"],
                    "manifest": tool["manifest"],
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
            manifest = self.tools[name]["manifest"]
            trigger = manifest.get("trigger", {})

            if trigger.get("type") != "on_demand":
                continue
            if "notification" in manifest:
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
        Create a worker callable for consumer.py service registration.

        Returns a _CronToolWorker instance (picklable with spawn start method).
        """
        return _CronToolWorker(tool_config)

    def invoke_webhook(self, tool_name: str, webhook_body: dict, dialog_callback=None) -> dict:
        """
        Invoke a webhook-triggered tool.

        Loads state from Redis, builds payload with _webhook key, runs container
        interactively (so "tool" output dialogs work), persists returned state.

        Args:
            tool_name: Registered tool name (must have trigger.type == "webhook").
            webhook_body: Parsed JSON body from the webhook POST.
            dialog_callback: Optional callable(result_dict) -> str for "tool" output.
                If None, "tool" output is treated as silent.

        Returns:
            Final result dict from container.

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
        except Exception:
            settings = {}

        # OAuth token refresh
        settings = self._refresh_oauth_token(tool_name, manifest, settings)

        raw_telemetry = {}
        try:
            from services.client_context_service import ClientContextService
            raw_telemetry = ClientContextService().get()
        except Exception:
            pass
        flattened_telemetry = self._build_telemetry(raw_telemetry)

        # Load persisted state (shared key with cron)
        tool_state = {}
        state_key = f"tool_state:{tool_name}"
        try:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
            state_json = redis.get(state_key)
            if state_json:
                tool_state = json.loads(state_json)
        except Exception:
            pass

        payload = {
            "params": {"_webhook": webhook_body, "_state": tool_state},
            "settings": settings,
            "telemetry": flattened_telemetry,
        }

        timeout = manifest.get("constraints", {}).get("timeout_seconds", 120)
        from services.tool_container_service import ToolContainerService
        result = ToolContainerService().run_interactive(
            tool["image"], payload,
            sandbox_config=tool.get("sandbox", {}),
            timeout=timeout,
            on_tool_output=dialog_callback,
        )

        # Persist returned state
        if isinstance(result, dict) and "_state" in result:
            try:
                from services.redis_client import RedisClientService
                redis = RedisClientService.create_connection()
                redis.setex(state_key, 7 * 24 * 3600, json.dumps(result.pop("_state")))
            except Exception as e:
                logger.warning(f"[TOOL REGISTRY] {tool_name}: failed to persist webhook state: {e}")

        return result

    def _parse_cron_interval(self, schedule: str) -> int:
        """Parse simple cron expression to sleep interval in seconds. Defaults 30min."""
        parts = schedule.strip().split()
        if len(parts) >= 1 and parts[0].startswith("*/"):
            try:
                return int(parts[0][2:]) * 60
            except ValueError:
                pass
        return 1800
