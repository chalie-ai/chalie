"""
Cron Tool Worker — Scheduled tool execution as a background service process.

Extracted from tool_registry_service.py. Defined at module level so Python's
spawn start method can pickle it (multiprocessing constraint).

ToolRegistryService.create_cron_worker() returns an instance of this class
instead of a local closure.
"""

import json
import time
import logging

from services.tool_output_utils import (
    sanitize_tool_output,
    build_tool_telemetry,
    MAX_OUTPUT_CHARS,
)


class CronToolWorker:
    """Picklable callable for cron-triggered tool service processes."""

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

                flattened_telemetry = build_tool_telemetry(raw_telemetry)

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
                        result_text = sanitize_tool_output(result.get("text", ""))
                        if len(result_text) > MAX_OUTPUT_CHARS:
                            result_text = result_text[:MAX_OUTPUT_CHARS]
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
                result_text = sanitize_tool_output(result_text)
                if len(result_text) > MAX_OUTPUT_CHARS:
                    result_text = result_text[:MAX_OUTPUT_CHARS]

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
