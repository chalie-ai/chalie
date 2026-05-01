# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Providers — thin singleton wrapping provider resolution and LLM send.

Resolves the correct LLM provider for a given job (e.g. 'frontal-cortex-unified'),
sends messages, and returns a raw LLMResponse. No response parsing here.
"""

import json
import threading
import logging
import time

logger = logging.getLogger(__name__)

# Fraction of a provider's max_tokens that triggers continuity compaction.
# Lives here so both the boot backfill and any future tuning have one source.
COMPACTION_THRESHOLD_RATIO = 0.80


class Providers:
    """Singleton provider gateway. Resolves provider by job, sends messages."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def send(self, user_prompt, system_prompt, job='unified', tools=None, cache_prefix=True, thinking_mode=None):
        """Sync send. Returns LLMResponse."""
        if tools is None:
            tools = self._get_tools(job)
        provider = self._resolve(job)
        messages = [{"role": "user", "content": user_prompt}]
        t0 = time.monotonic()
        response = provider.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system_prompt, messages, tools, job, response, wall_ms)
        return response

    def send_messages(self, system_prompt, messages, job='unified', tools=None, cache_prefix=True, thinking_mode=None):
        """Multi-turn send with a pre-built messages array. Returns LLMResponse.

        Used by the tool loop in MessageProcessor to send growing message arrays
        across iterations without rebuilding them from scratch.

        Args:
            system_prompt: Assembled system prompt string.
            messages: Full messages array (role/content dicts).
            job: Provider job name used to resolve the LLM config.
            tools: Tool schemas. If None, Providers resolves defaults.
            cache_prefix: Whether to apply prompt prefix caching.
            thinking_mode: Native deliberation flag. None = disabled (default),
                'medium' or 'high' = provider-native thinking enabled.
        """
        if tools is None:
            tools = self._get_tools(job)
        provider = self._resolve(job)
        t0 = time.monotonic()
        response = provider.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system_prompt, messages, tools, job, response, wall_ms)
        return response

    def _log_after_call(self, system_prompt, messages, tools, job, response, wall_ms=None):
        """Write the LLM request log file. Best-effort, never raises.

        Called by both :meth:`send` and :meth:`send_messages` so every LLM
        request goes through a single logging chokepoint. Also records the
        call's latency_ms into the bound processor's MetricsAccumulator so
        the per-turn timing snapshot reflects every LLM round-trip
        (ACT iterations, exploration, compaction).
        """
        try:
            from services.message_processor import current_processor
            proc = current_processor()
            if proc is not None:
                # Prefer the gateway-level wall_ms so providers that forget to
                # populate response.latency_ms (e.g. OllamaService prior to the
                # parallel patch) still report accurately. Fall back to the
                # provider-reported value if wall_ms wasn't passed (older
                # in-tree callers).
                latency_ms = wall_ms
                if latency_ms is None:
                    latency_ms = getattr(response, 'latency_ms', None)
                if latency_ms is not None:
                    try:
                        proc._metrics.record_llm_call(latency_ms)
                    except Exception as exc:
                        logger.debug(f"[LLM LOG] record_llm_call failed: {exc}")
        except Exception as exc:
            logger.debug(f"[LLM LOG] processor lookup failed: {exc}")
        try:
            from services.llm_request_logger import log_llm_request
            from services.message_processor import current_processor
            proc = current_processor()
            caller_name = type(proc).__name__ if proc is not None else 'unknown'
            user_msg_str = self._render_messages_for_log(messages)
            log_llm_request(
                caller=caller_name,
                job=job,
                provider=getattr(response, 'provider', 'unknown'),
                model=getattr(response, 'model', 'unknown'),
                system_message=system_prompt or '',
                user_message=user_msg_str,
                tools=tools,
            )
        except Exception as e:
            logger.debug(f"[LLM LOG] Hook failed: {e}", exc_info=True)

    @staticmethod
    def _render_messages_for_log(messages):
        """Render the messages array verbatim for a log file.

        Fidelity rules:
          * list-valued ``content`` (Anthropic content-block form) is
            JSON-serialised so nothing is silently lost to ``str(list)``.
          * Single-element user messages (the MessageProcessor v2 common case)
            are written as the raw string with no ``[user]`` prefix — the spec
            says "verbatim".
          * Multi-element arrays keep the ``[role]`` prefix per entry so
            reviewers can tell the messages apart in the log.
        """
        msgs = messages or []

        def _content_to_str(content):
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)

        if len(msgs) == 1 and msgs[0].get('role') == 'user':
            return _content_to_str(msgs[0].get('content', ''))

        return '\n\n'.join(
            f"[{m.get('role', '?')}] {_content_to_str(m.get('content', ''))}"
            for m in msgs
        )

    def send_async(self, user_prompt, system_prompt, job='unified', tools=None, callback=None):
        """Fire-and-forget in daemon thread. Calls callback(LLMResponse) when done."""
        def _run():
            try:
                result = self.send(user_prompt, system_prompt, job=job, tools=tools)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"[Providers] send_async failed: {e}", exc_info=True)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def _resolve(self, job):
        """Resolve LLM provider for a job. Uses ConfigService → create_llm_service."""
        from services.config_service import ConfigService
        from services.llm_service import create_llm_service
        config = ConfigService.resolve_agent_config(job)
        return create_llm_service(config)

    def _get_tools(self, job):
        """Get native tool schemas for the calling processor's ALWAYS_AVAILABLE scope.

        Honours the lazy-load contract: DISCOVERABLE abilities are NEVER
        pre-injected here.  Falls back to an empty list when no processor is
        bound (e.g. compaction / episode-encoder paths whose own
        ALWAYS_AVAILABLE is empty by design).  The hot path passes ``tools=``
        explicitly via ``MessageProcessor.send()``; this method is the
        safety-net default.
        """
        from services.message_processor import current_processor
        proc = current_processor()
        if proc is None:
            return []
        return proc.getTools()

    def get_context_limit(self, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).get_context_limit()

    def get_compact_at(self, *, job: str = 'unified') -> 'int | None':
        """Return the persisted compact_at threshold for the active provider on *job*.

        Reads providers.compact_at from the DB row whose name matches the
        provider resolved for *job*. Returns None if the row is missing or the
        column is NULL (boot backfill should always populate it; missing value
        signals a misconfigured provider row).
        """
        try:
            from services.config_service import ConfigService
            from services.database_service import get_shared_db_service
            config = ConfigService.resolve_agent_config(job)
            provider_name = config.get('name')
            if not provider_name:
                logger.warning(
                    "[COMPACTION] get_compact_at: resolved config for job=%s has no 'name' key",
                    job,
                )
                return None
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT compact_at FROM providers WHERE name = ? AND is_active = 1",
                    (provider_name,),
                ).fetchone()
            if row is None:
                return None
            return row[0]  # may be None if column is NULL
        except Exception as exc:
            logger.warning("[COMPACTION] get_compact_at(job=%s) failed: %s", job, exc)
            return None

    def count_tokens(self, messages, system_prompt='', tools=None, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).count_tokens(messages, system_prompt, tools)
