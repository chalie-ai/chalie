# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Providers — thin singleton wrapping provider resolution and LLM send.

Resolves the active DB provider, sends messages, returns a raw LLMResponse.
"""

import json
import threading
import logging
import time

logger = logging.getLogger(__name__)

# Hard ceiling on any provider's reported context window. Ensures the system
# never builds a request payload exceeding this size, regardless of what the
# upstream API reports (e.g. Gemini 1M, future models with larger windows).
MAX_CONTEXT_WINDOW = 200_000


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

    def send(self, user_prompt, system_prompt, job='unified', tools=None, cache_prefix=True, thinking_mode=None, mp=None):
        """Sync send. Returns LLMResponse.

        ``mp`` is the invoking MessageProcessor (the parent), threaded explicitly
        so metrics/token attribution binds to the right accumulator. None from
        background callers — metrics simply no-op, as before.
        """
        if tools is None:
            tools = self._get_tools(mp)
        provider = self._resolve(job, mp)
        messages = [{"role": "user", "content": user_prompt}]
        t0 = time.monotonic()
        response = provider.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system_prompt, messages, tools, job, response, wall_ms, mp)
        return response

    def send_messages(self, system_prompt, messages, job='unified', tools=None, cache_prefix=True, thinking_mode=None, mp=None):
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
            tools = self._get_tools(mp)
        provider = self._resolve(job, mp)
        t0 = time.monotonic()
        response = provider.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system_prompt, messages, tools, job, response, wall_ms, mp)
        return response

    def _log_after_call(self, system_prompt, messages, tools, job, response, wall_ms=None, mp=None):
        """Write the LLM request log file. Best-effort, never raises.

        Called by both :meth:`send` and :meth:`send_messages` so every LLM
        request goes through a single logging chokepoint. This is also THE
        metrics recording site (spec §4e): every send records, next to the
        existing latency capture and bucketed by the bound processor's
        ``config.channel``:

          * the send's token totals (folded into the processor's
            MetricsAccumulator — the loop never calls ``.accumulate()``);
          * a per-send ``requests_total`` counter; and
          * the per-channel turn counter (``dmn_turns_total`` for the dmn
            channel, ``subagent_turns_total`` for delegate channels).

        Recording once at the send means delegate / sub-processor token
        attribution is correct for free (the threaded ``mp`` binds the
        right accumulator), and the per-turn timing snapshot reflects every
        LLM round-trip (ACT iterations, exploration, compaction).
        """
        try:
            proc = mp
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
                # Token totals fold into the processor's accumulator at the send
                # (§4e) — the loop no longer calls .accumulate().
                try:
                    proc._metrics.accumulate(response)
                except Exception as exc:
                    logger.debug(f"[LLM LOG] accumulate failed: {exc}")
                # Per-send / per-channel counters (§4e).
                self._record_send_counters(proc)
        except Exception as exc:
            logger.debug(f"[LLM LOG] processor lookup failed: {exc}")
        try:
            from services.llm_request_logger import log_llm_request
            proc = mp
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
    def _record_send_counters(proc):
        """Record the per-send, per-channel turn counters (spec §4e).

        Every send increments ``requests_total``. The channel-specific turn
        counter is selected from the bound processor's ``config.channel``:
        ``dmn`` → ``dmn_turns_total``; a ``delegate:*`` channel →
        ``subagent_turns_total``. Best-effort: a metrics failure must never
        break the send path.
        """
        try:
            channel = getattr(getattr(proc, 'config', None), 'channel', '') or ''
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            if channel == 'dmn':
                m.record_counter('dmn_turns_total')
            elif channel.startswith('delegate:'):
                m.record_counter('subagent_turns_total')
        except Exception as exc:
            logger.debug(f"[LLM LOG] send-counter record failed: {exc}")

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

    def _resolve(self, job, mp=None):
        """Resolve the active DB provider and wrap it as an LLM service.

        Injects _job_name (for LoggingLLMService) and _usage_class (for
        llm_call_log tagging) into the config dict before calling
        create_llm_service.
        """
        from services.provider_cache_service import ProviderCacheService
        from services.llm_service import create_llm_service
        config = ProviderCacheService.get_selected_provider()
        if not config:
            providers = ProviderCacheService.get_providers()
            if providers:
                config = dict(next(iter(providers.values())))
            else:
                config = {}
        config = dict(config)  # don't mutate the cached dict
        config['_job_name'] = job
        proc = mp
        if proc is not None:
            proc_config = getattr(proc, 'config', None)
            usage_class = getattr(proc_config, 'usage_class', None) or 'chat'
            config['_usage_class'] = usage_class
        return create_llm_service(config)

    def _get_tools(self, mp=None):
        """Get native tool schemas for the calling processor's tool scope.

        Honours the lazy-load contract: DISCOVERABLE abilities are NEVER
        pre-injected here.  Falls back to an empty list when no processor is
        passed (e.g. compaction / episode-encoder paths whose own scope is empty
        by design).  The hot path passes ``tools=`` explicitly from the flat ACT
        loop; this method is only the safety-net default.
        """
        from abilities._registry import AbilityRegistry
        if mp is None:
            return []
        return AbilityRegistry.build_tools(mp)

    def get_context_limit(self, job='unified'):
        """Delegate to resolved provider, capped to MAX_CONTEXT_WINDOW."""
        return min(self._resolve(job).get_context_limit(), MAX_CONTEXT_WINDOW)

    def calculate(
        self,
        system_prompt: str,
        user_body: str,
        tools: list,
        job: str = 'unified',
        mp=None,
    ) -> float:
        """Return what fraction of the context window this request would use.

        Builds the real request body (same serialisation path as send_messages),
        counts tokens, divides by the provider's context window.

        Returns 0.0 on any error so the ACT loop safely skips compaction and
        proceeds to send_messages() — the same safe behaviour as the old
        estimate_payload_tokens() failure path.

        Returns 0.0 when the provider reports a zero or missing context limit
        to avoid zero-division.

        Spec §4b / E1–E4.
        """
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        try:
            provider = self._resolve(job, mp)
            body = provider.build_request_body(
                system_prompt,
                [{'role': 'user', 'content': user_body}],
                tools,
            )
            tokens = estimate_tokens(body)
            max_tokens = provider.get_context_limit()
            if not max_tokens:
                return 0.0
            return tokens / max_tokens
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Provider.calculate] failed: %s", exc)
            return 0.0

    def count_tokens(self, messages, system_prompt='', tools=None, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).count_tokens(messages, system_prompt, tools)
