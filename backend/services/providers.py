# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Providers — per-mp gateway wrapping provider resolution, window-fit, and send.

Resolves the active DB provider, fits the scaffolded request to the context
window (trim-then-compact, design §3.3), sends, and returns a raw LLMResponse.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

# Hard ceiling on any provider's reported context window. Ensures the system
# never builds a request payload exceeding this size, regardless of what the
# upstream API reports (e.g. Gemini 1M, future models with larger windows).
MAX_CONTEXT_WINDOW = 200_000


class Providers:
    """Per-mp provider gateway. Owns the bound MessageProcessor and scaffolds
    every send / size-check / resolution from it. One instance per mp."""

    def __init__(self, mp):
        self.mp = mp

    def send(self):
        """Scaffold the request from self.mp, fit it to the context window, send, log.

        Fitting is window-only (design §3.3 — trim-then-compact): the request is
        built with the full watermark-bounded history, and when it would not
        leave ``max(10% window, 8k)`` tokens free for the response the oldest
        history rows are dropped one at a time until it fits. When any row had to
        be dropped, ``mp._compaction_pending`` is set so the ACT loop compacts the
        (full) history into the next turn's checkpoint. Returns LLMResponse."""
        mp = self.mp
        system = mp.config.get_system_prompt(mp)
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        tools = AbilityRegistry.build_tools(mp)
        job = mp.config.job
        thinking_mode = getattr(mp.config, "thinking_mode", None) or mp.thinking_level
        provider = self._resolve(job, mp)

        user = self._fit_request(system, tools, provider)

        messages = [{"role": "user", "content": user}]
        t0 = time.monotonic()
        response = provider.send_messages(system, messages, True, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system, messages, tools, job, response, wall_ms, mp)
        return response

    def _fit_request(self, system, tools, provider):
        """Build the user payload, trimming oldest history until the request
        reserves response headroom (design §3.3 — trim-then-compact).

        The request must leave ``max(10% window, 8k)`` tokens free for the
        response (Dylan: "10% or 8k, whichever is highest"). While it does not,
        the oldest rendered history row is dropped (``drop_oldest_previous_message``)
        and the payload rebuilt. The drop loop is monotonic and bounded by the row
        count, so it ALWAYS terminates — an irreducible request (system + input +
        checkpoint already over budget, e.g. a deliberately tiny test window) is
        sent as-is and fails loudly at the provider rather than looping forever.

        Sets ``mp._compaction_pending`` True whenever any history row had to be
        dropped, so the ACT loop compacts the full history into the next turn's
        checkpoint and the dropped rows are not lost."""
        from services.message_processor import _wrap_with_checkpoint  # noqa: PLC0415
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        mp = self.mp

        def build():
            u = _wrap_with_checkpoint(mp.config.channel, mp.config.get_user_prompt(mp))
            body = provider.build_request_body(system, [{"role": "user", "content": u}], tools)
            return u, estimate_tokens(body)

        mp._history_drop = 0
        mp._compaction_pending = False
        user, req = build()
        window = self.get_context_limit()   # declared max_tokens, capped 200k (Step 3b)
        if not window:
            return user
        cap = window - max(int(0.10 * window), 8000)   # reserve response headroom
        if req <= cap:
            return user
        mp._compaction_pending = True
        while req > cap and mp.drop_oldest_previous_message():
            user, req = build()
        return user

    def selected_provider(self):
        """The resolved provider instance for this mp's job (design §3.1, §6.3)."""
        return self._resolve(self.mp.config.job, self.mp)

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

    def get_context_limit(self):
        """Declared context window for the active provider/model, hard-capped at
        MAX_CONTEXT_WINDOW (200k). Reads the backfilled providers.max_tokens
        (set by provider_token_limits.backfill_one = min(model window, 200k)).
        Falls back to the live provider method before backfill has run. §3.3."""
        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
        config = ProviderCacheService.get_selected_provider() or {}
        declared = config.get("max_tokens")
        if declared and int(declared) > 0:
            return min(int(declared), MAX_CONTEXT_WINDOW)
        mp = self.mp
        if mp is not None:
            return min(self._resolve(mp.config.job, mp).get_context_limit(), MAX_CONTEXT_WINDOW)
        return min(self._resolve('unified').get_context_limit(), MAX_CONTEXT_WINDOW)

    def count_tokens(self, messages, system_prompt='', tools=None, job='unified'):
        """Delegate to resolved provider."""
        return self._resolve(job).count_tokens(messages, system_prompt, tools)
