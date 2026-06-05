# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Providers — per-mp gateway wrapping provider resolution, fit check, and send.

Resolves the active DB provider, measures the scaffolded request against the
context window (compact-first, canonical design), sends, and returns a raw
LLMResponse — or the ``OVER_CAP`` sentinel when the request must be compacted
BEFORE it can be sent.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

# Hard ceiling on any provider's reported context window. Ensures the system
# never builds a request payload exceeding this size, regardless of what the
# upstream API reports (e.g. Gemini 1M, future models with larger windows).
MAX_CONTEXT_WINDOW = 200_000

#: Sentinel returned by :meth:`Providers.send` when the FULL request reaches the
#: compaction threshold (``RS >= window - max(10% window, 8k)``). The ACT loop
#: reacts by firing both compactors FIRST (compact-first, canonical design) and
#: rebuilding a collapsed request from the DB — no API call is made for an
#: over-cap request, and there is never a partial-view ("trimmed") turn.
OVER_CAP = object()


def resolve_thinking_mode(config_thinking_mode, override, level):
    """Single precedence rule for a send's thinking level.

    1. ``config_thinking_mode`` — a config that hard-pins a level (the thinking
       ability and the two compactors pin ``"high"``). These are deliberate,
       quality-critical internals and win over everything, so a user ``medium``
       override never downgrades compaction.
    2. ``override`` — the persisted user override (``'medium'`` | ``'high'`` |
       None). When set it replaces the deliberation gate on every channel.
    3. ``level`` — the gate-computed ``mp.thinking_level`` (auto behaviour).
    """
    return config_thinking_mode or override or level


class Providers:
    """Per-mp provider gateway. Owns the bound MessageProcessor and scaffolds
    every send / size-check / resolution from it. One instance per mp."""

    def __init__(self, mp):
        self.mp = mp

    def send(self, force: bool = False):
        """Scaffold the request from self.mp, measure it, then send or signal compaction.

        Compact-first (canonical design): the FULL request — system + tools +
        watermark-bounded history + current input + act-trail — is measured. When
        ``force`` is False and the request reaches the cap (``RS >= window -
        max(10% window, 8k)``), NO API call is made: ``OVER_CAP`` is returned so the
        ACT loop fires both compactors, advances the watermark, and rebuilds a
        collapsed request from the DB on the next iteration (no partial-view turn).
        When the request fits — or ``force`` is True for an irreducible request the
        loop could not shrink — it is sent and logged. Returns LLMResponse or the
        ``OVER_CAP`` sentinel."""
        mp = self.mp
        system = mp.config.get_system_prompt(mp)
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        tools = AbilityRegistry.build_tools(mp)
        provider = self._resolve()

        from services.message_processor import _wrap_with_checkpoint  # noqa: PLC0415
        user = _wrap_with_checkpoint(mp.config.channel, mp.config.get_user_prompt(mp))
        messages = [{"role": "user", "content": user}]

        if not force and self._over_cap(system, messages, tools, provider):
            return OVER_CAP

        thinking_mode = resolve_thinking_mode(
            getattr(mp.config, "thinking_mode", None),
            getattr(mp, "thinking_override", None),
            mp.thinking_level,
        )
        t0 = time.monotonic()
        response = provider.send_messages(system, messages, True, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system, messages, tools, response, wall_ms)
        return response

    def _over_cap(self, system, messages, tools, provider) -> bool:
        """True when the FULL request reaches the compaction threshold.

        Canonical design step 3: ``RS >= window - max(10% window, 8k)`` (reserve
        ``max(10% window, 8k)`` tokens for the response — Dylan: "10% or 8k,
        whichever is highest"). RS is measured on the serialized request body, so
        tools + watermark-bounded history + current input + act-trail all count.
        Returns False when no window is known (cannot decide → send as-is)."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        window = self.get_context_limit()   # declared max_tokens, capped 200k
        if not window:
            return False
        body = provider.build_request_body(system, messages, tools)
        cap = window - max(int(0.10 * window), 8000)
        return estimate_tokens(body) >= cap

    def selected_provider(self):
        """The resolved provider instance for this mp's job (design §3.1, §6.3)."""
        return self._resolve()

    def _log_after_call(self, system_prompt, messages, tools, response, wall_ms=None):
        """Write the LLM request log file. Best-effort, never raises.

        Called from :meth:`send` — the single logging chokepoint every LLM
        request flows through. This is also THE metrics recording site (spec
        §4e): every send records, next to the latency capture and bucketed by
        the bound processor's ``config.channel``:

          * the send's token totals (folded into the processor's
            MetricsAccumulator — the loop never calls ``.accumulate()``);
          * a per-send ``requests_total`` counter; and
          * the per-channel turn counter (``dmn_turns_total`` for the dmn
            channel, ``subagent_turns_total`` for delegate channels).

        Recording once at the send means delegate / sub-processor token
        attribution is correct for free (``self.mp`` binds the right
        accumulator), and the per-turn timing snapshot reflects every LLM
        round-trip (ACT iterations, exploration, compaction).
        """
        proc = self.mp
        try:
            # Prefer the gateway-level wall_ms so providers that forget to
            # populate response.latency_ms (e.g. OllamaService prior to the
            # parallel patch) still report accurately. Fall back to the
            # provider-reported value if wall_ms wasn't passed.
            latency_ms = wall_ms if wall_ms is not None else getattr(response, 'latency_ms', None)
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
            log_llm_request(
                caller=type(proc).__name__,
                job=self.mp.config.job,
                provider=getattr(response, 'provider', 'unknown'),
                model=getattr(response, 'model', 'unknown'),
                system_message=system_prompt or '',
                user_message=self._render_messages_for_log(messages),
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

    def _resolve(self):
        """Resolve the active DB provider and wrap it as an LLM service.

        Reads everything it needs from the bound ``self.mp`` (the per-mp gateway
        already owns it — no raw params are threaded in): ``_job_name`` (for
        LoggingLLMService) from ``mp.config.job`` and ``_usage_class`` (for
        llm_call_log tagging) from ``mp.config.usage_class``. Both are injected
        into the config dict before calling create_llm_service.
        """
        from services.provider_cache_service import ProviderCacheService
        from services.llm_service import create_llm_service
        config = ProviderCacheService.get_selected_provider()
        if not config:
            providers = ProviderCacheService.get_providers()
            config = dict(next(iter(providers.values()))) if providers else {}
        config = dict(config)  # don't mutate the cached dict
        config['_job_name'] = self.mp.config.job
        config['_usage_class'] = getattr(self.mp.config, 'usage_class', None) or 'chat'
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
        return min(self._resolve().get_context_limit(), MAX_CONTEXT_WINDOW)
