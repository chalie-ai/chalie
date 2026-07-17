# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-provider thinking-level → native-setting map.

This module is the single source of truth for how each :class:`ThinkingLevel`
member maps to the native flag each LLM platform expects.  All five thin
clients import their platform's mapping from here; no per-client constant
may exist in two places.

Semantics (applies to every row):
  - ``NONE``  = lowest possible per provider — explicit off where the
                provider supports it, else that provider's lowest setting.
  - ``LOW``   = no flag sent at all — leaves the provider's default in place.
                LOW is absent from every row below (its meaning is "not present").
  - ``MEDIUM``/``HIGH``/``MAX`` = escalating native settings.

Platform × Level matrix (values shown are what is sent to the provider):

┌─────────────────────┬──────┬───────┬──────┬───────┬──────┐
│ Platform            │ NONE │  LOW  │MED.  │ HIGH  │  MAX │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Anthropic           │disabled│ —  │4096  │16384  │req's │
│                     │      │       │      │       │max tk│
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ OpenAI              │ 'none'│ —   │medi. │ high  │ high │
│ (reasoning_effort)  │      │       │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ OpenAI compatible   │extra_│ —    │med.  │high   │high  │
│ (extra_body)        │body  │       │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Gemini              │  0   │  —   │ 4096 │16384  │32768 │
│ (thinking_budget)   │      │       │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Ollama              │False │  —   │ True │ True  │True  │
│ (think)             │      │       │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Codex CLI           │mini. │  —   │med.  │high   │high  │
│ (reasoning_effort)  │      │       │      │       │      │
└─────────────────────┴──────┴───────┴──────┴───────┴──────┘

Notes:
  - Anthropic MAX budget = the request's max_tokens (client-side special
    case, documented in AnthropicClient._thinking_native).
  - OpenAI has no 'max' reasoning_effort; MAX maps to 'high'.
  - OpenAI compatible extra_body is sent ONLY on the openai_compatible
    platform and ONLY for NONE (see OpenAIClient.send).
  - Gemini 2.5 Pro cannot disable thinking; when budget 0 is rejected the
    client falls back to GEMINI_NONE_FALLBACK_BUDGET (128).
  - Codex CLI accepts minimal|low|medium|high|xhigh; 'none' does not exist
    there. MAX maps to 'high' because 'xhigh' is model-dependent.
"""

from __future__ import annotations

from configs.enums.thinking_level import ThinkingLevel

# Anthropic: NONE sends an explicit disabled thinking block.
ANTHROPIC_NONE_THINKING: dict[str, object] = {'type': 'disabled'}

# Anthropic {'type':'enabled','budget_tokens':N} budgets for the graduated
# levels.  MAX is a client-side special case (budget = request max_tokens) —
# see AnthropicClient._thinking_native.
ANTHROPIC_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.MEDIUM: 4096,
    ThinkingLevel.HIGH: 16384,
}

# OpenAI reasoning_effort values.  OpenAI has no 'max'; MAX maps to 'high'.
OPENAI_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'none',
    ThinkingLevel.MEDIUM: 'medium',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'high',
}

# When a model rejects 'none' on OpenAI, retry with this effort.
OPENAI_NONE_FALLBACK_EFFORT = 'minimal'

# Vendor-extension body param sent via OpenAI SDK's extra_body for
# openai_compatible platforms (Z.ai GLM, DeepSeek v4).  Sent ONLY for NONE.
OPENAI_COMPATIBLE_NONE_BODY: dict[str, object] = {
    'thinking': {'type': 'disabled'},
}

# Gemini thinking_budget values (integers passed to ThinkingConfig).
GEMINI_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.NONE: 0,
    ThinkingLevel.MEDIUM: 4096,
    ThinkingLevel.HIGH: 16384,
    ThinkingLevel.MAX: 32768,
}

# When budget 0 is rejected (Gemini 2.5 Pro cannot disable thinking), retry
# with this budget — Gemini's documented floor.
GEMINI_NONE_FALLBACK_BUDGET = 128

# Ollama think flag (boolean).  LOW absent = no flag.
OLLAMA_THINK: dict[ThinkingLevel, bool] = {
    ThinkingLevel.NONE: False,
    ThinkingLevel.MEDIUM: True,
    ThinkingLevel.HIGH: True,
    ThinkingLevel.MAX: True,
}

# Codex CLI model_reasoning_effort config values.  'none' does not exist
# there; NONE maps to 'minimal'.  MAX maps to 'high' because 'xhigh' is
# model-dependent.
CODEX_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'minimal',
    ThinkingLevel.MEDIUM: 'medium',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'high',
}
