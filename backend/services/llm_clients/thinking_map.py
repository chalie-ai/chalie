# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-provider thinking-level → native-setting map.

This module is the single source of truth for how each :class:`ThinkingLevel`
member maps to the native flag each LLM platform expects.  A client imports
its platform's mapping from here; no per-client constant may exist in two
places.

**A platform with no row here inherits one, and inheriting the wrong row is
the failure this module exists to prevent.** Most clients subclass
``OpenAICompatibleClient``, which means they answer with OpenAI's vocabulary
unless they declare their own ``REASONING_EFFORTS``. That is right for a
vendor who copied OpenAI's enum and wrong for one who did not — vLLM below is
the worked example: it has no ``high`` at all, so an inherited ``high`` was a
guaranteed 400 on every request that asked for it.

Semantics (applies to every row):
  - ``NONE``  = lowest possible per provider — explicit off where the
                provider supports it, else that provider's lowest setting.
  - ``LOW``   = normally no flag sent at all, leaving the provider's default
                in place, so LOW is absent from most rows below (its meaning
                is "not present"). A provider whose default is one of its
                *higher* settings must spell LOW out instead — silence there
                buys the maximum, which inverts the scale the user chose from.
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
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ vLLM                │'none'│ low  │med.  │xhigh  │xhigh │
│ (reasoning_effort)  │      │      │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ xAI                 │ low  │ low  │med.  │high   │xhigh │
│ (reasoning_effort)  │      │      │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ DeepSeek            │'none'│ low  │high  │high   │max   │
│ (reasoning_effort)  │      │      │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Z.ai GLM            │'none'│ low  │high  │high   │max   │
│ (reasoning_effort)  │      │      │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Moonshot            │ low  │ low  │high  │high   │max   │
│ (reasoning_effort)  │      │      │      │       │      │
├─────────────────────┼──────┼───────┼──────┼───────┼──────┤
│ Mistral             │'none'│  —   │high  │high   │high  │
│ (reasoning_effort)  │      │      │      │       │      │
└─────────────────────┴──────┴───────┴──────┴───────┴──────┘

**Nine platforms deliberately have no row, and the absence is the finding.**
alibaba, baseten, fireworks, groq, llama_cpp, minimax, novita, nvidia and
openrouter are hosts that serve many models, and every one of them documents
the reasoning vocabulary as a property of the *model*, not of the host — Groq
takes ``none``/``default`` on Qwen but ``low``/``medium``/``high`` on GPT-OSS;
Fireworks accepts ``max`` broadly while one model rejects ``none`` and another
adds ``xhigh``; OpenRouter accepts the whole union and maps a request to the
nearest level its chosen model supports. A single row keyed on the platform
cannot be true for any of them, so writing one would repeat, one level up, the
exact mistake this module documents. They keep the base row — OpenAI's spelling
is the lingua franca those hosts normalise toward — and rely on the
strip-and-retry ladder for the models that diverge. Alibaba is here for a
different reason: its documented control is ``enable_thinking`` plus
``thinking_budget``, and ``reasoning_effort`` appears only on its newest models.

Notes:
  - Anthropic MAX budget = the request's max_tokens (client-side special
    case, documented in AnthropicClient._thinking_native).
  - OpenAI has no 'max' reasoning_effort; MAX maps to 'high'.
  - OpenAI compatible extra_body is sent ONLY for NONE, and only by clients
    declaring SENDS_THINKING_EXTRA_BODY (see OpenAICompatibleClient.send).
    OpenAIClient opts out: api.openai.com validates the body strictly and
    rejects the unknown top-level key.
  - Gemini 2.5 Pro cannot disable thinking; when budget 0 is rejected the
    client falls back to GEMINI_NONE_FALLBACK_BUDGET (128).
  - Codex CLI accepts minimal|low|medium|high|xhigh; 'none' does not exist
    there. MAX maps to 'high' because 'xhigh' is model-dependent.
  - vLLM is the one row here read off a server rather than a vendor doc.
  - **Five rows spell LOW out, and they are the reason LOW cannot stay a
    blanket "send nothing".** vLLM, xAI, DeepSeek, Z.ai and Moonshot all
    default their reasoning effort to the *top* of their own scale, so silence
    buys the maximum. Left absent, LOW would be the most expensive level on
    those five and the cheapest everywhere else.
  - **Four vendors have no ``medium``.** DeepSeek, Z.ai and Moonshot run
    low|high|max, and Mistral documents only none|high. MEDIUM therefore
    collapses onto HIGH there, the same way MAX collapses onto HIGH on OpenAI.
  - xAI cannot disable reasoning at all, so its NONE is its floor, ``low``.
    Moonshot does not document an off switch either and takes the same floor.
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

# Vendor-extension body param sent via OpenAI SDK's extra_body by every
# OpenAI-protocol client that declares SENDS_THINKING_EXTRA_BODY — the vendors
# reading it include Z.ai GLM and DeepSeek.  Sent ONLY for NONE.
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

# vLLM reasoning_effort values.  Read off a live server, one request per
# candidate: 'none', 'low', 'medium' and 'xhigh' answered 200; 'high' and
# 'minimal' answered 400 "Unexpected reasoning effort … Supported types are
# xhigh (default), medium, and low."  So vLLM's scale is low|medium|xhigh,
# with 'none' honoured as an explicit off even though it is not listed.
#
# Two entries differ from every other row and both are consequences of that
# vocabulary rather than preferences:
#   - HIGH cannot be 'high' — the value does not exist — so HIGH and MAX both
#     take 'xhigh', the same collapse OpenAI's row makes at its own ceiling.
#   - LOW is spelled out. vLLM's default is 'xhigh', so the usual "send no
#     flag" would hand the *most* thinking to the level named least.
#
# The vocabulary belongs to the model's reasoning parser, not to vLLM itself,
# so another model on another vLLM host can present a different scale. That is
# what the strip-and-retry ladder in OpenAICompatibleClient._call_completions
# is for: this map gets the served model right, the ladder survives the rest.
VLLM_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'none',
    ThinkingLevel.LOW: 'low',
    ThinkingLevel.MEDIUM: 'medium',
    ThinkingLevel.HIGH: 'xhigh',
    ThinkingLevel.MAX: 'xhigh',
}

# xAI reasoning_effort values: low|medium|high|xhigh, default 'high', and
# "Reasoning cannot be disabled" — 'none' and 'minimal' are not accepted, which
# is what the inherited OpenAI row was sending at NONE.  'xhigh' is newest-model
# only and older models silently treat it as 'high', so MAX is safe everywhere.
# NONE takes the floor, 'low', per this module's NONE rule.  LOW is spelled out
# because the default sits at 'high'.
# https://docs.x.ai/developers/model-capabilities/text/reasoning
XAI_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'low',
    ThinkingLevel.LOW: 'low',
    ThinkingLevel.MEDIUM: 'medium',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'xhigh',
}

# DeepSeek reasoning_effort values: low|high|max, thinking on by default at
# 'high'.  'medium' and 'xhigh' exist only as compatibility aliases that both
# resolve to 'high', so MEDIUM sends what it actually gets.  'none' disables
# thinking, and DeepSeek is one of the vendors whose extra_body
# {'thinking': {'type': 'disabled'}} says the same thing — both travel at NONE.
# LOW is spelled out because the default is 'high'.
# https://api-docs.deepseek.com/guides/thinking_mode/
DEEPSEEK_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'none',
    ThinkingLevel.LOW: 'low',
    ThinkingLevel.MEDIUM: 'high',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'max',
}

# Z.ai GLM reasoning_effort values: low|high|max, default 'max' — the highest
# default of any row here, so LOW must be spelled out or it buys the ceiling.
# 'none' and 'minimal' make the model skip thinking; 'medium' resolves to
# 'high' and 'xhigh' to 'max', so MEDIUM and MAX send their resolved values
# rather than lean on the aliases.  Z.ai is the other vendor reading the
# extra_body thinking toggle, which travels alongside at NONE.
# https://docs.z.ai/guides/capabilities/thinking-mode
ZHIPU_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'none',
    ThinkingLevel.LOW: 'low',
    ThinkingLevel.MEDIUM: 'high',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'max',
}

# Moonshot reasoning_effort values: low|high|max, default 'max'.  No off switch
# is documented, so NONE takes the floor rather than a value the vendor never
# published.  LOW is spelled out for the same reason as Z.ai — the default is
# the top of the scale.
# https://platform.moonshot.ai/docs/api/chat
MOONSHOT_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'low',
    ThinkingLevel.LOW: 'low',
    ThinkingLevel.MEDIUM: 'high',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'max',
}

# Mistral documents exactly two reasoning_effort values, 'none' and 'high', on
# the models that take the parameter at all; its native reasoning models keep
# reasoning always on and ignore it.  So every graduated level collapses onto
# 'high' and LOW stays absent — Mistral publishes no default for it to invert.
# Narrow, but every value here is one the vendor actually names, which the
# inherited row's 'medium' was not.
# https://docs.mistral.ai/studio-api/conversations/reasoning/adjustable
MISTRAL_REASONING_EFFORTS: dict[ThinkingLevel, str] = {
    ThinkingLevel.NONE: 'none',
    ThinkingLevel.MEDIUM: 'high',
    ThinkingLevel.HIGH: 'high',
    ThinkingLevel.MAX: 'high',
}
