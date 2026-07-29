"""Context-window defaults — the one owner of "what window do we assume?".

Every client's ``get_context_limit`` prefers an authoritative answer: Anthropic's
published figure, Gemini's ``input_token_limit``, Ollama's ``/api/show``, codex's
local model cache, a size field a self-hosted OpenAI server volunteers. None of
those cover every model, and a missing window is not a harmless gap —
``ProviderDbService.pin_context_window`` raises on it, so an unmeasured provider
fails every single turn.

So when a provider **answers but volunteers no size**, the window comes from
here instead of from nowhere. The distinction that matters:

  provider answered, no size  → this default (it is live; size it and move on)
  provider unreachable/errored → None (leave the column NULL, re-probe next send)

That second case stays None deliberately: a briefly-down host must not
permanently stamp a fabricated window onto its row.

Consumed by: services.llm_clients.{openai,ollama,gemini,codex_cli}.
Not used by anthropic — it publishes a real figure for every model it serves.
"""

from __future__ import annotations

# Modern large-context families run 128k or more, so an unrecognised member of
# one is far better served by 128k than by the conservative floor. Matched as a
# substring, not a prefix, so vendor-namespaced slugs ('zai-org/GLM-4.6',
# 'anthropic/claude-3-5-sonnet') still resolve to their family.
LARGE_WINDOW_FAMILIES: tuple[str, ...] = ('glm', 'claude', 'gpt', 'gemini', 'qwen3', 'llama-4')
DEFAULT_LARGE_WINDOW = 128_000

# Anything unrecognised. Deliberately conservative: under-promising a window
# costs some compaction headroom, while over-promising one gets the payload
# rejected mid-turn by a provider that never had the room.
DEFAULT_WINDOW = 32_000


def default_window_for_model(model: "str | None") -> int:
    """A usable window for a live model that reported no size of its own.

    A missing slug is a legitimate input, not a caller error — a client may hold
    no model name at all — and it lands on the conservative default like any
    other unrecognised one.
    """
    slug = (model or '').lower()
    if any(family in slug for family in LARGE_WINDOW_FAMILIES):
        return DEFAULT_LARGE_WINDOW
    return DEFAULT_WINDOW
