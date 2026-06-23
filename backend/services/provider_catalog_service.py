"""Curated provider catalog — presets for the reactive provider-setup wizard.

Each preset maps a popular AI provider onto one of Chalie's five existing
platforms (``ollama`` / ``openai`` / ``anthropic`` / ``gemini`` /
``openai_compatible``) plus its base URL, so the wizard can pre-fill the host and
reuse the live model-fetch path (``POST /providers/list-models``) unchanged.

A preset is NEVER a runtime platform of its own — ``platform`` is always a real
Chalie platform, and the provider is created through the ordinary create path
with no catalog special-casing. That is why this module is a plain in-process
constant: no file I/O, no path resolution.
"""

from __future__ import annotations

CURATED_PROVIDERS: list[dict[str, object]] = [
    {"id": "ollama", "name": "Ollama (local)", "platform": "ollama",
     "host": "http://localhost:11434", "needs_key": False},
    {"id": "openai", "name": "OpenAI", "platform": "openai",
     "host": "", "needs_key": True},
    {"id": "anthropic", "name": "Anthropic", "platform": "anthropic",
     "host": "", "needs_key": True},
    {"id": "gemini", "name": "Google Gemini", "platform": "gemini",
     "host": "", "needs_key": True},
    {"id": "deepseek", "name": "DeepSeek", "platform": "openai_compatible",
     "host": "https://api.deepseek.com", "needs_key": True},
    {"id": "xai", "name": "xAI (Grok)", "platform": "openai_compatible",
     "host": "https://api.x.ai/v1", "needs_key": True},
    {"id": "mistral", "name": "Mistral", "platform": "openai_compatible",
     "host": "https://api.mistral.ai/v1", "needs_key": True},
    {"id": "groq", "name": "Groq", "platform": "openai_compatible",
     "host": "https://api.groq.com/openai/v1", "needs_key": True},
    {"id": "openrouter", "name": "OpenRouter", "platform": "openai_compatible",
     "host": "https://openrouter.ai/api/v1", "needs_key": True},
    {"id": "nvidia", "name": "NVIDIA", "platform": "openai_compatible",
     "host": "https://integrate.api.nvidia.com/v1", "needs_key": True},
    {"id": "minimax", "name": "MiniMax", "platform": "openai_compatible",
     "host": "https://api.minimax.io/v1", "needs_key": True},
    {"id": "moonshot", "name": "Moonshot (Kimi)", "platform": "openai_compatible",
     "host": "https://api.moonshot.ai/v1", "needs_key": True},
    {"id": "fireworks", "name": "Fireworks AI", "platform": "openai_compatible",
     "host": "https://api.fireworks.ai/inference/v1", "needs_key": True},
    {"id": "zhipu", "name": "Zhipu (GLM)", "platform": "openai_compatible",
     "host": "https://open.bigmodel.cn/api/paas/v4", "needs_key": True},
    {"id": "alibaba", "name": "Alibaba (Qwen)", "platform": "openai_compatible",
     "host": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "needs_key": True},
    {"id": "novita", "name": "Novita AI", "platform": "openai_compatible",
     "host": "https://api.novita.ai/openai", "needs_key": True},
    {"id": "baseten", "name": "Baseten", "platform": "openai_compatible",
     "host": "https://inference.baseten.co/v1", "needs_key": True},
]


def get_catalog() -> list[dict[str, object]]:
    return CURATED_PROVIDERS
