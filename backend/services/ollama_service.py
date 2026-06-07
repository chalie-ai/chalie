"""
Ollama service helpers — shared utilities used by OllamaClient and tests.

``OllamaService`` was deleted in TKT-846: the new thin client lives at
``services/llm_clients/ollama.py``.  This module is retained as a helper
bag because:

- ``services/llm_clients/ollama.py`` imports ``_validate_model`` and
  ``_validate_host`` from here (CodeQL sanitisation barriers — moving them
  would require a test-import-path change that the tester must own).
- Tests import ``_ollama_convert_messages`` and ``_parse_chat_response``
  from this module path; those imports must not be broken until the tester
  agent migrates them.

Live helpers (all functions below):
  ``_validate_model``, ``_validate_host``, ``_parse_chat_response``,
  ``_ollama_convert_messages``.
"""

import re
from urllib.parse import urlparse
from uuid import uuid4

# Model identifier accepts alphanumeric, dot, underscore, dash, slash, and the
# `:cloud` / `:7b` size suffix separator. Validating in __init__ acts as a
# CodeQL sanitisation barrier so logging the model name later does not
# trip py/clear-text-logging-sensitive-data on the config-derived value.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:\-/]+$")
_ALLOWED_HOST_SCHEMES = frozenset({"http", "https"})


def _validate_model(raw_model) -> str:
    """Reject Ollama model identifiers that contain anything but the canonical
    name characters. Raising here acts as a CodeQL sanitisation barrier so
    ``self.model`` is no longer treated as derived-from-config-dict tainted
    data when it later lands in log calls.
    """
    if raw_model is None:
        raise ValueError("Ollama config requires a 'model' field")
    text = str(raw_model)
    if not _MODEL_NAME_RE.fullmatch(text):
        raise ValueError(f"Invalid Ollama model identifier: {raw_model!r}")
    return text


def _validate_host(raw_host) -> str:
    """Validate the Ollama host URL via ``urllib.parse.urlparse`` + scheme +
    netloc gates. Same CodeQL-barrier rationale as ``_validate_model``.
    """
    if raw_host is None:
        raise ValueError("Ollama config requires a 'host' field")
    text = str(raw_host)
    parsed = urlparse(text)
    if parsed.scheme not in _ALLOWED_HOST_SCHEMES or not parsed.netloc:
        raise ValueError(f"Invalid Ollama host URL: {raw_host!r}")
    return text


def _parse_chat_response(data: dict, default_model: str):
    """Build a response object from a raw Ollama /api/chat response dict.

    Native tool-calling only: tool calls are read exclusively from the
    structured ``message.tool_calls`` field. No inline-content fallback.

    Returns a ``ProviderApiResponse`` (tests that import this helper and
    compare attributes rely on the same dataclass regardless of import path).
    """
    from services.provider_api import ProviderApiResponse  # noqa: PLC0415
    msg = data.get('message', {})
    text = msg.get('content', '')
    tool_calls = None
    raw_tool_calls = msg.get('tool_calls')
    if raw_tool_calls:
        # call_id must be globally unique across the whole turn, not just
        # within one response: the index resets every LLM call, so an
        # index-only id collides across ACT iterations and strands the
        # ACT-trail spinner (TKT-786). A uuid suffix guarantees uniqueness.
        tool_calls = [
            {
                'id': f"ollama_{tc.get('function', {}).get('name', 'unknown')}_{uuid4().hex[:8]}",
                'name': tc.get('function', {}).get('name', ''),
                'input': tc.get('function', {}).get('arguments', {}),
            }
            for tc in raw_tool_calls
        ]
    return ProviderApiResponse(
        text=text,
        model=data.get('model', default_model),
        provider='ollama',
        tokens_input=data.get('prompt_eval_count'),
        tokens_output=data.get('eval_count'),
        tool_calls=tool_calls,
        stop_reason='tool_use' if tool_calls else 'end_turn',
        response_code=200,
    )


def _ollama_convert_messages(messages: list) -> list:
    """Convert normalized messages to Ollama format (OpenAI-compatible)."""
    result = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append({
                "role": "assistant",
                "content": msg.get('content', ''),
                "tool_calls": [
                    {
                        "function": {
                            "name": tc['name'],
                            "arguments": tc['input'],
                        },
                    }
                    for tc in msg['tool_calls']
                ],
            })
        elif msg['role'] == 'tool':
            result.append({
                "role": "tool",
                "content": msg.get('content', ''),
            })
        else:
            img = msg.get('image')
            if img:
                result.append({
                    "role": msg['role'],
                    "content": msg.get('content', ''),
                    "images": [img['data']],
                })
            else:
                result.append(msg)
    return result
