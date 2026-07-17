"""Provider probe — live connectivity checks and model discovery for LLM providers.

Bare-function toolkit (no service class, no DB access) covering three concerns
that all boil down to "reach out to a provider and report back":

- SSRF-hardened host validation for Ollama / openai_compatible endpoints.
- Live model-list fetch per platform (never raises — always ``(models, error)``).
- Live connectivity test per platform (never raises — always a
  :class:`ProviderTestOutcome`, the plain dataclass equivalent of the API
  layer's ``ProviderTestResult`` DTO; this module must not import from
  ``api.*``, so it owns its own outcome shape and the API layer converts it).

Also carries the small validation-message allowlist shared by the provider
CRUD endpoint's create/update/delete paths, and the cache-invalidation call
every provider-mutating write makes.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import requests as req

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
from services.provider_db_service import PROVIDER_IN_USE_MSG

logger = logging.getLogger(__name__)

DUPLICATE_NAME_MSG = "A provider with that name already exists"
_HOST_NOT_ALLOWED_MSG = "Host is not allowed"
_API_KEY_REQUIRED_MSG = "API key is required"
_INVALID_API_KEY_MSG = "Invalid API key"

SAFE_VALIDATION_MESSAGES = {
    "'model' is required",
    "openai_compatible provider requires 'host' field "
    "(base URL, e.g. 'https://api.minimax.io/v1')",
    "openai_compatible provider requires 'api_key' field",
    PROVIDER_IN_USE_MSG,
}


def safe_validation_msg(exc: ValueError) -> str:
    """Return a user-facing message from a provider validation error.

    Only allowlisted messages are returned verbatim; anything else is
    replaced with a generic string to prevent internal detail leakage.
    """
    msg = str(exc)
    if msg in SAFE_VALIDATION_MESSAGES:
        return msg
    return "Invalid provider configuration"


def invalidate_provider_cache() -> None:
    """Bust the in-memory provider cache after any provider row/role mutation."""
    try:
        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
        ProviderCacheService.invalidate()
    except Exception as e:
        logger.warning(f"[Provider probe] Failed to invalidate provider cache: {e}")


# SSRF hard denies — cloud metadata, link-local, and common cloud-provider
# private endpoints. Loopback and RFC1918 ranges are allowed (local-first app).
_SSRF_BLOCKED_HOSTS = {
    '169.254.169.254',   # AWS / Azure / GCP metadata
    '100.100.100.200',   # Alibaba Cloud metadata
    'metadata.google.internal',
    'metadata',
}


def normalise_ollama_host(host: str) -> str:
    host = (host or '').strip().rstrip('/')
    if host and '://' not in host:
        host = 'http://' + host
    return host or 'http://localhost:11434'


def _check_dns_resolved_ips(hostname: str) -> str | None:
    """Resolve ``hostname`` via DNS and return an error message if any
    resolved address is blocked/link-local/multicast/unspecified, else
    ``None``. Extracted from validate_ollama_host."""
    try:
        resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        for _, _, _, _, sockaddr in resolved:
            ip_str = sockaddr[0]
            if ip_str in _SSRF_BLOCKED_HOSTS:
                return _HOST_NOT_ALLOWED_MSG
            ip = ipaddress.ip_address(ip_str)
            if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
                return _HOST_NOT_ALLOWED_MSG
    except socket.gaierror:
        return "Cannot resolve hostname"
    return None


def validate_ollama_host(host: str) -> tuple[str | None, str | None]:
    safe = normalise_ollama_host(host)
    parsed = urlparse(safe)
    if parsed.scheme not in ('http', 'https'):
        return None, f"Unsupported scheme '{parsed.scheme}' — use http or https"
    hostname = (parsed.hostname or '').lower()
    if not hostname:
        return None, "Host is required"
    if hostname in _SSRF_BLOCKED_HOSTS:
        return None, _HOST_NOT_ALLOWED_MSG
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
            return None, _HOST_NOT_ALLOWED_MSG
    except ValueError:
        err = _check_dns_resolved_ips(hostname)
        if err is not None:
            return None, err
    return safe, None


# --- Live model-list fetch helpers -----------------------------------------
#
# Each helper returns ``(models, error)`` where ``models`` is a list of dicts
# ``{"id": str, "display_name": str | None}`` and exactly one of the two is
# non-None. All helpers cap upstream calls at 8s and never raise.

_LIST_MODELS_TIMEOUT = 8

# OpenAI model IDs we accept: chat-capable text models only. Drop audio/realtime/
# image/tts/whisper/embedding/moderation variants — none are useful for ACT.
_OPENAI_PREFIX_OK = ('gpt-', 'o1', 'o3', 'o4', 'o5')
_OPENAI_DENY_SUBSTR = (
    'audio', 'realtime', 'image', 'tts', 'whisper', 'embedding', 'moderation',
)


def fetch_ollama_models(host: str) -> tuple[list[dict[str, str | None]] | None, str | None]:
    safe_host, err = validate_ollama_host(host)
    if err is not None:
        return None, err
    try:
        r = req.get(f"{cast(str, safe_host)}/api/tags", timeout=_LIST_MODELS_TIMEOUT)
        r.raise_for_status()
        models_data = r.json()
        models = []
        for m in (models_data.get('models') or []):
            name = m.get('name') or m.get('model', '')
            if name:
                models.append({"id": name, "display_name": None})
        return models, None
    except req.exceptions.ConnectionError:
        return None, f"Cannot connect to {safe_host} — is the service running?"
    except req.exceptions.Timeout:
        return None, f"Connection to {safe_host} timed out"
    except Exception as e:
        logger.warning(f"[Provider probe] Ollama model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch Ollama models"


def fetch_openai_models(api_key: str) -> tuple[list[dict[str, str | None]] | None, str | None]:
    """Filter to chat-capable text models only."""
    if not api_key:
        return None, _API_KEY_REQUIRED_MSG
    try:
        r = req.get(
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=_LIST_MODELS_TIMEOUT,
        )
        if r.status_code in (401, 403):
            return None, _INVALID_API_KEY_MSG
        if not r.ok:
            return None, f"OpenAI API returned {r.status_code}"
        data = r.json()
        items = data.get('data') or []
        # Filter to chat-capable text models, then sort newest first by 'created'.
        kept = []
        for m in items:
            mid = m.get('id') or ''
            if not mid or not mid.startswith(_OPENAI_PREFIX_OK):
                continue
            lid = mid.lower()
            if any(bad in lid for bad in _OPENAI_DENY_SUBSTR):
                continue
            kept.append(m)
        kept.sort(key=lambda m: m.get('created') or 0, reverse=True)
        return [{"id": m['id'], "display_name": None} for m in kept], None
    except req.exceptions.ConnectionError:
        return None, "Cannot connect to OpenAI API"
    except req.exceptions.Timeout:
        return None, "OpenAI API request timed out"
    except Exception as e:
        logger.warning(f"[Provider probe] OpenAI model list failed: {type(e).__name__}: {e}")
        return None, "OpenAI API request failed"


def fetch_anthropic_models(api_key: str) -> tuple[list[dict[str, str | None]] | None, str | None]:
    if not api_key:
        return None, _API_KEY_REQUIRED_MSG
    try:
        r = req.get(
            'https://api.anthropic.com/v1/models',
            params={'limit': 1000},
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            },
            timeout=_LIST_MODELS_TIMEOUT,
        )
        if r.status_code in (401, 403):
            return None, _INVALID_API_KEY_MSG
        if not r.ok:
            return None, f"Anthropic API returned {r.status_code}"
        data = r.json()
        items = list(data.get('data') or [])
        # Sort by created_at desc when available; falsy values sink to the end.
        items.sort(key=lambda m: m.get('created_at') or '', reverse=True)
        models = []
        for m in items:
            mid = m.get('id')
            if not mid:
                continue
            models.append({
                "id": mid,
                "display_name": m.get('display_name'),
            })
        return models, None
    except req.exceptions.ConnectionError:
        return None, "Cannot connect to Anthropic API"
    except req.exceptions.Timeout:
        return None, "Anthropic API request timed out"
    except Exception as e:
        logger.warning(f"[Provider probe] Anthropic model list failed: {type(e).__name__}: {e}")
        return None, "Anthropic API request failed"


_GEMINI_MODELS_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
# Cap pagination follow-through so a misbehaving upstream cannot loop forever.
_GEMINI_MAX_PAGES = 10


def _extract_gemini_models_from_page(data: Any) -> list[dict[str, str | None]]:
    """Filter one Gemini list-models page to generateContent-capable models,
    stripping the ``models/`` prefix from each id. Extracted from
    fetch_gemini_models."""
    models: list[dict[str, str | None]] = []
    for m in (data.get('models') or []):
        methods = m.get('supportedGenerationMethods') or []
        if 'generateContent' not in methods:
            continue
        name = m.get('name') or ''
        mid = name[len('models/'):] if name.startswith('models/') else name
        if not mid:
            continue
        models.append({"id": mid, "display_name": m.get('displayName')})
    return models


def _fetch_gemini_page(
    api_key: str, page_token: str | None,
) -> tuple[list[dict[str, str | None]], str | None, str | None]:
    """Fetch and filter one page of Gemini models. Returns
    ``(models, next_page_token, error)``; ``models`` is empty and
    ``next_page_token`` is ``None`` when ``error`` is set. Extracted from
    fetch_gemini_models."""
    params: dict[str, str | int] = {'pageSize': 1000}
    if page_token:
        params['pageToken'] = page_token
    r = req.get(
        _GEMINI_MODELS_URL,
        params=params,
        headers={'x-goog-api-key': api_key},
        timeout=_LIST_MODELS_TIMEOUT,
    )
    if r.status_code in (400, 401, 403):
        return [], None, _INVALID_API_KEY_MSG
    if not r.ok:
        return [], None, f"Gemini API returned {r.status_code}"
    data = r.json()
    models = _extract_gemini_models_from_page(data)
    return models, data.get('nextPageToken'), None


def fetch_gemini_models(api_key: str) -> tuple[list[dict[str, str | None]] | None, str | None]:
    """Cap pagination at 10 pages. Only ``generateContent`` models; strips ``models/`` prefix."""
    if not api_key:
        return None, _API_KEY_REQUIRED_MSG
    try:
        models = []
        page_token = None
        for _ in range(_GEMINI_MAX_PAGES):
            page_models, page_token, err = _fetch_gemini_page(api_key, page_token)
            if err is not None:
                return None, err
            models.extend(page_models)
            if not page_token:
                break
        return models, None
    except req.exceptions.ConnectionError:
        return None, "Cannot connect to Gemini API"
    except req.exceptions.Timeout:
        return None, "Gemini API request timed out"
    except Exception as e:
        logger.warning(f"[Provider probe] Gemini model list failed: {type(e).__name__}: {e}")
        return None, "Gemini API request failed"


def _parse_openai_compatible_models(data: Any) -> list[dict[str, str | None]]:
    """Normalise an OpenAI-compatible /models response (dict-wrapped, or a
    bare list; string or object entries) into the common model-dict shape.
    Extracted from fetch_openai_compatible_models."""
    items = data.get('data') or []
    if isinstance(data, list):
        items = data
    models: list[dict[str, str | None]] = []
    for m in items:
        if isinstance(m, str):
            models.append({"id": m, "display_name": None})
            continue
        mid = m.get('id') or m.get('name') or ''
        if not mid:
            continue
        models.append({"id": mid, "display_name": m.get('display_name')})
    models.sort(key=lambda m: cast(str, m['id']))
    return models


def fetch_openai_compatible_models(
    host: str, api_key: str,
) -> tuple[list[dict[str, str | None]] | None, str | None]:
    if not host:
        return None, "Host URL is required"
    if not api_key:
        return None, _API_KEY_REQUIRED_MSG
    safe_host, err = validate_ollama_host(host)
    if err is not None:
        return None, err
    url = cast(str, safe_host).rstrip('/') + '/models'
    try:
        r = req.get(
            url,
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=_LIST_MODELS_TIMEOUT,
        )
        if r.status_code in (401, 403):
            return None, _INVALID_API_KEY_MSG
        if not r.ok:
            return None, f"API returned {r.status_code}"
        data = r.json()
        models = _parse_openai_compatible_models(data)
        return models, None
    except req.exceptions.ConnectionError:
        return None, f"Cannot connect to {safe_host}"
    except req.exceptions.Timeout:
        return None, f"Request to {safe_host} timed out"
    except Exception as e:
        logger.warning(f"[Provider probe] OpenAI-compatible model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch models"


def fetch_codex_models() -> tuple[list[dict[str, str | None]] | None, str | None]:
    from services.llm_clients.codex_cli import list_codex_models  # noqa: PLC0415
    models = list_codex_models()
    if not models:
        return None, "Codex CLI not initialised — install the codex CLI and run `codex login`"
    return models, None


def map_api_error(error_str: str, platform: str, model: str) -> str:
    el = error_str.lower()
    if any(k in el for k in ('authentication', 'auth_token', 'api_key', 'invalid_api', '401', 'unauthorized', 'invalid x-api-key')):
        return _INVALID_API_KEY_MSG
    if any(k in el for k in ('model_not_found', 'not found', 'does not exist', 'no such model', '404')):
        return f"Model '{model}' not found — check the model name"
    if any(k in el for k in ('quota', 'rate_limit', 'rate limit', '429', 'too many')):
        return "API quota exceeded or rate limited — try again later"
    if any(k in el for k in ('timeout', 'timed out')):
        return f"Connection to {platform} timed out after 10s"
    if any(k in el for k in ('connectionerror', 'connection refused', 'connect')):
        return f"Cannot connect to {platform} — is the service running?"
    if any(k in el for k in ('network', 'ssl')):
        return "Network error — check your internet connection"
    logger.warning("[Provider probe] Unmapped upstream provider error for platform=%s model=%s: %s", platform, model, error_str)
    return "Upstream provider error"


@dataclass
class ProviderTestOutcome:
    """Result of a connectivity test — the services-layer equivalent of the API
    layer's ``ProviderTestResult`` response DTO (this module cannot import
    ``api.*``, so it owns this plain shape; the test action converts it)."""

    success: bool
    model: str | None = None
    latency_ms: int | None = None
    message: str | None = None
    error: str | None = None
    hint: str | None = None


def test_ollama_provider(config: dict[str, object], model: str, start: float) -> ProviderTestOutcome:
    import time
    available, err = fetch_ollama_models(cast(str, config.get('host', '')))
    latency_ms = int((time.time() - start) * 1000)

    if err is not None:
        return ProviderTestOutcome(success=False, error=err)

    available_names = [cast(str, m['id']) for m in (available or [])]
    model_base = model.split(':')[0]
    model_found = any(
        m == model or m.startswith(model + ':') or m.split(':')[0] == model_base
        for m in available_names
    )

    if not model_found and not available_names:
        return ProviderTestOutcome(
            success=True, model=model, latency_ms=latency_ms,
            message="Connected to Ollama (no models installed yet)",
        )

    if not model_found:
        return ProviderTestOutcome(
            success=False,
            error=f"Model '{model}' not found on this Ollama instance.",
            hint=f"Run: ollama pull {model}  ·  Available: {', '.join(available_names[:5])}",
        )

    return ProviderTestOutcome(
        success=True, model=model, latency_ms=latency_ms,
        message=f"Connected · {len(available_names)} model(s) available",
    )


def test_api_provider(config: dict[str, object], platform: str, model: str, start: float) -> ProviderTestOutcome:
    import time
    api_key = config.get('api_key')
    if not api_key:
        return ProviderTestOutcome(
            success=False,
            error="API key is required to test this provider",
            hint="Enter your API key in the field above",
        )

    try:
        test_config: dict[str, object] = {
            'platform': platform, 'model': model,
            'api_key': api_key, 'max_tokens': 1,
        }
        host = config.get('host')
        if host:
            test_config['host'] = host
        from services.llm_clients.factory import build_client
        from services.provider_api import ProviderApiRequest
        client = build_client(test_config)
        dto = ProviderApiRequest(
            system="You are a test assistant.",
            messages=[{"role": "user", "content": "Say: ok"}],
            type=ProviderType.CHAT,
            thinking_mode=ThinkingLevel.LOW,
            cache_prefix=False,
            max_tokens=1,
        )
        client.send(dto)
        latency_ms = int((time.time() - start) * 1000)
        return ProviderTestOutcome(success=True, model=model, latency_ms=latency_ms, message="Connected successfully")
    except Exception as e:
        return ProviderTestOutcome(success=False, error=map_api_error(str(e), platform, model))


def test_codex_provider(model: str, start: float) -> ProviderTestOutcome:
    # codex_cli is subscription-billed on a scarce free tier — never run inference
    # to test it. A binary + login presence check is sufficient and costs 0 tokens.
    import os
    import subprocess
    import time

    from services.llm_clients.codex_cli import _codex_home
    binary = os.environ.get("CODEX_BIN", "codex")
    try:
        proc = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ProviderTestOutcome(
            success=False,
            error="Codex CLI not found",
            hint="Install the codex CLI and ensure it is on PATH",
        )
    if proc.returncode != 0:
        return ProviderTestOutcome(
            success=False,
            error="Codex CLI not found",
            hint="Install the codex CLI and ensure it is on PATH",
        )

    if not (_codex_home() / "auth.json").exists():
        return ProviderTestOutcome(
            success=False,
            error="Codex CLI is not logged in",
            hint="Run `codex login`",
        )

    latency_ms = int((time.time() - start) * 1000)
    return ProviderTestOutcome(
        success=True, model=model, latency_ms=latency_ms,
        message=f"Codex CLI ready ({(proc.stdout or '').strip()})",
    )
