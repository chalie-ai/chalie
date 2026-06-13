"""
Providers blueprint — manage LLM provider configuration via REST API.
"""

import ipaddress
import logging
import socket
import sqlite3
from urllib.parse import urlparse

import requests as req
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

_ERR_PROVIDER_NOT_FOUND = "Provider not found"
_DUPLICATE_NAME_MSG = "A provider with that name already exists"

_SAFE_VALIDATION_MESSAGES = {
    "'model' is required",
    "openai_compatible provider requires 'host' field "
    "(base URL, e.g. 'https://api.minimax.io/v1')",
    "openai_compatible provider requires 'api_key' field",
}


def _safe_validation_msg(exc: ValueError) -> str:
    """Return a user-facing message from a provider validation error.

    Only allowlisted messages are returned verbatim; anything else is
    replaced with a generic string to prevent internal detail leakage.
    """
    msg = str(exc)
    if msg in _SAFE_VALIDATION_MESSAGES:
        return msg
    return "Invalid provider configuration"

providers_bp = Blueprint('providers', __name__, url_prefix='/providers')

# SSRF hard denies — cloud metadata, link-local, and common cloud-provider
# private endpoints. Loopback and RFC1918 ranges are allowed (local-first app).
_SSRF_BLOCKED_HOSTS = {
    '169.254.169.254',   # AWS / Azure / GCP metadata
    '100.100.100.200',   # Alibaba Cloud metadata
    'metadata.google.internal',
    'metadata',
}


def _normalise_ollama_host(host: str) -> str:
    """Strip trailing slash; prepend http:// if no scheme is present."""
    host = (host or '').strip().rstrip('/')
    if host and '://' not in host:
        host = 'http://' + host
    return host or 'http://localhost:11434'


def _validate_ollama_host(host: str) -> tuple[str | None, str | None]:
    """Return ``(safe_host, error)``. ``safe_host`` is the normalised URL if OK.

    Rejects non-http(s) schemes and cloud-metadata endpoints. Private/loopback
    IPs are explicitly allowed — this is a local-first app where pointing at
    192.168.x.y or localhost is the common case.
    """
    safe = _normalise_ollama_host(host)
    parsed = urlparse(safe)
    if parsed.scheme not in ('http', 'https'):
        return None, f"Unsupported scheme '{parsed.scheme}' — use http or https"
    hostname = (parsed.hostname or '').lower()
    if not hostname:
        return None, "Host is required"
    if hostname in _SSRF_BLOCKED_HOSTS:
        return None, "Host is not allowed"
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
            return None, "Host is not allowed"
    except ValueError:
        try:
            resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
            for _, _, _, _, sockaddr in resolved:
                ip_str = sockaddr[0]
                if ip_str in _SSRF_BLOCKED_HOSTS:
                    return None, "Host is not allowed"
                ip = ipaddress.ip_address(ip_str)
                if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
                    return None, "Host is not allowed"
        except socket.gaierror:
            return None, "Cannot resolve hostname"
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


def _fetch_ollama_models(host: str):
    """Fetch the model list from an Ollama instance via GET /api/tags."""
    safe_host, err = _validate_ollama_host(host)
    if err is not None:
        return None, err
    try:
        r = req.get(f"{safe_host}/api/tags", timeout=_LIST_MODELS_TIMEOUT)
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
        logger.warning(f"[REST API] Ollama model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch Ollama models"


def _fetch_openai_models(api_key: str):
    """Fetch chat-capable model IDs from GET https://api.openai.com/v1/models."""
    if not api_key:
        return None, "API key is required"
    try:
        r = req.get(
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=_LIST_MODELS_TIMEOUT,
        )
        if r.status_code in (401, 403):
            return None, "Invalid API key"
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
        logger.warning(f"[REST API] OpenAI model list failed: {type(e).__name__}: {e}")
        return None, "OpenAI API request failed"


def _fetch_anthropic_models(api_key: str):
    """Fetch chat models from GET https://api.anthropic.com/v1/models."""
    if not api_key:
        return None, "API key is required"
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
            return None, "Invalid API key"
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
        logger.warning(f"[REST API] Anthropic model list failed: {type(e).__name__}: {e}")
        return None, "Anthropic API request failed"


_GEMINI_MODELS_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
# Cap pagination follow-through so a misbehaving upstream cannot loop forever.
_GEMINI_MAX_PAGES = 10


def _fetch_gemini_models(api_key: str):
    """Fetch chat-capable model IDs from the Gemini ListModels REST API.

    Keeps only models advertising ``generateContent`` and strips the leading
    ``models/`` prefix from each name so the IDs match what the provider form
    and ``GeminiService`` expect (e.g. ``gemini-2.5-flash``). The key travels
    in the ``x-goog-api-key`` header — never the URL — to keep it out of access
    and proxy logs, mirroring the other helpers.
    """
    if not api_key:
        return None, "API key is required"
    try:
        models = []
        page_token = None
        for _ in range(_GEMINI_MAX_PAGES):
            params = {'pageSize': 1000}
            if page_token:
                params['pageToken'] = page_token
            r = req.get(
                _GEMINI_MODELS_URL,
                params=params,
                headers={'x-goog-api-key': api_key},
                timeout=_LIST_MODELS_TIMEOUT,
            )
            if r.status_code in (400, 401, 403):
                return None, "Invalid API key"
            if not r.ok:
                return None, f"Gemini API returned {r.status_code}"
            data = r.json()
            for m in (data.get('models') or []):
                methods = m.get('supportedGenerationMethods') or []
                if 'generateContent' not in methods:
                    continue
                name = m.get('name') or ''
                mid = name[len('models/'):] if name.startswith('models/') else name
                if not mid:
                    continue
                models.append({"id": mid, "display_name": m.get('displayName')})
            page_token = data.get('nextPageToken')
            if not page_token:
                break
        return models, None
    except req.exceptions.ConnectionError:
        return None, "Cannot connect to Gemini API"
    except req.exceptions.Timeout:
        return None, "Gemini API request timed out"
    except Exception as e:
        logger.warning(f"[REST API] Gemini model list failed: {type(e).__name__}: {e}")
        return None, "Gemini API request failed"


def _fetch_openai_compatible_models(host: str, api_key: str):
    """Fetch models from an OpenAI-compatible endpoint via GET {host}/models."""
    if not host:
        return None, "Host URL is required"
    if not api_key:
        return None, "API key is required"
    safe_host, err = _validate_ollama_host(host)
    if err is not None:
        return None, err
    url = safe_host.rstrip('/') + '/models'
    try:
        r = req.get(
            url,
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=_LIST_MODELS_TIMEOUT,
        )
        if r.status_code in (401, 403):
            return None, "Invalid API key"
        if not r.ok:
            return None, f"API returned {r.status_code}"
        data = r.json()
        items = data.get('data') or []
        if isinstance(data, list):
            items = data
        models = []
        for m in items:
            if isinstance(m, str):
                models.append({"id": m, "display_name": None})
                continue
            mid = m.get('id') or m.get('name') or ''
            if not mid:
                continue
            models.append({"id": mid, "display_name": m.get('display_name')})
        models.sort(key=lambda m: m['id'])
        return models, None
    except req.exceptions.ConnectionError:
        return None, f"Cannot connect to {safe_host}"
    except req.exceptions.Timeout:
        return None, f"Request to {safe_host} timed out"
    except Exception as e:
        logger.warning(f"[REST API] OpenAI-compatible model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch models"


def get_provider_service():
    """Get ProviderDbService instance."""
    from services.database_service import get_shared_db_service
    from services.provider_db_service import ProviderDbService
    db = get_shared_db_service()
    return ProviderDbService(db)


@providers_bp.route('', methods=['GET'])
@require_session
def list_providers():
    """List all active providers (omit api_key value)."""
    try:
        service = get_provider_service()
        providers = service.list_providers_summary()
        return jsonify({"providers": providers}), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to list providers: {e}")
        return jsonify({"error": "Failed to list providers"}), 500


@providers_bp.route('/catalog', methods=['GET'])
@require_session
def list_catalog_providers():
    """Return the curated provider presets for the setup wizard.

    Each preset maps a popular provider onto an existing Chalie platform plus a
    pre-fillable base URL — see :mod:`services.provider_catalog_service`.
    """
    try:
        from services.provider_catalog_service import get_catalog
        return jsonify({"catalog": get_catalog()}), 200
    except Exception as e:
        logger.exception(f"[REST API] Failed to load provider catalog: {e}")
        return jsonify({"error": "Failed to load provider catalog"}), 500


@providers_bp.route('', methods=['POST'])
@require_session
def create_provider():
    """Create a new provider."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Require name + platform + (model or models)
        if "name" not in data:
            return jsonify({"error": "Missing required field: name"}), 400
        if "platform" not in data:
            return jsonify({"error": "Missing required field: platform"}), 400
        if "model" not in data and "models" not in data:
            return jsonify({"error": "Missing required field: model or models"}), 400

        service = get_provider_service()
        provider = service.create_provider(data)

        # Invalidate provider cache
        try:
            from services.provider_cache_service import ProviderCacheService
            ProviderCacheService.invalidate()
        except Exception as e:
            logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")

        # Omit api_key value
        if provider.get("api_key"):
            provider["api_key"] = "***"

        return jsonify({"provider": provider}), 201
    except ValueError as e:
        logger.warning(f"[REST API] Provider validation error: {e}")
        return jsonify({"error": _safe_validation_msg(e)}), 400
    except sqlite3.IntegrityError as e:
        logger.warning(f"[REST API] Provider name conflict: {e}")
        return jsonify({"error": _DUPLICATE_NAME_MSG}), 409
    except Exception as e:
        logger.error(f"[REST API] Failed to create provider: {e}")
        return jsonify({"error": "Failed to create provider"}), 500


@providers_bp.route('/<int:provider_id>', methods=['GET'])
@require_session
def get_provider(provider_id):
    """Get a single provider by ID."""
    try:
        service = get_provider_service()
        provider = service.get_provider_by_id(provider_id)

        if not provider:
            return jsonify({"error": _ERR_PROVIDER_NOT_FOUND}), 404

        # Omit api_key value
        if provider.get("api_key"):
            provider["api_key"] = "***"

        return jsonify({"provider": provider}), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to get provider: {e}")
        return jsonify({"error": "Failed to get provider"}), 500


@providers_bp.route('/<int:provider_id>', methods=['PUT'])
@require_session
def update_provider(provider_id):
    """Update a provider."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        service = get_provider_service()
        provider = service.update_provider(provider_id, data)

        # Invalidate provider cache
        try:
            from services.provider_cache_service import ProviderCacheService
            ProviderCacheService.invalidate()
        except Exception as e:
            logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")

        # Omit api_key value
        if provider.get("api_key"):
            provider["api_key"] = "***"

        return jsonify({"provider": provider}), 200
    except ValueError as e:
        logger.warning(f"[REST API] Provider validation error: {e}")
        return jsonify({"error": _safe_validation_msg(e)}), 400
    except sqlite3.IntegrityError as e:
        logger.warning(f"[REST API] Provider name conflict: {e}")
        return jsonify({"error": _DUPLICATE_NAME_MSG}), 409
    except Exception as e:
        logger.error(f"[REST API] Failed to update provider: {e}")
        return jsonify({"error": "Failed to update provider"}), 500


@providers_bp.route('/<int:provider_id>', methods=['DELETE'])
@require_session
def delete_provider(provider_id):
    """Permanently delete a provider."""
    try:
        service = get_provider_service()
        service.delete_provider(provider_id)

        # Invalidate provider cache
        try:
            from services.provider_cache_service import ProviderCacheService
            ProviderCacheService.invalidate()
        except Exception as e:
            logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")

        return jsonify({"status": "deleted"}), 200
    except ValueError as e:
        logger.warning(f"[REST API] Cannot delete provider {provider_id}: {e}")
        return jsonify({"error": _safe_validation_msg(e)}), 409
    except Exception as e:
        logger.error(f"[REST API] Failed to delete provider: {e}")
        return jsonify({"error": "Failed to delete provider"}), 500


@providers_bp.route('/list-models', methods=['POST'])
@require_session
def list_models():
    """Live model-list fetch for the brain provider form.

    Body JSON:
      {"platform": "ollama" | "openai" | "anthropic" | "gemini"
                   | "openai_compatible",
       "api_key": "...",   # openai / anthropic / gemini / openai_compatible
       "host": "..."}      # ollama / openai_compatible

    Always returns HTTP 200 — auth/network/parse failures land in the body's
    ``error`` field so the UI can render them inline. Credentials are POSTed
    (never a query param) to keep them out of Flask access logs and reverse-
    proxy logs.

    Response 200: {"models": [{"id": "...", "display_name": "..."|null}, ...],
                   "error": "..."|null}
    Response 400: {"error": "..."}  (only for malformed/unsupported requests)
    """
    body = request.get_json(silent=True) or {}
    platform = (body.get('platform') or '').strip().lower()
    if platform == 'ollama':
        models, err = _fetch_ollama_models(body.get('host') or '')
    elif platform == 'openai':
        models, err = _fetch_openai_models((body.get('api_key') or '').strip())
    elif platform == 'anthropic':
        models, err = _fetch_anthropic_models((body.get('api_key') or '').strip())
    elif platform == 'gemini':
        models, err = _fetch_gemini_models((body.get('api_key') or '').strip())
    elif platform == 'openai_compatible':
        models, err = _fetch_openai_compatible_models(
            body.get('host') or '', (body.get('api_key') or '').strip(),
        )
    else:
        return jsonify({"models": [], "error": f"Unsupported platform '{platform}'"}), 400

    if err is not None:
        return jsonify({"models": [], "error": err}), 200
    return jsonify({"models": models, "error": None}), 200


def _map_api_error(error_str: str, platform: str, model: str) -> str:
    """Map a raw exception message to a user-facing error string."""
    el = error_str.lower()
    if any(k in el for k in ('authentication', 'auth_token', 'api_key', 'invalid_api', '401', 'unauthorized', 'invalid x-api-key')):
        return "Invalid API key"
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
    return error_str[:300]


def _test_ollama_provider(config: dict, model: str, start: float):
    """Test an Ollama provider and return a JSON response tuple."""
    import time
    available, err = _fetch_ollama_models(config.get('host', ''))
    latency_ms = int((time.time() - start) * 1000)

    if err is not None:
        return jsonify({"success": False, "error": err}), 200

    available_names = [m['id'] for m in (available or [])]
    model_base = model.split(':')[0]
    model_found = any(
        m == model or m.startswith(model + ':') or m.split(':')[0] == model_base
        for m in available_names
    )

    if not model_found and not available_names:
        return jsonify({
            "success": True, "model": model, "latency_ms": latency_ms,
            "message": "Connected to Ollama (no models installed yet)"
        }), 200

    if not model_found:
        return jsonify({
            "success": False,
            "error": f"Model '{model}' not found on this Ollama instance.",
            "hint": f"Run: ollama pull {model}  ·  Available: {', '.join(available_names[:5])}"
        }), 200

    return jsonify({
        "success": True, "model": model, "latency_ms": latency_ms,
        "message": f"Connected · {len(available_names)} model(s) available"
    }), 200


def _test_api_provider(config: dict, platform: str, model: str, start: float):
    """Test an API-based provider (Anthropic, OpenAI, etc.) and return a JSON response tuple."""
    import time
    api_key = config.get('api_key')
    if not api_key:
        return jsonify({
            "success": False,
            "error": "API key is required to test this provider",
            "hint": "Enter your API key in the field above"
        }), 200

    try:
        test_config = {
            'platform': platform, 'model': model,
            'api_key': api_key, 'max_tokens': 1, 'timeout': 10,
        }
        host = config.get('host')
        if host:
            test_config['host'] = host
        from services.llm_clients.factory import build_client
        from services.provider_api import ProviderApiRequest, ThinkingLevel, ProviderType
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
        return jsonify({
            "success": True, "model": model, "latency_ms": latency_ms,
            "message": "Connected successfully"
        }), 200
    except Exception as e:
        error_msg = _map_api_error(str(e), platform, model)
        return jsonify({"success": False, "error": error_msg}), 200


@providers_bp.route('/test', methods=['POST'])
@require_session
def test_provider():
    """Test a provider connection with a lightweight call."""
    import time

    try:
        data = request.get_json() or {}
        provider_id = data.get('provider_id')

        # Start from stored provider config if an ID is given
        config = {}
        if provider_id:
            service = get_provider_service()
            stored = service.get_provider_by_id(int(provider_id))
            if not stored:
                return jsonify({"success": False, "error": _ERR_PROVIDER_NOT_FOUND}), 200
            config = {k: v for k, v in stored.items() if v is not None}

        # Overlay fields from request body (so a new api_key / host can be tested)
        for field in ('platform', 'model', 'host', 'api_key'):
            val = data.get(field)
            if val:
                config[field] = val

        platform = config.get('platform')
        model = config.get('model')

        if not platform:
            return jsonify({"success": False, "error": "Platform is required"}), 200
        if not model:
            return jsonify({"success": False, "error": "Model is required"}), 200

        start = time.time()

        if platform == 'ollama':
            return _test_ollama_provider(config, model, start)
        else:
            return _test_api_provider(config, platform, model, start)

    except Exception as e:
        logger.error(f"[REST API] Provider test failed unexpectedly: {e}")
        return jsonify({"success": False, "error": "Test failed unexpectedly"}), 500


@providers_bp.route('/selected', methods=['GET'])
@require_session
def get_selected_provider():
    """Return the currently selected provider."""
    try:
        service = get_provider_service()
        provider = service.get_selected_provider()
        if not provider:
            return jsonify({"provider": None}), 200
        # Omit api_key value
        if provider.get("api_key"):
            provider["api_key"] = "***"
        return jsonify({"provider": provider}), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to get selected provider: {e}")
        return jsonify({"error": "Failed to get selected provider"}), 500


@providers_bp.route('/selected', methods=['PUT'])
@require_session
def set_selected_provider():
    """Set the selected provider ID."""
    try:
        data = request.get_json()
        if not data or "provider_id" not in data:
            return jsonify({"error": "Request body must contain 'provider_id'"}), 400

        provider_id = data["provider_id"]
        service = get_provider_service()

        # Validate the provider exists
        provider = service.get_provider_by_id(int(provider_id))
        if not provider:
            return jsonify({"error": _ERR_PROVIDER_NOT_FOUND}), 404

        service.set_selected_provider(int(provider_id))

        # Invalidate provider cache
        try:
            from services.provider_cache_service import ProviderCacheService
            ProviderCacheService.invalidate()
        except Exception as e:
            logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")

        # Omit api_key value
        if provider.get("api_key"):
            provider["api_key"] = "***"

        return jsonify({"provider": provider}), 200
    except ValueError as e:
        logger.warning(f"[REST API] Provider validation error: {e}")
        return jsonify({"error": _safe_validation_msg(e)}), 400
    except Exception as e:
        logger.error(f"[REST API] Failed to set selected provider: {e}")
        return jsonify({"error": "Failed to set selected provider"}), 500


@providers_bp.route('/vision', methods=['GET'])
@require_session
def get_vision_provider():
    """Return the configured vision provider + how it was resolved."""
    try:
        service = get_provider_service()
        status = service.get_vision_provider_status()
        provider = status['provider']
        if provider and provider.get('api_key'):
            provider['api_key'] = '***'
        return jsonify({'provider': provider, 'source': status['source']}), 200
    except Exception:
        logger.exception("[REST API] Failed to get vision provider")
        return jsonify({"error": "Failed to get vision provider"}), 500


@providers_bp.route('/vision', methods=['PUT'])
@require_session
def set_vision_provider():
    """Set (or clear) the explicit vision provider id."""
    try:
        data = request.get_json() or {}
        if 'provider_id' not in data:
            return jsonify({"error": "Request body must contain 'provider_id'"}), 400

        service = get_provider_service()
        pid = data['provider_id']

        if pid is None:
            service.set_vision_provider(None)
            return jsonify({'provider': None, 'source': 'none'}), 200

        provider = service.get_provider_by_id(int(pid))
        if not provider:
            return jsonify({"error": _ERR_PROVIDER_NOT_FOUND}), 404
        if not provider.get('supports_vision'):
            return jsonify({"error": "Provider does not support vision"}), 400

        service.set_vision_provider(int(pid))
        if provider.get('api_key'):
            provider['api_key'] = '***'
        return jsonify({'provider': provider, 'source': 'explicit'}), 200
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid provider_id"}), 400
    except Exception:
        logger.exception("[REST API] Failed to set vision provider")
        return jsonify({"error": "Failed to set vision provider"}), 500
