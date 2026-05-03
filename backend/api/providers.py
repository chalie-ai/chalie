"""
Providers blueprint — manage LLM provider configuration via REST API.
"""

import ipaddress
import logging
from urllib.parse import urlparse

import requests as req
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

_ERR_PROVIDER_NOT_FOUND = "Provider not found"

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
        pass  # hostname (not an IP literal) — fine
    return safe, None


def _fetch_ollama_models(host: str):
    """Fetch the model list from an Ollama instance via GET /api/tags.

    Returns ``(model_names, error_str)`` where exactly one is non-None.
    """
    safe_host, err = _validate_ollama_host(host)
    if err is not None:
        return None, err
    try:
        r = req.get(f"{safe_host}/api/tags", timeout=5)
        r.raise_for_status()
        models_data = r.json()
        names = [
            m.get('name') or m.get('model', '')
            for m in (models_data.get('models') or [])
            if m.get('name') or m.get('model', '')
        ]
        return names, None
    except req.exceptions.ConnectionError:
        return None, f"Cannot connect to {safe_host} — is the service running?"
    except req.exceptions.Timeout:
        return None, f"Connection to {safe_host} timed out"
    except Exception as e:
        logger.warning(f"[REST API] Ollama model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch Ollama models"


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
        return jsonify({"error": str(e)}), 400
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
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[REST API] Failed to update provider: {e}")
        return jsonify({"error": "Failed to update provider"}), 500


@providers_bp.route('/<int:provider_id>', methods=['DELETE'])
@require_session
def delete_provider(provider_id):
    """Delete a provider (set is_active=FALSE)."""
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
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        logger.error(f"[REST API] Failed to delete provider: {e}")
        return jsonify({"error": "Failed to delete provider"}), 500


@providers_bp.route('/ollama/models', methods=['GET'])
@require_session
def list_ollama_models():
    """Proxy GET /api/tags on an Ollama host and return model names.

    Query param:
      host  — Ollama base URL (default: http://localhost:11434)

    Response 200: {"models": ["name:tag", ...]}
    Response 502: {"error": "..."}
    """
    host = request.args.get('host', 'http://localhost:11434')
    names, err = _fetch_ollama_models(host)
    if err is not None:
        return jsonify({"error": err}), 502
    return jsonify({"models": names}), 200


@providers_bp.route('/anthropic/models', methods=['POST'])
@require_session
def list_anthropic_models():
    """Proxy the Anthropic models list endpoint server-side.

    Body JSON:
      {"api_key": "sk-ant-..."}

    The key is POSTed (not a query param) so it does not land in Flask access
    logs, browser history, or reverse-proxy logs.

    Response 200: {"models": ["claude-...", ...]}
    Response 400: {"error": "api_key is required"}
    Response 502: {"error": "..."}
    """
    body = request.get_json(silent=True) or {}
    api_key = (body.get('api_key') or '').strip()
    if not api_key:
        return jsonify({"error": "api_key is required"}), 400

    try:
        r = req.get(
            'https://api.anthropic.com/v1/models',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            },
            timeout=10,
        )
        if not r.ok:
            return jsonify({"error": f"Anthropic API returned {r.status_code}"}), 502
        data = r.json()
        model_ids = [m['id'] for m in (data.get('data') or []) if m.get('id')]
        return jsonify({"models": model_ids}), 200
    except req.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Anthropic API"}), 502
    except req.exceptions.Timeout:
        return jsonify({"error": "Anthropic API request timed out"}), 502
    except Exception as e:
        logger.error(f"[REST API] Anthropic model list failed: {type(e).__name__}: {e}")
        return jsonify({"error": "Anthropic API request failed"}), 502


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
            available, err = _fetch_ollama_models(config.get('host', ''))
            latency_ms = int((time.time() - start) * 1000)

            if err is not None:
                return jsonify({"success": False, "error": err}), 200

            model_base = model.split(':')[0]
            model_found = any(
                m == model or m.startswith(model + ':') or m.split(':')[0] == model_base
                for m in available
            )

            if not model_found and not available:
                return jsonify({
                    "success": True,
                    "model": model,
                    "latency_ms": latency_ms,
                    "message": "Connected to Ollama (no models installed yet)"
                }), 200

            if not model_found:
                return jsonify({
                    "success": False,
                    "error": f"Model '{model}' not found on this Ollama instance.",
                    "hint": f"Run: ollama pull {model}  ·  Available: {', '.join(available[:5])}"
                }), 200

            return jsonify({
                "success": True,
                "model": model,
                "latency_ms": latency_ms,
                "message": f"Connected · {len(available)} model(s) available"
            }), 200

        else:
            # API-based providers (anthropic, openai, gemini)
            api_key = config.get('api_key')
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "API key is required to test this provider",
                    "hint": "Enter your API key in the field above"
                }), 200

            try:
                test_config = {
                    'platform': platform,
                    'model': model,
                    'api_key': api_key,
                    'max_tokens': 1,
                    'timeout': 10,
                }
                host = config.get('host')
                if host:
                    test_config['host'] = host
                from services.llm_service import create_llm_service
                llm = create_llm_service(test_config)
                llm.send_message("You are a test assistant.", "Say: ok")
                latency_ms = int((time.time() - start) * 1000)
                return jsonify({
                    "success": True,
                    "model": model,
                    "latency_ms": latency_ms,
                    "message": "Connected successfully"
                }), 200

            except Exception as e:
                error_str = str(e)
                el = error_str.lower()
                error_msg = error_str[:300]
                if any(k in el for k in ('authentication', 'auth_token', 'api_key', 'invalid_api', '401', 'unauthorized', 'invalid x-api-key')):
                    error_msg = "Invalid API key"
                elif any(k in el for k in ('model_not_found', 'not found', 'does not exist', 'no such model', '404')):
                    error_msg = f"Model '{model}' not found — check the model name"
                elif any(k in el for k in ('quota', 'rate_limit', 'rate limit', '429', 'too many')):
                    error_msg = "API quota exceeded or rate limited — try again later"
                elif any(k in el for k in ('timeout', 'timed out')):
                    error_msg = f"Connection to {platform} timed out after 10s"
                elif any(k in el for k in ('connectionerror', 'connection refused', 'connect')):
                    error_msg = f"Cannot connect to {platform} — is the service running?"
                elif any(k in el for k in ('network', 'ssl')):
                    error_msg = "Network error — check your internet connection"
                return jsonify({
                    "success": False,
                    "error": error_msg,
                }), 200

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
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[REST API] Failed to set selected provider: {e}")
        return jsonify({"error": "Failed to set selected provider"}), 500
