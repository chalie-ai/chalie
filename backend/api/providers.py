"""
Providers blueprint — manage LLM provider configuration via REST API.
"""

import ipaddress
import logging
import socket
import sqlite3
from typing import TYPE_CHECKING, TypeAlias, cast
from urllib.parse import urlparse

import requests as req
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.provider_db_service import PROVIDER_IN_USE_MSG
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.provider import Provider, ProviderCreate, ProviderUpdate
from .dto.provider_models import (
    ListModelsRequest,
    ListModelsResult,
    ModelInfo,
    ProviderTestRequest,
    ProviderTestResult,
)
from .dto.provider_role import NullableProviderRef, ProviderRef, ProviderRole

if TYPE_CHECKING:
    from services.provider_db_service import ProviderDbService

logger = logging.getLogger(__name__)

_ERR_PROVIDER_NOT_FOUND = "Provider not found"
_DUPLICATE_NAME_MSG = "A provider with that name already exists"

_OptStr: TypeAlias = "str | None"
_OptInt: TypeAlias = "int | None"
_OptDictStrObj: TypeAlias = "dict[str, object] | None"

_SAFE_VALIDATION_MESSAGES = {
    "'model' is required",
    "openai_compatible provider requires 'host' field "
    "(base URL, e.g. 'https://api.minimax.io/v1')",
    "openai_compatible provider requires 'api_key' field",
    PROVIDER_IN_USE_MSG,
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


providers_ns = Namespace('providers', description='Provider management', path='/api/providers')

register_dto(
    providers_ns,
    Provider, ProviderCreate, ProviderUpdate,
    ProviderRef, NullableProviderRef, ProviderRole,
    ListModelsRequest, ListModelsResult, ModelInfo,
    ProviderTestRequest, ProviderTestResult,
    Error,
)

_M = providers_ns.models

# SSRF hard denies — cloud metadata, link-local, and common cloud-provider
# private endpoints. Loopback and RFC1918 ranges are allowed (local-first app).
_SSRF_BLOCKED_HOSTS = {
    '169.254.169.254',   # AWS / Azure / GCP metadata
    '100.100.100.200',   # Alibaba Cloud metadata
    'metadata.google.internal',
    'metadata',
}


def _normalise_ollama_host(host: str) -> str:
    host = (host or '').strip().rstrip('/')
    if host and '://' not in host:
        host = 'http://' + host
    return host or 'http://localhost:11434'


def _validate_ollama_host(host: str) -> tuple[str | None, str | None]:
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


def _fetch_ollama_models(host: str) -> "tuple[list[dict[str, str | None]] | None, str | None]":
    safe_host, err = _validate_ollama_host(host)
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
        logger.warning(f"[REST API] Ollama model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch Ollama models"


def _fetch_openai_models(api_key: str) -> "tuple[list[dict[str, str | None]] | None, str | None]":
    """Filter to chat-capable text models only."""
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


def _fetch_anthropic_models(api_key: str) -> "tuple[list[dict[str, str | None]] | None, str | None]":
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


def _fetch_gemini_models(api_key: str) -> "tuple[list[dict[str, str | None]] | None, str | None]":
    """Cap pagination at 10 pages. Only ``generateContent`` models; strips ``models/`` prefix."""
    if not api_key:
        return None, "API key is required"
    try:
        models = []
        page_token = None
        for _ in range(_GEMINI_MAX_PAGES):
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


def _fetch_openai_compatible_models(host: str, api_key: str) -> "tuple[list[dict[str, str | None]] | None, str | None]":
    if not host:
        return None, "Host URL is required"
    if not api_key:
        return None, "API key is required"
    safe_host, err = _validate_ollama_host(host)
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
        models.sort(key=lambda m: cast(str, m['id']))
        return models, None
    except req.exceptions.ConnectionError:
        return None, f"Cannot connect to {safe_host}"
    except req.exceptions.Timeout:
        return None, f"Request to {safe_host} timed out"
    except Exception as e:
        logger.warning(f"[REST API] OpenAI-compatible model list failed: {type(e).__name__}: {e}")
        return None, "Failed to fetch models"


def _map_api_error(error_str: str, platform: str, model: str) -> str:
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
    logger.warning("[REST API] Unmapped upstream provider error for platform=%s model=%s: %s", platform, model, error_str)
    return "Upstream provider error"


def _test_ollama_provider(config: "dict[str, object]", model: str, start: float) -> ProviderTestResult:
    import time
    available, err = _fetch_ollama_models(cast(str, config.get('host', '')))
    latency_ms = int((time.time() - start) * 1000)

    if err is not None:
        return ProviderTestResult(success=False, error=err)

    available_names = [cast(str, m['id']) for m in (available or [])]
    model_base = model.split(':')[0]
    model_found = any(
        m == model or m.startswith(model + ':') or m.split(':')[0] == model_base
        for m in available_names
    )

    if not model_found and not available_names:
        return ProviderTestResult(
            success=True, model=model, latency_ms=latency_ms,
            message="Connected to Ollama (no models installed yet)",
        )

    if not model_found:
        return ProviderTestResult(
            success=False,
            error=f"Model '{model}' not found on this Ollama instance.",
            hint=f"Run: ollama pull {model}  ·  Available: {', '.join(available_names[:5])}",
        )

    return ProviderTestResult(
        success=True, model=model, latency_ms=latency_ms,
        message=f"Connected · {len(available_names)} model(s) available",
    )


def _test_api_provider(config: "dict[str, object]", platform: str, model: str, start: float) -> ProviderTestResult:
    import time
    api_key = config.get('api_key')
    if not api_key:
        return ProviderTestResult(
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
        return ProviderTestResult(success=True, model=model, latency_ms=latency_ms, message="Connected successfully")
    except Exception as e:
        return ProviderTestResult(success=False, error=_map_api_error(str(e), platform, model))


def get_provider_service() -> "ProviderDbService":
    from services.provider_db_service import ProviderDbService
    return ProviderDbService()


def _provider_dto(row: "dict[str, object]") -> Provider:
    """Build a Provider read DTO from a service dict, stripping the api_key secret."""
    return Provider(
        id=cast(int, row['id']),
        name=cast(str, row['name']),
        platform=cast(str, row['platform']),
        model=cast(str, row['model']),
        host=cast(_OptStr, row.get('host')),
        dimensions=cast(_OptInt, row.get('dimensions')),
        timeout=cast(_OptInt, row.get('timeout')),
        supports_vision=bool(row.get('supports_vision')),
        max_tokens=cast(_OptInt, row.get('max_tokens')),
    )


def _invalidate_cache() -> None:
    try:
        from services.provider_cache_service import ProviderCacheService
        ProviderCacheService.invalidate()
    except Exception as e:
        logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@providers_ns.route('')
class ProviderListResource(Resource):
    @require_session
    @providers_ns.response(200, "List of providers", model=[_M["Provider"]])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=200)
    def get(self) -> "list[Provider] | ResponseReturnValue":
        try:
            return [_provider_dto(r) for r in get_provider_service().list_providers_summary()]
        except Exception as e:
            logger.error(f"[REST API] Failed to list providers: {e}")
            return error("Failed to list providers", 500)

    @require_session
    @providers_ns.expect(_M["ProviderCreate"])
    @providers_ns.response(201, "Provider created", model=_M["Provider"])
    @providers_ns.response(400, "Invalid configuration", model=_M["Error"])
    @providers_ns.response(409, "Name conflict", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=201)
    @expects(ProviderCreate)
    def post(self, dto: ProviderCreate) -> "Provider | ResponseReturnValue":
        try:
            row = cast("dict[str, object]", get_provider_service().create_provider(dto.model_dump(mode="json", exclude_none=False)))
            _invalidate_cache()
            return _provider_dto(row)
        except ValueError as e:
            logger.warning(f"[REST API] Provider validation error: {e}")
            return error(_safe_validation_msg(e), 400)
        except sqlite3.IntegrityError as e:
            logger.warning(f"[REST API] Provider name conflict: {e}")
            return error(_DUPLICATE_NAME_MSG, 409)
        except Exception as e:
            logger.error(f"[REST API] Failed to create provider: {e}")
            return error("Failed to create provider", 500)


@providers_ns.route('/catalog')
class ProviderCatalogResource(Resource):
    @require_session
    @providers_ns.response(200, "Provider catalog")
    @providers_ns.response(500, "Server error", model=_M["Error"])
    def get(self) -> ResponseReturnValue:
        """Return the curated provider presets for the setup wizard."""
        try:
            from services.provider_catalog_service import get_catalog
            return {"catalog": get_catalog()}, 200
        except Exception as e:
            logger.exception(f"[REST API] Failed to load provider catalog: {e}")
            return error("Failed to load provider catalog", 500)


@providers_ns.route('/<int:provider_id>')
class ProviderResource(Resource):
    @require_session
    @providers_ns.param("provider_id", "Provider ID")
    @providers_ns.response(200, "Provider details", model=_M["Provider"])
    @providers_ns.response(404, "Provider not found", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=200)
    def get(self, provider_id: int) -> "Provider | ResponseReturnValue":
        try:
            row = get_provider_service().get_provider_by_id(provider_id)
            if not row:
                return error(_ERR_PROVIDER_NOT_FOUND, 404)
            return _provider_dto(row)
        except Exception as e:
            logger.error(f"[REST API] Failed to get provider: {e}")
            return error("Failed to get provider", 500)

    @require_session
    @providers_ns.param("provider_id", "Provider ID")
    @providers_ns.expect(_M["ProviderUpdate"])
    @providers_ns.response(200, "Updated provider", model=_M["Provider"])
    @providers_ns.response(400, "Invalid configuration", model=_M["Error"])
    @providers_ns.response(404, "Provider not found", model=_M["Error"])
    @providers_ns.response(409, "Name conflict", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=200)
    @expects(ProviderUpdate)
    def put(self, provider_id: int, dto: ProviderUpdate) -> "Provider | ResponseReturnValue":
        try:
            row = cast("dict[str, object]", get_provider_service().update_provider(provider_id, dto.model_dump(mode="json", exclude_unset=True)))
            _invalidate_cache()
            return _provider_dto(row)
        except ValueError as e:
            logger.warning(f"[REST API] Provider validation error: {e}")
            return error(_safe_validation_msg(e), 400)
        except sqlite3.IntegrityError as e:
            logger.warning(f"[REST API] Provider name conflict: {e}")
            return error(_DUPLICATE_NAME_MSG, 409)
        except Exception as e:
            logger.error(f"[REST API] Failed to update provider: {e}")
            return error("Failed to update provider", 500)

    @require_session
    @providers_ns.param("provider_id", "Provider ID")
    @providers_ns.response(204, "Deleted")
    @providers_ns.response(409, "Provider in use", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    def delete(self, provider_id: int) -> ResponseReturnValue:
        try:
            get_provider_service().delete_provider(provider_id)
            _invalidate_cache()
            return "", 204
        except ValueError as e:
            logger.warning(f"[REST API] Cannot delete provider {provider_id}: {e}")
            return error(_safe_validation_msg(e), 409)
        except Exception as e:
            logger.error(f"[REST API] Failed to delete provider: {e}")
            return error("Failed to delete provider", 500)


@providers_ns.route('/list-models')
class ProviderListModelsResource(Resource):
    @require_session
    @providers_ns.expect(_M["ListModelsRequest"])
    @providers_ns.response(200, "Model list", model=_M["ListModelsResult"])
    @providers_ns.response(400, "Unsupported platform", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(ListModelsResult, code=200)
    @expects(ListModelsRequest)
    def post(self, dto: ListModelsRequest) -> ListModelsResult | ResponseReturnValue:
        """List available models for a given platform."""
        platform = dto.platform.strip().lower()
        if platform == 'ollama':
            models, err = _fetch_ollama_models(dto.host or '')
        elif platform == 'openai':
            models, err = _fetch_openai_models((dto.api_key or '').strip())
        elif platform == 'anthropic':
            models, err = _fetch_anthropic_models((dto.api_key or '').strip())
        elif platform == 'gemini':
            models, err = _fetch_gemini_models((dto.api_key or '').strip())
        elif platform == 'openai_compatible':
            models, err = _fetch_openai_compatible_models(
                dto.host or '', (dto.api_key or '').strip(),
            )
        else:
            return ListModelsResult(models=[], error=f"Unsupported platform '{platform}'").model_dump(mode="json"), 400

        if err is not None:
            return ListModelsResult(models=[], error=err)
        return ListModelsResult(
            models=[ModelInfo(id=cast(str, m['id']), display_name=m.get('display_name')) for m in (models or [])],
        )


@providers_ns.route('/test')
class ProviderTestResource(Resource):
    @require_session
    @providers_ns.expect(_M["ProviderTestRequest"])
    @providers_ns.response(200, "Test result", model=_M["ProviderTestResult"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["ProviderTestResult"])
    @responds(ProviderTestResult, code=200)
    @expects(ProviderTestRequest)
    def post(self, dto: ProviderTestRequest) -> ProviderTestResult:
        import time
        try:
            config: dict[str, object] = {}
            if dto.provider_id is not None:
                service = get_provider_service()
                stored = service.get_provider_by_id(dto.provider_id)
                if not stored:
                    return ProviderTestResult(success=False, error=_ERR_PROVIDER_NOT_FOUND)
                config = {k: v for k, v in stored.items() if v is not None}

            for field in ('platform', 'model', 'host', 'api_key'):
                val = getattr(dto, field)
                if val:
                    config[field] = val

            platform = cast(_OptStr, config.get('platform'))
            model = cast(_OptStr, config.get('model'))

            if not platform:
                return ProviderTestResult(success=False, error="Platform is required")
            if not model:
                return ProviderTestResult(success=False, error="Model is required")

            start = time.time()
            if platform == 'ollama':
                return _test_ollama_provider(config, model, start)
            return _test_api_provider(config, platform, model, start)

        except Exception as e:
            logger.error(f"[REST API] Provider test failed unexpectedly: {e}")
            return ProviderTestResult(success=False, error="Test failed unexpectedly")


@providers_ns.route('/selected')
class ProviderSelectedResource(Resource):
    @require_session
    @providers_ns.response(200, "Selected provider", model=_M["Provider"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=200)
    def get(self) -> "Provider | None | ResponseReturnValue":
        try:
            row = get_provider_service().get_selected_provider()
            return _provider_dto(row) if row else None
        except Exception as e:
            logger.error(f"[REST API] Failed to get selected provider: {e}")
            return error("Failed to get selected provider", 500)

    @require_session
    @providers_ns.expect(_M["ProviderRef"])
    @providers_ns.response(200, "Updated selected provider", model=_M["Provider"])
    @providers_ns.response(404, "Provider not found", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(Provider, code=200)
    @expects(ProviderRef)
    def put(self, dto: ProviderRef) -> "Provider | ResponseReturnValue":
        try:
            service = get_provider_service()
            row = service.get_provider_by_id(dto.provider_id)
            if not row:
                return error(_ERR_PROVIDER_NOT_FOUND, 404)
            service.set_selected_provider(dto.provider_id)
            _invalidate_cache()
            return _provider_dto(row)
        except ValueError as e:
            logger.warning(f"[REST API] Provider validation error: {e}")
            return error(_safe_validation_msg(e), 400)
        except Exception as e:
            logger.error(f"[REST API] Failed to set selected provider: {e}")
            return error("Failed to set selected provider", 500)


@providers_ns.route('/vision')
class ProviderVisionResource(Resource):
    @require_session
    @providers_ns.response(200, "Vision provider", model=_M["ProviderRole"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(ProviderRole, code=200)
    def get(self) -> ProviderRole | ResponseReturnValue:
        try:
            status = get_provider_service().get_vision_provider_status()
            row = cast(_OptDictStrObj, status['provider'])
            return ProviderRole(
                provider=_provider_dto(row) if row else None,
                source=cast(str, status['source']),
            )
        except Exception:
            logger.exception("[REST API] Failed to get vision provider")
            return error("Failed to get vision provider", 500)

    @require_session
    @providers_ns.expect(_M["NullableProviderRef"])
    @providers_ns.response(200, "Updated vision provider", model=_M["ProviderRole"])
    @providers_ns.response(400, "Invalid request", model=_M["Error"])
    @providers_ns.response(404, "Provider not found", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(ProviderRole, code=200)
    @expects(NullableProviderRef)
    def put(self, dto: NullableProviderRef) -> ProviderRole | ResponseReturnValue:
        try:
            service = get_provider_service()
            if dto.provider_id is None:
                service.set_vision_provider(None)
                return ProviderRole(provider=None, source='none')

            row = service.get_provider_by_id(dto.provider_id)
            if not row:
                return error(_ERR_PROVIDER_NOT_FOUND, 404)
            if not row.get('supports_vision'):
                return error("Provider does not support vision", 400)

            service.set_vision_provider(dto.provider_id)
            _invalidate_cache()
            return ProviderRole(provider=_provider_dto(row), source='explicit')
        except (ValueError, TypeError):
            return error("Invalid provider_id", 400)
        except Exception:
            logger.exception("[REST API] Failed to set vision provider")
            return error("Failed to set vision provider", 500)


@providers_ns.route('/delegate')
class ProviderDelegateResource(Resource):
    @require_session
    @providers_ns.response(200, "Delegate provider", model=_M["ProviderRole"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(ProviderRole, code=200)
    def get(self) -> ProviderRole | ResponseReturnValue:
        """Return the configured delegate provider + resolution source."""
        try:
            status = get_provider_service().get_delegate_provider_status()
            row = cast(_OptDictStrObj, status['provider'])
            return ProviderRole(
                provider=_provider_dto(row) if row else None,
                source=cast(str, status['source']),
            )
        except Exception:
            logger.exception("[REST API] Failed to get delegate provider")
            return error("Failed to get delegate provider", 500)

    @require_session
    @providers_ns.expect(_M["NullableProviderRef"])
    @providers_ns.response(200, "Updated delegate provider", model=_M["ProviderRole"])
    @providers_ns.response(400, "Invalid request", model=_M["Error"])
    @providers_ns.response(404, "Provider not found", model=_M["Error"])
    @providers_ns.response(422, "Validation failed", model=_M["Error"])
    @providers_ns.response(500, "Server error", model=_M["Error"])
    @responds(ProviderRole, code=200)
    @expects(NullableProviderRef)
    def put(self, dto: NullableProviderRef) -> ProviderRole | ResponseReturnValue:
        """Set or clear the delegate provider ID."""
        try:
            service = get_provider_service()
            if dto.provider_id is None:
                service.set_delegate_provider(None)
                status = service.get_delegate_provider_status()
                row = cast(_OptDictStrObj, status['provider'])
                return ProviderRole(
                    provider=_provider_dto(row) if row else None,
                    source=cast(str, status['source']),
                )

            row = service.get_provider_by_id(dto.provider_id)
            if not row:
                return error(_ERR_PROVIDER_NOT_FOUND, 404)

            service.set_delegate_provider(dto.provider_id)
            _invalidate_cache()
            return ProviderRole(provider=_provider_dto(row), source='explicit')
        except (ValueError, TypeError):
            return error("Invalid provider_id", 400)
        except Exception:
            logger.exception("[REST API] Failed to set delegate provider")
            return error("Failed to set delegate provider", 500)
