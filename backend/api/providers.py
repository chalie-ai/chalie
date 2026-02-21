"""
Providers blueprint — manage LLM provider configuration via REST API.
"""

import logging
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

providers_bp = Blueprint('providers', __name__, url_prefix='/providers')


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

        required = ["name", "platform", "model"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        service = get_provider_service()

        # Check if this is the first provider (before creation)
        existing_providers = service.list_providers_summary()
        is_first_provider = len(existing_providers) == 0

        provider = service.create_provider(data)

        # Auto-assign all 13 jobs if this is the first provider
        if is_first_provider:
            all_jobs = [
                'frontal-cortex', 'frontal-cortex-respond', 'frontal-cortex-clarify',
                'frontal-cortex-acknowledge', 'frontal-cortex-act', 'frontal-cortex-proactive',
                'memory-chunker', 'episodic-memory', 'semantic-memory',
                'mode-tiebreaker', 'mode-reflection', 'cognitive-drift', 'experience-assimilation'
            ]
            try:
                for job in all_jobs:
                    service.set_job_assignment(job, provider["id"])
            except Exception as e:
                logger.error(f"[REST API] Failed to auto-assign jobs to first provider: {e}")

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
            return jsonify({"error": "Provider not found"}), 404

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


@providers_bp.route('/test', methods=['POST'])
@require_session
def test_provider():
    """Test a provider connection with a lightweight call."""
    import time
    import requests as req

    try:
        data = request.get_json() or {}
        provider_id = data.get('provider_id')

        # Start from stored provider config if an ID is given
        config = {}
        if provider_id:
            service = get_provider_service()
            stored = service.get_provider_by_id(int(provider_id))
            if not stored:
                return jsonify({"success": False, "error": "Provider not found"}), 200
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
            host = config.get('host', 'http://localhost:11434')
            try:
                r = req.get(f"{host}/api/tags", timeout=5)
                r.raise_for_status()
                models_data = r.json()
                available = [m.get('name') or m.get('model', '') for m in (models_data.get('models') or [])]
                latency_ms = int((time.time() - start) * 1000)

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

            except req.exceptions.ConnectionError:
                return jsonify({
                    "success": False,
                    "error": f"Cannot connect to Ollama at {host}",
                    "hint": "Make sure Ollama is running: ollama serve"
                }), 200
            except req.exceptions.Timeout:
                return jsonify({
                    "success": False,
                    "error": "Connection timed out",
                    "hint": f"Check that {host} is reachable"
                }), 200
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 200

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
                    'timeout': 15,
                }
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
                hint = None
                if any(k in el for k in ('authentication', 'auth_token', 'api_key', 'invalid_api', '401', 'unauthorized', 'invalid x-api-key')):
                    hint = "Your API key appears to be invalid or expired"
                elif any(k in el for k in ('model_not_found', 'not found', 'does not exist', 'no such model', '404')):
                    hint = f"Model '{model}' may not be available — check the model name"
                elif any(k in el for k in ('quota', 'rate_limit', 'rate limit', '429', 'too many')):
                    hint = "API quota exceeded or rate limited — try again later"
                elif any(k in el for k in ('connect', 'timeout', 'network', 'ssl')):
                    hint = "Network error — check your internet connection"
                return jsonify({
                    "success": False,
                    "error": error_str[:300],
                    "hint": hint
                }), 200

    except Exception as e:
        logger.error(f"[REST API] Provider test failed unexpectedly: {e}")
        return jsonify({"success": False, "error": "Test failed unexpectedly"}), 500


@providers_bp.route('/jobs', methods=['GET'])
@require_session
def list_job_assignments():
    """List all job→provider assignments."""
    try:
        service = get_provider_service()
        assignments = service.get_all_job_assignments()
        return jsonify({"assignments": assignments}), 200
    except Exception as e:
        logger.error(f"[REST API] Failed to list job assignments: {e}")
        return jsonify({"error": "Failed to list job assignments"}), 500


@providers_bp.route('/jobs/<job_name>', methods=['PUT'])
@require_session
def assign_job_provider(job_name):
    """Assign a provider to a job."""
    try:
        data = request.get_json()
        if not data or "provider_id" not in data:
            return jsonify({"error": "Request body must contain 'provider_id'"}), 400

        provider_id = data["provider_id"]
        service = get_provider_service()
        assignment = service.set_job_assignment(job_name, provider_id)

        # Invalidate provider cache so workers pick up the new assignment
        try:
            from services.provider_cache_service import ProviderCacheService
            ProviderCacheService.invalidate()
        except Exception as e:
            logger.warning(f"[REST API] Failed to invalidate provider cache: {e}")

        return jsonify({"assignment": assignment}), 200
    except ValueError as e:
        logger.warning(f"[REST API] Provider validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[REST API] Failed to assign job provider: {e}")
        return jsonify({"error": "Failed to assign job provider"}), 500
