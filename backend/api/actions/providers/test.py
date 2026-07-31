"""Provider connectivity-test action — nested resource under /api/providers/test.

Covers:
- POST /api/providers/test  → post (live connectivity probe: a stored provider,
  inline override fields, or both layered together)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
matching the discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
Every branch — including an unknown ``provider_id`` and internal probe
failures — returns 200 with a failure payload rather than a non-2xx status,
matching the legacy resource exactly (the test result carries its own
success/error fields; the HTTP layer only fails on malformed input).
"""

from __future__ import annotations

import time
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.request import Request
from api.request.provider_models import ProviderTestRequest
from api.response.provider_models import ProviderTestResult
from services.provider_db_service import ProviderDbService
from services.provider_probe import test_api_provider, test_codex_provider, test_ollama_provider


class ProviderTest(Action):
    """Action running a live connectivity test against a provider config."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = ProviderTestRequest
    response_dto = {"post": DocumentedResponse(ProviderTestResult)}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(ProviderTestRequest, data)

        config: dict[str, object] = {}
        if dto.provider_id is not None:
            stored = ProviderDbService().get_provider_by_id(dto.provider_id)
            if not stored:
                return ProviderTestResult(success=False, error="Provider not found").single()
            config = {k: v for k, v in stored.items() if v is not None}

        for field in ('platform', 'model', 'host', 'api_key'):
            val = getattr(dto, field)
            if val:
                config[field] = val

        platform = cast("str | None", config.get('platform'))
        model = cast("str | None", config.get('model'))

        if not platform:
            return ProviderTestResult(success=False, error="Platform is required").single()
        if not model:
            return ProviderTestResult(success=False, error="Model is required").single()
        start = time.time()
        if platform == 'ollama':
            outcome = test_ollama_provider(cast(str, config.get('host', '')), model, start)
        elif platform == 'codex_cli':
            outcome = test_codex_provider(model, start)
        else:
            outcome = test_api_provider(cast(str | None, config.get('api_key')), cast(str | None, config.get('host')), platform, model, start)

        return ProviderTestResult(
            success=outcome.success,
            model=outcome.model,
            latency_ms=outcome.latency_ms,
            message=outcome.message,
            error=outcome.error,
            hint=outcome.hint,
        ).single()
