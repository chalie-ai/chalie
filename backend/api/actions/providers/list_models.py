"""Provider model-listing action — nested resource under /api/providers/list-models.

Covers:
- POST /api/providers/list-models  → post (live model-list fetch per platform)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
matching the discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
An unsupported platform raises (400) rather than the legacy inline ``model_dump``
return — the uniform error envelope, same as every other migrated endpoint.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, EndpointError, NotFoundError
from api.request import Request
from api.request.provider_models import ListModelsRequest
from api.response.provider_models import ListModelsResult, ModelInfo
from services.provider_probe import (
    fetch_anthropic_models,
    fetch_codex_models,
    fetch_gemini_models,
    fetch_ollama_models,
    fetch_openai_compatible_models,
    fetch_openai_models,
)


class ProviderListModels(Action):
    """Action fetching the live model list for a platform+credentials pair."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = ListModelsRequest
    response_dto = {"post": DocumentedResponse(ListModelsResult)}

    def slug(self) -> str:
        return "providers"

    def verb(self) -> str:
        return "list-models"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(ListModelsRequest, data)
        platform = dto.platform.strip().lower()

        models: list[dict[str, str | None]] | None
        err: str | None
        if platform == 'ollama':
            models, err = fetch_ollama_models(dto.host or '')
        elif platform == 'openai':
            models, err = fetch_openai_models((dto.api_key or '').strip())
        elif platform == 'anthropic':
            models, err = fetch_anthropic_models((dto.api_key or '').strip())
        elif platform == 'gemini':
            models, err = fetch_gemini_models((dto.api_key or '').strip())
        elif platform == 'openai_compatible':
            models, err = fetch_openai_compatible_models(dto.host or '', (dto.api_key or '').strip())
        elif platform == 'codex_cli':
            models, err = fetch_codex_models()
        else:
            raise EndpointError(f"Unsupported platform '{platform}'")

        if err is not None:
            return ListModelsResult(models=[], error=err).single()
        return ListModelsResult(
            models=[
                ModelInfo(id=cast(str, m['id']), display_name=m.get('display_name'))
                for m in (models or [])
            ],
        ).single()
