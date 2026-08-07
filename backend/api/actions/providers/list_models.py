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
from api.endpoint import DocumentedResponse
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.provider_models import ListModelsRequest
from api.response.provider_models import ListModelsResult, ModelInfo
from services.llm_clients.registry import Registry


class ProviderListModels(Action):
    """Action fetching the live model list for a platform+credentials pair."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    request_dto: ClassVar[type[Request] | None] = ListModelsRequest
    response_dto = {"post": DocumentedResponse(ListModelsResult)}

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(ListModelsRequest, data)
        platform = dto.platform.strip().lower()

        # Each client knows how its own vendor lists models, so a new provider
        # is listable the moment its module exists — there is no branch here to
        # forget to extend.
        client_class = Registry.client_class_for(platform)
        if client_class is None:
            raise EndpointError(f"Unsupported platform '{platform}'")

        models, err = client_class.fetch_models(
            dto.host or '', (dto.api_key or '').strip(),
        )

        if err is not None:
            return ListModelsResult(models=[], error=err).single()
        return ListModelsResult(
            models=[
                ModelInfo(id=cast(str, m['id']), display_name=m.get('display_name'))
                for m in (models or [])
            ],
        ).single()
