"""Swagger bridge — register Pydantic DTOs into the flask-restx ``Api``.

flask-restx serves OpenAPI 2.0, whose ``$ref`` targets live under
``#/definitions/{model}``. Pydantic v2 emits JSON Schema 2020-12 with nested
models under ``$defs``; :func:`register_dto` lifts those into separately
registered top-level definitions so every ``$ref`` resolves, then registers the
DTO itself under its class name.

Attach a registered model to a route with the one-liner the namespaces reuse::

    register_dto(api, ListItem, ListItemCreate)
    @ns.response(200, "The item", model=api.models["ListItem"])
    @ns.expect(api.models["ListItemCreate"])
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from .base import DTO


class _SchemaRegistrar(Protocol):
    """Structural type for the flask-restx ``Api`` model-registration surface."""

    def schema_model(self, name: str, schema: dict[str, object]) -> object: ...


def _to_swagger_schema(cls: type[BaseModel]) -> dict[str, object]:
    """Render a Pydantic model as an OpenAPI 2.0 schema with definition-scoped ``$ref``s."""
    return cls.model_json_schema(ref_template="#/definitions/{model}")


def register_dto(api: _SchemaRegistrar, *dto_classes: type[DTO]) -> None:
    """Register each DTO and its nested models as top-level flask-restx schema models."""
    for cls in dto_classes:
        schema = _to_swagger_schema(cls)
        defs = schema.pop("$defs", {})
        if isinstance(defs, dict):
            for def_name, def_schema in defs.items():
                api.schema_model(def_name, def_schema)
        api.schema_model(cls.__name__, schema)
