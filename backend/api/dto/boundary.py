"""HTTP-boundary decorators that parse requests into DTOs and serialize responses.

``@expects(DTO)`` validates the inbound request and hands the handler a typed
``dto``; ``@responds(DTO, code=...)`` writes the handler's DTO return back as
JSON. Both preserve the wrapped Resource-method signature (``ParamSpec``) and
clear ``mypy --strict`` with zero type-suppression comments.

Usage::

    @ns.route("/<list_id>")
    class ListResource(Resource):
        @responds(List)
        @expects(ListUpdate)
        def put(self, list_id: str, dto: ListUpdate) -> List:
            ...

Multipart uploads validate through the :data:`File` annotated type on a DTO
field when ``expects(..., source="form")`` is used.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Literal, ParamSpec, TypeVar, cast

from flask import Response, request
from flask.typing import ResponseReturnValue
from pydantic import GetCoreSchemaHandler, ValidationError
from pydantic_core import CoreSchema, core_schema
from pydantic.json_schema import JsonSchemaValue
from werkzeug.datastructures import FileStorage

from .base import DTO
from .error import Error

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=DTO)

Source = Literal["json", "args", "form"]
"""Inbound request source for :func:`expects`."""

_VALIDATION_FAILED = "Validation failed"


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------


def _read_payload(source: Source) -> object:
    """Materialize the request payload for the given ``source``."""
    if source == "json":
        return request.get_json(silent=True) or {}
    if source == "args":
        # Single-valued params collapse to scalars (ergonomic for int/str fields);
        # repeated params stay lists (for sequence fields).
        return {
            key: (values[0] if len(values) == 1 else list(values))
            for key, values in request.args.to_dict(flat=False).items()
        }
    if source == "form":
        return {**request.form, **request.files}
    raise ValueError(f"unknown expects source: {source!r}")


def _validation_body(exc: ValidationError) -> tuple[dict[str, object], int]:
    """Build the uniform 422 ``Error`` body for a validation failure."""
    return (
        Error(
            error=_VALIDATION_FAILED,
            details=cast("list[dict[str, object]]", exc.errors(include_url=False)),
        ).model_dump(mode="json"),
        422,
    )


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def expects(
    dto_cls: type[T],
    *,
    source: Source = "json",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate the inbound request into a DTO, injected as the handler's ``dto`` kwarg.

    On :class:`pydantic.ValidationError` returns a uniform ``Error`` body with
    HTTP 422. Path parameters stay positional (``def put(self, list_id, dto)``).
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                dto = dto_cls.model_validate(_read_payload(source))
            except ValidationError as exc:
                return cast("R", _validation_body(exc))
            return cast("Callable[..., R]", func)(*args, **kwargs, dto=dto)

        return wrapper

    return decorator


def responds(
    dto_cls: type[DTO] | None = None,
    *,
    code: int = 200,
) -> Callable[[Callable[P, object]], Callable[P, ResponseReturnValue]]:
    """Serialize the handler return into a JSON response.

    Accepts a single :class:`DTO`, a list of DTOs, a werkzeug :class:`Response`
    (passed through untouched), or ``None`` (empty body). ``dto_cls`` declares
    the response schema for the swagger bridge; serialization is driven by the
    runtime type of the return value.
    """

    def decorator(func: Callable[P, object]) -> Callable[P, ResponseReturnValue]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> ResponseReturnValue:
            return _dump(func(*args, **kwargs), code)

        return wrapper

    return decorator


def _dump(result: object, code: int) -> ResponseReturnValue:
    """Serialize a handler return value into a Flask response return value."""
    if isinstance(result, Response):
        return result
    if isinstance(result, DTO):
        return result.model_dump(mode="json"), code
    if isinstance(result, list):
        return [_dump_value(item) for item in result], code
    if result is None:
        return "", code
    if isinstance(result, (str, bytes, dict)):
        return result, code
    # A pre-shaped (body, status) tuple — e.g. an Error handed up by @expects —
    # is returned verbatim so its status code is preserved.
    return cast("ResponseReturnValue", result)


def _dump_value(value: object) -> object:
    """Serialize one element of a list return: DTOs dump, everything else passes through."""
    return value.model_dump(mode="json") if isinstance(value, DTO) else value


# ---------------------------------------------------------------------------
# FileStorage annotated type (multipart uploads)
# ---------------------------------------------------------------------------


def _validate_file(value: object) -> FileStorage:
    if not isinstance(value, FileStorage):
        raise TypeError("expected an uploaded file (werkzeug FileStorage)")
    return value


class _FileStorageAnnotation:
    """Pydantic core-schema + json-schema hooks so :data:`File` validates and renders in any strict DTO."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(_validate_file)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetCoreSchemaHandler,
    ) -> JsonSchemaValue:
        # Render as a binary string in OpenAPI — a multipart file upload is not a
        # JSON-encodable value, but the schema must still resolve so register_dto
        # can lift an UploadRequest-shaped DTO into the swagger definitions.
        return {"type": "string", "format": "binary"}


File = Annotated[FileStorage, _FileStorageAnnotation]
"""DTO field type for a single multipart file upload; works without ``arbitrary_types_allowed``."""
