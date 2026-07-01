"""Wrappers namespace — external bearer-token management.

Tokens are minted via POST and shown once; the raw secret never appears on any
read shape. All endpoints require an authenticated session; creation is
cookie-only (never a bearer).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.log_utils import safe
from .auth import require_auth, _cookie_only
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.wrapper import Wrapper, WrapperCreate, WrapperCreated

if TYPE_CHECKING:
    from services.wrapper_auth_service import WrapperAuthService

logger = logging.getLogger(__name__)

wrappers_ns = Namespace("wrappers", description="Wrapper token management", path="/api/wrappers")

register_dto(wrappers_ns, Wrapper, WrapperCreate, WrapperCreated, Error)

_M = wrappers_ns.models

_NOT_FOUND = "Wrapper not found"


def _get_service() -> "WrapperAuthService":
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _wrapper_dto(row: dict[str, object]) -> Wrapper:
    return Wrapper(
        id=cast(str, row["id"]),
        wrapper_id=cast(str, row["wrapper_id"]),
        name=cast(str, row["name"]),
        metadata=cast("dict[str, object]", row["metadata"]),
        last_seen_at=cast("datetime | None", row["last_seen_at"]),
        created_at=cast(datetime, row["created_at"]),
    )


# ---------------------------------------------------------------------------
# /api/wrappers — token collection
# ---------------------------------------------------------------------------

@wrappers_ns.route("")
class WrapperRootResource(Resource):
    @require_auth
    @_cookie_only
    @wrappers_ns.expect(_M["WrapperCreate"])
    @wrappers_ns.response(201, "Wrapper created", model=_M["WrapperCreated"])
    @wrappers_ns.response(422, "Validation failed", model=_M["Error"])
    @responds(WrapperCreated, code=201)
    @expects(WrapperCreate)
    def post(self, dto: WrapperCreate) -> WrapperCreated | ResponseReturnValue:
        """Mint a new external wrapper token; the raw token is shown only here."""
        raw_token, wrapper_id = _get_service().create_token(
            name=dto.name,
            metadata=dto.metadata,
        )
        logger.info("[Wrappers API] Created wrapper token: %s name=%r", wrapper_id, dto.name)
        return WrapperCreated(wrapper_id=wrapper_id, token=raw_token)

    @require_auth
    @wrappers_ns.response(200, "All wrappers", model=[_M["Wrapper"]])
    @responds(Wrapper, code=200)
    def get(self) -> list[Wrapper]:
        """List every non-revoked wrapper token."""
        return [_wrapper_dto(row) for row in _get_service().list_wrappers()]


# ---------------------------------------------------------------------------
# /api/wrappers/<wrapper_id> — single token
# ---------------------------------------------------------------------------

@wrappers_ns.route("/<wrapper_id>")
class WrapperItemResource(Resource):
    @require_auth
    @wrappers_ns.param("wrapper_id", "Wrapper identifier")
    @wrappers_ns.response(200, "Wrapper details", model=_M["Wrapper"])
    @wrappers_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(Wrapper, code=200)
    def get(self, wrapper_id: str) -> Wrapper | ResponseReturnValue:
        """Get details for a single wrapper token."""
        wrapper = _get_service().get_wrapper(wrapper_id)
        if wrapper is None:
            return error(_NOT_FOUND, 404)
        return _wrapper_dto(wrapper)

    @require_auth
    @wrappers_ns.param("wrapper_id", "Wrapper identifier")
    @wrappers_ns.response(204, "Revoked")
    @wrappers_ns.response(404, _NOT_FOUND, model=_M["Error"])
    @responds(code=204)
    def delete(self, wrapper_id: str) -> None | ResponseReturnValue:
        """Revoke a wrapper token."""
        if not _get_service().revoke(wrapper_id):
            return error(_NOT_FOUND, 404)
        logger.info("[Wrappers API] Revoked wrapper: %s", safe(wrapper_id))
        return None
