"""MCP server regenerate-token action — mint a fresh bearer token.

Covers:
- POST /api/mcp-server/regenerate-token  → post (revoke old, mint new)

This is the only way to obtain the raw bearer token; it cannot be recovered
server-side after this point. The old wrapper token is revoked before the new
one is created to prevent overlap.
"""

from __future__ import annotations

import uuid
from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.response.mcp_settings import RegenerateTokenResult
from models.setting import Setting
from services.wrapper_auth_service import WrapperAuthService


class RegenerateTokenAction(Action):
    """Action for regenerating the MCP server bearer token.

    Mint a fresh credential, revoking the prior one if present. The raw token
    is returned in the response (preserved current behaviour); it cannot be
    recovered server-side after this point.
    """

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    # Genuinely mints a credential resource — the 201 is real.
    _post_may_create: ClassVar[bool] = True
    response_dto = {
        "post": DocumentedResponse(RegenerateTokenResult),
    }

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Mint a fresh MCP server bearer token, revoking the prior one if present."""
        auth_svc = WrapperAuthService()

        old_wrapper_id = Setting.get_value("mcp_server_token_wrapper_id")
        if old_wrapper_id:
            auth_svc.revoke(old_wrapper_id)

        raw_token, wrapper_id = auth_svc.create_token(
            name="MCP Server (External Agents)",
            wrapper_id_override=f"__mcp_server_{uuid.uuid4().hex[:8]}__",
        )

        Setting.set("mcp_server_token_wrapper_id", wrapper_id)
        Setting.set("mcp_server_token", raw_token)

        return RegenerateTokenResult(token=raw_token, wrapper_id=wrapper_id).single(), 201
