# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MCP server built on the SDK's ``MCPServer`` — single ``talk_to_chalie`` tool for external agents.

Streamable HTTP on a dedicated port (default 8462). Bearer tokens validated by
ASGI middleware against ``wrapper_tokens`` (same as the REST API).
"""

import asyncio
import contextvars
import logging
import re
import sqlite3

import uvicorn
from mcp.server import MCPServer
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from configs.channels import EAMPConfig
from exceptions import ProviderRetriesExhaustedError
from services.database import Database

logger = logging.getLogger(__name__)

_MCP_SERVER_NAME = "chalie"
_DEFAULT_PORT = 8462
_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9 _\-\.]{1,100}$')

_current_wrapper_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'mcp_current_wrapper_id', default=None
)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that validates Bearer tokens against wrapper_tokens."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        raw_token = auth_header[len("Bearer "):]
        if not raw_token:
            return JSONResponse(
                {"error": "Empty bearer token"},
                status_code=401,
            )

        try:
            wrapper_id = await asyncio.to_thread(self._validate_token, raw_token)
        except Exception:
            # Token validation is the mandatory gate on EVERY MCP request, and it
            # reads the database. When that read fails the server is up but cannot
            # serve anyone — a state the thread-liveness supervisor cannot see.
            # Without this branch the traceback escapes into the ASGI server's own
            # stdout logger, never reaching the app log, so the outage is silent.
            # Log it loudly here and answer 503 instead of an opaque 500: the MCP
            # server is not the broken component, so restarting it would not help.
            logger.exception(
                "[MCP] Token validation failed — MCP server is running but cannot "
                "serve requests; the underlying store is unreachable"
            )
            return JSONResponse(
                {"error": "MCP server temporarily unavailable"},
                status_code=503,
            )

        if wrapper_id is None:
            return JSONResponse(
                {"error": "Invalid or revoked token"},
                status_code=401,
            )

        ctx_token = _current_wrapper_id.set(wrapper_id)
        try:
            response = await call_next(request)
        finally:
            _current_wrapper_id.reset(ctx_token)
        return response

    @staticmethod
    def _validate_token(raw_token: str) -> str | None:
        from services.wrapper_auth_service import _hash_token
        from services.time_utils import utc_now

        token_hash = _hash_token(raw_token)

        with Database.transaction() as conn:
            row = conn.execute(
                "SELECT wrapper_id FROM wrapper_tokens "
                "WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            ).fetchone()

        if row is None:
            return None

        wrapper_id: str = row[0] if isinstance(row, (tuple, list)) else row["wrapper_id"]

        now = utc_now().isoformat()
        with Database.transaction() as conn:
            conn.execute(
                "UPDATE wrapper_tokens SET last_seen_at = ? WHERE wrapper_id = ?",
                (now, wrapper_id),
            )

        return wrapper_id


def create_mcp_server() -> MCPServer:
    """Create and configure the MCP server with the talk_to_chalie tool.

    Bind host and port are not set here — the constructor does not take them
    (they are applied where they matter: ``host`` in ``_build_app`` and ``port``
    by ``uvicorn.run`` in ``run_mcp_server``).
    """
    mcp = MCPServer(name=_MCP_SERVER_NAME)

    @mcp.tool()
    async def talk_to_chalie(
        message: str,
        agent_name: str,
        project_or_task_name: str,
        loop_in_human: bool = False,
    ) -> str:
        errors = []
        if not message or not message.strip():
            errors.append("'message' is required and cannot be empty.")
        if not agent_name:
            errors.append("'agent_name' is required.")
        elif not _SAFE_NAME_RE.match(agent_name):
            errors.append(
                f"'agent_name' is invalid (got {len(agent_name)} chars). "
                "Must be 1-100 characters, only letters, numbers, spaces, hyphens, underscores, and dots."
            )
        if not project_or_task_name:
            errors.append("'project_or_task_name' is required.")
        elif not _SAFE_NAME_RE.match(project_or_task_name):
            errors.append(
                f"'project_or_task_name' is invalid (got {len(project_or_task_name)} chars). "
                "Must be 1-100 characters, only letters, numbers, spaces, hyphens, underscores, and dots."
            )
        if errors:
            return "Invalid parameters:\n" + "\n".join(f"- {e}" for e in errors)

        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        wrapper_id = _current_wrapper_id.get()
        logger.info(
            "[MCP] talk_to_chalie: agent=%s project=%s loop_in_human=%s wrapper=%s",
            agent_name, project_or_task_name, loop_in_human, wrapper_id,
        )


        def _run() -> str:
            config = EAMPConfig(
                agent_name=agent_name,
                project=project_or_task_name,
                loop_in_human=loop_in_human,
            )
            return MessageProcessor.process(config, raw_input=message).result()

        try:
            response = await asyncio.to_thread(_run)
        except ProviderRetriesExhaustedError as exc:
            return str(exc)
        return response or "(No response generated)"

    return mcp


def _build_app(mcp: MCPServer) -> Starlette:
    """Wrap the MCP Starlette app with bearer token auth middleware."""
    # host="0.0.0.0" is load-bearing, not cosmetic: the SDK auto-enables
    # DNS-rebinding protection (allowed_hosts locked to 127.0.0.1/localhost/[::1])
    # ONLY when this host is a loopback address. Passing the real bind host keeps
    # that protection OFF, so networked external agents can connect — the
    # permissive transport 1.x always ran. Do not "simplify" this back to the
    # default 127.0.0.1.
    app: Starlette = mcp.streamable_http_app(host="0.0.0.0")
    app.add_middleware(BearerTokenMiddleware)
    return app


def run_mcp_server() -> None:
    """Run the MCP server (blocking). Intended as a WorkerManager service."""
    from models.setting import Setting

    enabled = Setting.get_value("mcp_server_enabled")
    if enabled is not None and str(enabled).lower() in ("false", "0", "no"):
        logger.info("[MCP] Server disabled via settings (mcp_server_enabled=false)")
        return

    port_setting = Setting.get_value("mcp_server_port")
    try:
        port = int(port_setting) if port_setting else _DEFAULT_PORT
    except (ValueError, TypeError):
        port = _DEFAULT_PORT

    _ensure_mcp_token()

    logger.info("[MCP] Starting MCP server on port %d", port)
    mcp = create_mcp_server()
    app = _build_app(mcp)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _ensure_mcp_token() -> None:
    """Generate an MCP auth token on first boot if none exists."""
    from services.wrapper_auth_service import WrapperAuthService
    from models.setting import Setting

    existing = Setting.get_value("mcp_server_token_wrapper_id")
    if existing:
        auth_svc = WrapperAuthService()
        wrapper = auth_svc.get_wrapper(existing)
        if wrapper:
            return

    auth_svc = WrapperAuthService()
    try:
        raw_token, wrapper_id = auth_svc.create_token(
            name="MCP Server (External Agents)",
            wrapper_id_override="__mcp_server__",
        )
    except sqlite3.IntegrityError:
        logger.info("[MCP] Token already exists (concurrent boot); skipping")
        return

    Setting.set("mcp_server_token_wrapper_id", wrapper_id)

    logger.info(
        "[MCP] Generated MCP auth token (wrapper_id=%s). "
        "Retrieve via: Settings > MCP Server in the brain dashboard.",
        wrapper_id,
    )
    logger.info("[MCP] Token (shown once): %s", raw_token)
