# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MCP Server implementation using FastMCP.

Exposes a single tool ``talk_to_chalie`` that external agents use to
communicate. Auth is validated via bearer token against the wrapper_tokens
table (same pattern as the dashboard gateway).

Transport: Streamable HTTP on a dedicated port (default 8462).
Auth: Custom ASGI middleware validates Bearer tokens before requests
reach the MCP protocol layer.
"""

import asyncio
import logging
import re
import sqlite3

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

_MCP_SERVER_NAME = "chalie"
_DEFAULT_PORT = 8462
_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9 _\-\.]{1,100}$')


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that validates Bearer tokens against wrapper_tokens."""

    async def dispatch(self, request: Request, call_next):
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

        wrapper_id = await asyncio.to_thread(self._validate_token, raw_token)
        if wrapper_id is None:
            return JSONResponse(
                {"error": "Invalid or revoked token"},
                status_code=401,
            )

        response = await call_next(request)
        return response

    @staticmethod
    def _validate_token(raw_token: str) -> str | None:
        from services.wrapper_auth_service import _hash_token
        from services.database_service import get_shared_db_service
        from services.time_utils import utc_now

        token_hash = _hash_token(raw_token)
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT wrapper_id FROM wrapper_tokens "
                "WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            )
            row = cursor.fetchone()
            cursor.close()

        if row is None:
            return None

        wrapper_id = row[0] if isinstance(row, (tuple, list)) else row["wrapper_id"]

        now = utc_now().isoformat()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE wrapper_tokens SET last_seen_at = ? WHERE wrapper_id = ?",
                (now, wrapper_id),
            )
            cursor.close()

        return wrapper_id


def create_mcp_server(host: str = "0.0.0.0", port: int = _DEFAULT_PORT) -> FastMCP:
    """Create and configure the MCP server with the talk_to_chalie tool."""
    mcp = FastMCP(
        name=_MCP_SERVER_NAME,
        host=host,
        port=port,
    )

    @mcp.tool()
    async def talk_to_chalie(
        message: str,
        agent_name: str,
        project_or_task_name: str,
        loop_in_human: bool = False,
    ) -> str:
        """Communicate with Chalie — your executive assistant.

        Send messages, share updates, ask questions, or request actions.
        Chalie has full access to memory, documents, schedules, and more.

        Args:
            message: What you want to say or ask Chalie.
            agent_name: Your identity (e.g. "Claude Code", "Codex", "CI Bot").
            project_or_task_name: The project or task context for this conversation.
            loop_in_human: If true, Chalie will disclose this exchange to the user.

        Returns:
            Chalie's response.
        """
        if not _SAFE_NAME_RE.match(agent_name):
            return "Error: agent_name must be 1-100 chars of [A-Za-z0-9 _\\-.]"
        if not _SAFE_NAME_RE.match(project_or_task_name):
            return "Error: project_or_task_name must be 1-100 chars of [A-Za-z0-9 _\\-.]"

        from services.external_agent_message_processor import ExternalAgentMessageProcessor

        logger.info(
            "[MCP] talk_to_chalie: agent=%s project=%s loop_in_human=%s",
            agent_name, project_or_task_name, loop_in_human,
        )

        def _run():
            proc = ExternalAgentMessageProcessor(
                raw_input=message,
                agent_name=agent_name,
                project_or_task_name=project_or_task_name,
                loop_in_human=loop_in_human,
            )
            return proc.send()

        response = await asyncio.to_thread(_run)
        return response or "(No response generated)"

    return mcp


def _build_app(mcp: FastMCP) -> Starlette:
    """Wrap the MCP Starlette app with bearer token auth middleware."""
    app = mcp.streamable_http_app()
    app.add_middleware(BearerTokenMiddleware)
    return app


def run_mcp_server() -> None:
    """Run the MCP server (blocking). Intended as a WorkerManager service."""
    from services.settings_service import SettingsService
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    settings = SettingsService(db)

    enabled = settings.get("mcp_server_enabled")
    if enabled is not None and str(enabled).lower() in ("false", "0", "no"):
        logger.info("[MCP] Server disabled via settings (mcp_server_enabled=false)")
        return

    port_setting = settings.get("mcp_server_port")
    try:
        port = int(port_setting) if port_setting else _DEFAULT_PORT
    except (ValueError, TypeError):
        port = _DEFAULT_PORT

    _ensure_mcp_token(db)

    logger.info("[MCP] Starting MCP server on port %d", port)
    mcp = create_mcp_server(host="0.0.0.0", port=port)
    app = _build_app(mcp)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _ensure_mcp_token(db) -> None:
    """Generate an MCP auth token on first boot if none exists."""
    from services.wrapper_auth_service import WrapperAuthService
    from services.settings_service import SettingsService

    settings = SettingsService(db)
    existing = settings.get("mcp_server_token_wrapper_id")
    if existing:
        auth_svc = WrapperAuthService(db)
        wrapper = auth_svc.get_wrapper(existing)
        if wrapper:
            return

    auth_svc = WrapperAuthService(db)
    try:
        raw_token, wrapper_id = auth_svc.create_token(
            name="MCP Server (External Agents)",
            capabilities={"signals": [], "intents": ["talk_to_chalie"]},
            permissions={"query": ["*"], "update": ["*"], "broadcast": False},
            wrapper_id_override="__mcp_server__",
        )
    except sqlite3.IntegrityError:
        logger.info("[MCP] Token already exists (concurrent boot); skipping")
        return

    settings.set("mcp_server_token_wrapper_id", wrapper_id)

    logger.info(
        "[MCP] Generated MCP auth token (wrapper_id=%s). "
        "Retrieve via: Settings > MCP Server in the brain dashboard.",
        wrapper_id,
    )
    logger.info("[MCP] Token (shown once): %s", raw_token)
