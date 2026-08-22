# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MCP server built on the SDK's ``MCPServer`` — single ``talk_to_chalie`` tool for external agents.

Streamable HTTP on a dedicated port (default 8462). The transport carries no
authentication of its own: what an external agent may do is decided per tool
by the external-agent policy channel. :class:`McpListener` owns the uvicorn
lifecycle, so ``mcp_server_enabled`` and ``mcp_server_port`` take effect the
moment they change — no backend restart.
"""

import asyncio
import logging
import re
import socket
import threading
import time

import uvicorn
from mcp.server import MCPServer
from starlette.applications import Starlette

from configs.channels import EAMPConfig
from exceptions import ProviderRetriesExhaustedError
from models.setting import Setting

logger = logging.getLogger(__name__)

_MCP_SERVER_NAME = "chalie"
_DEFAULT_PORT = 8462
_BIND_HOST = "0.0.0.0"
_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9 _\-\.]{1,100}$')
# Every wait is bounded so a wedged server can park neither the settings
# endpoint nor the reconcile loop.
_START_TIMEOUT_S = 10.0
_STOP_TIMEOUT_S = 10.0
_RECONCILE_INTERVAL_S = 5.0


def create_mcp_server() -> MCPServer:
    """Create and configure the MCP server with the talk_to_chalie tool.

    Bind host and port are not set here — the constructor does not take them
    (they are applied where they matter: ``host`` in ``_build_app`` and ``port``
    by :class:`McpListener`).
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

        logger.info(
            "[MCP] talk_to_chalie: agent=%s project=%s loop_in_human=%s",
            agent_name, project_or_task_name, loop_in_human,
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
    """Build the streamable-HTTP ASGI app for one listener start.

    The SDK's session manager runs once per instance, so every start builds a
    fresh server and app instead of reusing the previous one.
    """
    # host="0.0.0.0" is load-bearing, not cosmetic: the SDK auto-enables
    # DNS-rebinding protection (allowed_hosts locked to 127.0.0.1/localhost/[::1])
    # ONLY when this host is a loopback address. Passing the real bind host keeps
    # that protection OFF, so networked external agents can connect — the
    # permissive transport 1.x always ran. Do not "simplify" this back to the
    # default 127.0.0.1.
    app: Starlette = mcp.streamable_http_app(host=_BIND_HOST)
    return app


class McpListener:
    """Owns the inbound MCP server's lifecycle.

    Desired state lives in the ``settings`` table (``mcp_server_enabled``,
    ``mcp_server_port``); :meth:`reconcile` reads it and starts, stops, or
    moves the uvicorn server until the live listener matches. Every caller —
    the settings endpoint right after a write, the worker loop on its tick —
    goes through one lock, so two reconciles never race each other.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None
        self.error: str | None = None

    @property
    def listening(self) -> bool:
        """True while the uvicorn server has started and its thread is alive."""
        return (
            self._server is not None
            and self._server.started
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def listening_port(self) -> int | None:
        """The port actually being served, or None when nothing is listening."""
        return self._port if self.listening else None

    @staticmethod
    def desired_state() -> tuple[bool, int]:
        """The owner's intent from the settings table: ``(enabled, port)``.

        Absent rows mean the defaults — enabled, on 8462 — so a fresh install
        serves without anyone having visited the settings page.
        """
        enabled = Setting.get_value("mcp_server_enabled")
        port_setting = Setting.get_value("mcp_server_port")
        try:
            port = int(port_setting) if port_setting else _DEFAULT_PORT
        except (ValueError, TypeError):
            port = _DEFAULT_PORT
        return enabled is None or str(enabled).lower() not in ("false", "0", "no"), port

    def reconcile(self) -> None:
        """Make the live listener match the settings table. Safe from any thread."""
        with self._lock:
            enabled, port = self.desired_state()
            if self._thread is not None and not self._thread.is_alive():
                # The serving thread ended on its own (crash, startup failure):
                # it already logged why, so just forget it and start clean below.
                self._server = self._thread = self._port = None
            if self.listening and enabled and self._port == port:
                return
            if self._server is not None:
                self._stop()
            if enabled:
                self._start(port)
            else:
                self.error = None

    def _start(self, port: int) -> None:
        # Bind here rather than letting uvicorn do it: uvicorn answers a taken
        # port with sys.exit inside its own thread, which would surface only as
        # a dead thread. Binding synchronously turns it into a recorded,
        # retryable error the settings page can show.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((_BIND_HOST, port))
        except OSError as exc:
            sock.close()
            self.error = f"cannot bind port {port}: {exc}"
            logger.warning("[MCP] %s — retrying every %gs", self.error, _RECONCILE_INTERVAL_S)
            return

        server = uvicorn.Server(
            uvicorn.Config(
                _build_app(create_mcp_server()),
                host=_BIND_HOST,
                port=port,
                log_level="info",
                timeout_graceful_shutdown=5,
            )
        )
        thread = threading.Thread(
            target=self._serve,
            args=(server, sock, port),
            name=f"mcp-listener-{port}",
            daemon=True,
        )
        self._server, self._thread, self._port = server, thread, port
        thread.start()

        deadline = time.monotonic() + _START_TIMEOUT_S
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if server.started:
            self.error = None
            logger.info("[MCP] Listening on %s:%d", _BIND_HOST, port)
        elif thread.is_alive():
            self.error = f"port {port}: server did not start within {_START_TIMEOUT_S:g}s"
            logger.error("[MCP] %s", self.error)

    def _serve(self, server: uvicorn.Server, sock: socket.socket, port: int) -> None:
        try:
            server.run(sockets=[sock])
        except (Exception, SystemExit) as exc:
            self.error = f"port {port}: listener crashed ({type(exc).__name__}: {exc})"
            logger.exception("[MCP] Listener on port %d crashed", port)
        finally:
            sock.close()

    def _stop(self) -> None:
        server, thread, port = self._server, self._thread, self._port
        self._server = self._thread = self._port = None
        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(_STOP_TIMEOUT_S)
        if thread.is_alive():
            server.force_exit = True
            thread.join(_STOP_TIMEOUT_S)
        if thread.is_alive():
            logger.error("[MCP] Listener on port %s did not stop within %gs", port, 2 * _STOP_TIMEOUT_S)
            return
        logger.info("[MCP] Listener on port %s stopped", port)


listener = McpListener()


def run_mcp_server() -> None:
    """WorkerManager service: keep the inbound listener matched to its settings.

    Applies the settings at boot, then re-checks on a fixed tick so a crashed
    listener is restarted, a failed bind is retried, and a settings row edited
    outside the API still takes effect. Never returns — the manager flags a
    returning service as dead and respawns it every five seconds.
    """
    while True:
        listener.reconcile()
        time.sleep(_RECONCILE_INTERVAL_S)
