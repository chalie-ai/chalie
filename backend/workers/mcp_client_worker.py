"""
MCP Client Heartbeat Worker — pings configured remote MCP servers every 15 minutes.

Registered in run.py via:
    _try_register(manager, "mcp-client-heartbeat",
                  "workers.mcp_client_worker", "mcp_client_worker")

On each cycle, McpClientService.run_heartbeat() iterates every enabled server,
connects via the MCP streamable-HTTP transport, syncs the tool list into
data/mcp_tools.sqlite, and flips the server's status to online/offline.

Errors are caught per-server inside run_heartbeat() — one bad server never
blocks the others.
"""

import logging
import time

logger = logging.getLogger(__name__)

# How long to wait before the first heartbeat so the rest of Chalie (DB, API)
# has time to initialize.
_INITIAL_DELAY_SECS = 30

# How long between heartbeat cycles.  15 minutes matches the design contract.
_HEARTBEAT_INTERVAL_SECS = 900


class McpClientWorker:
    """Daemon thread entry point for the MCP client heartbeat service."""

    @classmethod
    def run(cls) -> None:
        logger.info("[MCP CLIENT WORKER] Starting; initial delay %ds", _INITIAL_DELAY_SECS)
        time.sleep(_INITIAL_DELAY_SECS)

        from services.mcp_client_service import McpClientService
        svc = McpClientService()

        while True:
            try:
                svc.run_heartbeat()
            except Exception as exc:
                logger.exception("[MCP CLIENT WORKER] Heartbeat cycle error: %s", exc)
            time.sleep(_HEARTBEAT_INTERVAL_SECS)
