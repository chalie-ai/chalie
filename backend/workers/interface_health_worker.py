"""
Interface Health Monitor — periodic liveness check for paired interfaces.

Runs as a daemon thread. Every 30 seconds, pings all paired interfaces
via their health endpoint. After 3 consecutive failures, marks an interface
as offline (its tools become invisible to the LLM). Auto-recovers when
the interface comes back.
"""

import logging
import time

logger = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 30  # seconds


def interface_health_worker():
    """Daemon thread: periodically health-check all paired interfaces."""
    logger.info("[INTERFACE HEALTH] Worker started (interval: %ds)", HEALTH_CHECK_INTERVAL)

    # Wait for startup to complete before first check
    time.sleep(10)

    while True:
        try:
            from services.interface_registry_service import InterfaceRegistryService
            svc = InterfaceRegistryService()
            interfaces = svc.list_interfaces()

            if interfaces:
                for iface in interfaces:
                    if iface.get("status") != "revoked":
                        svc.health_check(iface["id"])
        except Exception as e:
            logger.warning("[INTERFACE HEALTH] Check cycle failed: %s", e)

        time.sleep(HEALTH_CHECK_INTERVAL)
