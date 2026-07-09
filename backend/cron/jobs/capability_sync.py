"""Idle-gated cron job — capability server sync.

Ported verbatim from ``SubconsciousWorker._step_capability_sync``.

Implements Step 7 (IMAP / CalDAV / CardDAV server sync) as a self-contained
idle-gated cron job. Iterates every loaded :class:`AbstractCapability` and
calls ``monitor()`` on each connected instance.
"""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob

logger = logging.getLogger(__name__)


class CapabilitySyncJob(IdleGatedJob):
    """Idle-gated cron job that syncs every connected capability's server.

    Mirrors the original ``SubconsciousWorker._step_capability_sync`` step.
    The base class provides the idle-window gate, min-interval gate, and the
    ``execute()`` wrapper that logs the returned detail string.
    """

    name = "capability_sync"

    def _run(self) -> str:
        """Step 7 — IMAP / CalDAV / CardDAV server sync.

        Ported verbatim from ``SubconsciousWorker._step_capability_sync``.
        """
        from capabilities import load_capabilities  # noqa: PLC0415

        synced = []
        for cap in load_capabilities().values():
            if cap.is_connected():
                cap.monitor()
                synced.append(cap.get_id())
        return (
            f"synced: {', '.join(synced)}" if synced else "no connected capabilities"
        )
