"""Heartbeat service — cached client-telemetry singleton.

Writer: POST /health (heartbeat.js, every ~5 min).
Readers: WorldState, locale_service, act_dispatcher.

Holds ONLY the process-level cache of the current snapshot; all telemetry-table
SQL (flatten/unflatten + the snapshot swap) lives on the :class:`~models.telemetry.Telemetry`
model, which this service delegates to.
"""

import logging
import time

from models.telemetry import Telemetry

logger = logging.getLogger(__name__)


class HeartbeatService:

    def __init__(self) -> None:
        self._ctx: dict[str, object] | None = None

    def read(self) -> dict[str, object]:
        if self._ctx is None:
            self._ctx = Telemetry.curr()
        return self._ctx

    def write(self, ctx: dict[str, object]) -> None:
        ctx["saved_at"] = time.time()
        Telemetry.replace(ctx)
        self._ctx = ctx


heartbeat_service = HeartbeatService()
