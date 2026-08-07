"""Idle-gated cron job: Memory v3 consolidation (background distillation).

Fires :class:`services.memory_consolidator_service.MemoryConsolidatorService`
when the user is idle. The 10-minute idle window matches the Memory v3 design
(inactivity trigger); ``min_interval`` keeps it from re-firing every minute once
idle. Per-turn idempotence (a consolidated turn is not re-consolidated) lives in
the service, not the schedule.
"""

from __future__ import annotations

from datetime import timedelta

from cron.base import IdleGatedJob


class MemoryConsolidatorJob(IdleGatedJob):
    name = "memory_consolidator"
    idle_window = timedelta(minutes=10)
    min_interval = timedelta(minutes=5)

    def _run(self) -> str:
        from services.memory_consolidator_service import (  # noqa: PLC0415
            MemoryConsolidatorService,
        )

        return MemoryConsolidatorService().tick()
