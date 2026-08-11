"""Fixed 30-minute cron job: Memory v3 consolidation (background distillation).

Fires :class:`services.memory_consolidator_service.MemoryConsolidatorService`
every 30 minutes. No idle gate — the consolidator runs on a fixed schedule
regardless of user activity, since the window is defined by the
``transcript.consolidated`` flag, not by turn boundaries.
"""

from __future__ import annotations

from cron.base import ScheduledJob


class MemoryConsolidatorJob(ScheduledJob):
    name = "memory_consolidator"
    minute = "*/30"

    def _run(self) -> str:
        from services.memory_consolidator_service import (  # noqa: PLC0415
            MemoryConsolidatorService,
        )

        return MemoryConsolidatorService().tick()
