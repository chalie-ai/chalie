"""DTOs for turn-control endpoints.

The subagent-cancel DTOs formerly here moved to ``api/response/subagent.py``
with the subagents namespace's migration onto the Endpoint base contract
(``api/endpoints/subagents.py``); this module keeps only :class:`Interrupted`,
still consumed by the legacy ``api/threads.py`` turn-interrupt route.
"""

from __future__ import annotations

from .base import DTO


class Interrupted(DTO):
    """200 ack for DELETE /api/thread/<turn_id> (turn interrupt). ``interrupted``
    is True when a live turn was cancelled, else ``reason`` is ``no_active_turn``."""

    ok: bool = True
    interrupted: bool | None = None
    reason: str | None = None
