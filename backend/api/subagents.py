"""Subagents namespace — async-delegate (subagent) lifecycle for the Processes panel.

Routes (mounted under ``/api``):
  GET    /api/subagents         — active async delegates (hydrates the panel on
                                  load/reconnect, since WS push is missed while away).
  DELETE /api/subagent/<sub_id> — cancel a running delegate by its sub_id (the
                                  panel's per-row stop control). Returns 200 always.

Collection is plural (``/subagents``), member is singular (``/subagent/<sub_id>``),
cancel is DELETE — the same taxonomy the thread resource uses (api/threads.py). A
delegate is keyed by its own ``sub_id``, NOT a turn_id: backgrounded calls are
fire-and-notify and outlive the ACT iteration that spawned them, so stopping one is
independent of interrupting a turn (``DELETE /api/thread/<turn_id>``). The delegate's
own cancel_event is what gets set — see services/async_delegate_runner.py.
"""

import logging

from flask_restx import Namespace, Resource

from .auth import require_auth
from .dto import Error, register_dto, responds
from .dto.subagent import ActiveSubagents, SubagentStopResult
from services.log_utils import safe

logger = logging.getLogger(__name__)

subagents_ns = Namespace("subagents", description="Async delegate lifecycle for the Processes panel", path="/api")

register_dto(subagents_ns, ActiveSubagents, SubagentStopResult, Error)
_M = subagents_ns.models


@subagents_ns.route("/subagents")
class SubagentsResource(Resource):
    @require_auth
    @subagents_ns.response(200, "OK", model=_M["ActiveSubagents"])
    @responds(ActiveSubagents)
    def get(self) -> ActiveSubagents:
        """Active async delegates — hydrates the Processes panel on load/reconnect,
        since WS push events are missed while the client is disconnected. Each row
        carries the tool name, the model's summary of what it's doing, and when it
        started."""
        from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415

        return ActiveSubagents(subagents=async_delegate_runner.active())


@subagents_ns.route("/subagent/<sub_id>")
class SubagentResource(Resource):
    @require_auth
    @subagents_ns.param("sub_id", "Async delegate id")
    @subagents_ns.response(200, "Cancel ack", model=_M["SubagentStopResult"])
    @responds(SubagentStopResult)
    def delete(self, sub_id: str) -> SubagentStopResult:
        """Cancel a running async delegate — the Processes panel's per-row stop.
        Sets the delegate's cancel_event (it exits at the next ACT boundary); an
        unknown/finished sub_id is a harmless ``not_found`` ack. Addressed by sub_id,
        not turn_id — a backgrounded delegate is independent of its spawning turn."""
        from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415

        if async_delegate_runner.cancel(sub_id):
            logger.info("[Subagents API] cancel delivered to delegate %s", safe(sub_id[:8]))
            return SubagentStopResult(cancelled=True)
        return SubagentStopResult(reason="not_found")
