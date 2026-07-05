"""
Chat API — HTTP turn-control and rich-card action endpoints.

Routes:
  POST /api/action             — receive a rich-card control click; dispatches via
                             an inert MessageProcessor's dispatch_service.dispatch()
                             and returns the result synchronously in the response
                             body (no WS frame).

  Async-delegate (subagent) lifecycle lives in its own resource group —
  GET /api/subagents and DELETE /api/subagent/<sub_id> (see api/subagents.py).

Send path:
  User messages are sent via POST /api/thread (new thread) or
  POST /api/thread/<turn_id> (reply) — see api/threads.py. Both return 200 with
  the turn's freshly-opened turn_execution row: the turn_id is allocated
  synchronously (atomic input-row write + turn_executions insert) before the
  response, so the FE holds the handle without waiting on a WS signal. Live
  output still surfaces via WS signals → REST pull.

Design:
  WS is receive-only push (server→client). All client→server requests use HTTP.
  turn_id is the only handle: each send spawns an isolated turn on its own daemon
  thread; the MessageProcessor allocates (new spine turn) or reuses (thread reply)
  a turn_id and carries it end to end — every WS signal stamps it, so the surface
  keys its updates on it and holds no shared "active turn" state. A second message
  to a busy surface is never sent mid-turn: the frontend QUEUES it per turn_id and
  dispatches once that turn's execution reaches a terminal state. Interrupt is the
  lone exception — DELETE /api/thread/<turn_id> (see api/threads.py) requests
  cancellation on a running turn by turn_id, which self-cleans its rows.

  User-channel messages flow through MessageProcessor.process() with the
  UserConfig ProcessorConfig subclass — no MessageProcessor subclass. Live
  output is signal-only, gated by the config's BROADCASTS_STATE: the surface
  holds no event memory and refetches turn blocks over REST. Mid-turn progress
  (the `updated` block-refetch poke) is emitted by MessageProcessor itself
  through its `broadcast` chokepoint; each live tool call is a single
  `tool_name`-bearing frame (state=started/done/error) emitted by ActTrail
  (services/act_trail.py); the turn's lifecycle
  (working/completed/cancelled/crashed) is a separate `turn_execution`
  WS frame emitted by its ExecutionTracker (services/execution_tracker.py) on
  every state flip, including a crash (so the surface always learns a failed
  turn ended, even one that died before producing a reply). The only signal
  originating here is the channel-wide error toast a crashed turn also needs.
"""

import logging
import time
import uuid
from typing import Sequence

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.markup import sanitize
from services.processor_config import ProcessorConfig
from .auth import require_auth
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.chat import ActionRequest, ActionResult

logger = logging.getLogger(__name__)

chat_ns = Namespace("chat", description="Chat turn control and rich-card actions", path="/api")

register_dto(chat_ns, ActionRequest, ActionResult, Error)

_M = chat_ns.models


# ── Background helpers ────────────────────────────────────────────────────────


def deliver_async_result(mp: object, result_text: str) -> None:
    """Appends another assistant turn for a finished async delegate — a plain
    MessageProcessor.process() call, independent of any foreground turn (each runs
    on its own thread, keyed by its own turn_id). The synthesis turn opens its OWN
    turn_executions row (see MessageProcessor.__init__), so the Processes-panel
    stop control cancels it the same way as any other turn — no cancel handle
    needs threading in from the originating delegate.

    Inherits the originating turn's ``turn_id`` so the synthesised reply lands
    in the same thread the delegate was spawned from.
    """
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

    config = getattr(mp, "config", None)
    if config is None:
        logger.warning("[Chat API] async delivery skipped: captured mp has no config")
        return

    synth_config = config.with_hidden_input()
    # Clone the originating metadata but suppress the input row and drop
    # attachments — they were already ingested on the originating turn and must
    # not re-upload on the synthesis turn. Inherit the originating turn id so
    # the synthesised reply lands in the same thread. A delegate produces an
    # assistant row AFTER the turn's settle0, so the synthesis IS a forked reply:
    # it inherits turn_id → the MessageProcessor switches itself to FORK view
    # internally (no external flag).
    metadata = dict(getattr(mp, "metadata", None) or {})
    metadata["hidden_input"] = True
    metadata["attachments"] = []
    turn_id = getattr(mp, "turn_id", None)
    # The synthesis turn is a full UserConfig turn, so its lifecycle signals
    # come from MessageProcessor itself — nothing to emit here. Fire-and-forget:
    # nothing here consumes the final text, so the drive thread is never joined.
    MessageProcessor.process(synth_config, result_text, metadata, turn_id if turn_id is not None else -1)


def _stage_chat_uploads(files: Sequence[object]) -> list[object]:
    """Returns temp paths that _seed_turn_zero feeds to document.upload — which
    ingests by PATH, never bytes, so no file blob ever reaches the act-trail.
    """
    from services.filename_utils import safe_filename  # noqa: PLC0415
    from services.tmp_storage import new_tmp_path  # noqa: PLC0415

    paths: list[object] = []
    for f in files:
        if not f or not getattr(f, 'filename', None):
            continue
        name = safe_filename(getattr(f, 'filename')) or "attachment"
        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{name}")
        getattr(f, 'save')(tmp_path)
        paths.append(tmp_path)
    return paths


# ── Action-button dispatch helpers ───────────────────────────────────────────


class _ActionButtonConfig(ProcessorConfig):
    """Minimal ProcessorConfig for action-button dispatches (no ACT loop runs).

    No ACT loop is executed; prompt_service returns "" for the action_button
    channel, so no prompt builders are needed here.
    """

    def __init__(self) -> None:
        super().__init__(
            channel="action_button",
            role="action_button",
            policy_channel=ProcessorConfig.PolicyChannel.CHAT,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@chat_ns.route("/action")
class ActionResource(Resource):
    @require_auth
    @chat_ns.doc(
        description=(
            "Dispatches a rich-card control click via an inert MessageProcessor's "
            "dispatch_service and returns the result synchronously. A card action is "
            "a silent state mutation, not a conversation turn: it never persists to "
            "the transcript and crosses no WS frame, so the calling card resolves its "
            "optimistic update straight from this HTTP response."
        )
    )
    @chat_ns.expect(_M["ActionRequest"])
    @chat_ns.response(200, "Action result", model=_M["ActionResult"])
    @chat_ns.response(400, "Unknown skill", model=_M["Error"])
    @chat_ns.response(422, "Validation failed", model=_M["Error"])
    @chat_ns.response(500, "Action handler error", model=_M["Error"])
    @responds(ActionResult, code=200)
    @expects(ActionRequest)
    def post(self, dto: ActionRequest) -> ActionResult | ResponseReturnValue:
        """Run the skill and return its result inline — the WS bus stays signal-only."""
        action_start = time.time()
        try:
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415

            # Inert (I2): no process()/begin(), so zero DB/WS side-effects at
            # construction — the config's skip_transcript/skip_input_row/
            # broadcast_to=None flags keep the dispatch itself silent, and with
            # no turn_executions row, should_stop() fails open (always False) —
            # there is no running turn for a stop button to reach.
            mp = MessageProcessor(_ActionButtonConfig())
            result_text = mp.dispatch_service.dispatch(dto.skill, dto.params)
            if result_text.startswith("Unknown tool:"):
                return error(f"Unknown skill: {dto.skill}", 400)

            return ActionResult(
                content=sanitize(result_text or "Done."),
                duration_ms=int((time.time() - action_start) * 1000),
            )
        except Exception as exc:
            logger.exception("[Chat API] Action handler error: %s", exc)
            return error("Action handler error", 500)
