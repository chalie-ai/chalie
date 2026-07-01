"""
Chat API — HTTP turn-control and rich-card action endpoints.

Routes:
  POST /api/action             — receive a rich-card control click; dispatches via
                             ToolDispatcher.dispatch() and returns the result
                             synchronously in the response body (no WS frame).

  Async-delegate (subagent) lifecycle lives in its own resource group —
  GET /api/subagents and DELETE /api/subagent/<sub_id> (see api/subagents.py).

Send path:
  User messages are sent via POST /api/thread (new thread) or
  POST /api/thread/<turn_id> (reply) — see api/threads.py. Both return 200 with
  {turn_id, channel}: the turn_id is allocated synchronously (atomic input-row
  write) before the response, so the FE holds the handle without waiting on a WS
  signal. Live output still surfaces via WS signals → REST pull.

Design:
  WS is receive-only push (server→client). All client→server requests use HTTP.
  turn_id is the only handle: each send spawns an isolated turn on its own daemon
  thread; the MessageProcessor allocates (new spine turn) or reuses (thread reply)
  a turn_id and carries it end to end — every WS signal stamps it, so the surface
  keys its updates on it and holds no shared "active turn" state. A second message
  to a busy surface is never sent mid-turn: the frontend QUEUES it per turn_id and
  dispatches once that turn's ``done`` arrives. Interrupt is the lone exception —
  DELETE /api/thread/<turn_id> (see api/threads.py) cancels a running turn by
  turn_id, which self-cleans its rows.

  User-channel messages flow through MessageProcessor.process() with the
  UserConfig ProcessorConfig subclass — no MessageProcessor subclass. Live
  output is signal-only, gated by the config's BROADCASTS_STATE: the surface
  holds no event memory and refetches turn blocks over REST. Every lifecycle
  signal (working/updated/done + the per-tool tool_called/tool_done timers) is
  emitted by MessageProcessor itself through its `broadcast` chokepoint,
  including the terminal ``done`` on a FAILED turn (fired from the
  live instance — it owns turn_id). The only signal originating here is the
  channel-wide error toast that failed turn also needs.
"""

import logging
import threading
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


def deliver_async_result(mp: object, result_text: str, cancel_event: threading.Event) -> None:
    """Appends another assistant turn for a finished async delegate — a plain
    MessageProcessor.process() call, independent of any foreground turn (each runs
    on its own thread, keyed by its own turn_id).

    The delegate's ``cancel_event`` is threaded into the synthesis turn so the
    Processes-panel stop control aborts a spiralling delegate at the next chain
    boundary (the processor returns "" and self-cleans when it is set).

    Inherits the originating turn's ``turn_id`` so the synthesised reply lands
    in the same thread the delegate was spawned from.
    """
    from services.message_processor import MessageProcessor  # noqa: PLC0415

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
    metadata = dict(getattr(mp, "_metadata", None) or {})
    metadata["hidden_input"] = True
    metadata["attachments"] = []
    turn_id = getattr(mp, "turn_id", None)
    if turn_id is not None:
        metadata["turn_id"] = turn_id
    # The synthesis turn is a full UserConfig turn, so its working → … → done
    # lifecycle signals come from MessageProcessor itself — nothing to emit here.
    MessageProcessor.process(result_text, synth_config, metadata, cancel_event=cancel_event)


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

    No ACT loop is executed; the three prompt builders are never invoked —
    they return "" to satisfy the abstract base.
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

    def get_user_definition(self, mp: object) -> str:
        return ""

    def get_user_prompt(self, mp: object) -> str:
        return ""

    def get_system_prompt(self, mp: object) -> str:
        return ""


class _ActionCtx:
    """Minimal mp-like context for ToolDispatcher in action-button dispatches.

    ``broadcast_to=None`` on the config keeps dispatches silent. ``uid=None``
    signals a non-user channel dispatch. A fresh ``cancel_event`` per instance
    means each action-button dispatch can be independently signalled.
    """

    config = _ActionButtonConfig()
    uid = None

    def __init__(self) -> None:
        self.cancel_event = threading.Event()


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@chat_ns.route("/action")
class ActionResource(Resource):
    @require_auth
    @chat_ns.doc(
        description=(
            "Dispatches a rich-card control click via ToolDispatcher and returns the "
            "result synchronously. A card action is a silent state mutation, not a "
            "conversation turn: it never persists to the transcript and crosses no WS "
            "frame, so the calling card resolves its optimistic update straight from "
            "this HTTP response."
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
            from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

            result_text = ToolDispatcher(_ActionCtx()).dispatch(dto.skill, dto.params)
            if result_text.startswith("Unknown tool:"):
                return error(f"Unknown skill: {dto.skill}", 400)

            return ActionResult(
                content=sanitize(result_text or "Done."),
                duration_ms=int((time.time() - action_start) * 1000),
            )
        except Exception as exc:
            logger.exception("[Chat API] Action handler error: %s", exc)
            return error("Action handler error", 500)
