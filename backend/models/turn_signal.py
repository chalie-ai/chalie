"""Transient mid-turn WS signal frames the spine emits (§4.1).

Three lean kinds with no persisted counterpart: ``updated`` (TranscriptService
pokes a surface to refetch the turn's block), ``provider_retry``
(ProviderService toasts that an upstream resend is underway — a notice, not a
turn state) and ``context_usage`` (ProviderService reports how full the request
it just sent was against the window — the context meter's live feed). Each is a
fixed wire shape the frontend router (``utils/driftDispatcher.ts::
_dispatchTurnSignal``) discriminates on ``status``: ``updated`` →
``{status, turn_id, type}``; ``provider_retry`` adds ``message`` / ``attempt``
/ ``max_attempts``; ``context_usage`` adds ``tokens_input`` /
``context_window``. Subclasses
:class:`~models.ws_message.WsMessage` — kwargs field storage plus the
all-public-fields ``to_dict`` it inherits, no ``save``, no service reach; the
owning service hands one to ``Websocket.broadcast`` (Rule 9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from models.ws_message import WsMessage

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class TurnSignal(WsMessage):
    """A lean transient turn-progress frame: ``updated``, ``provider_retry`` or
    ``context_usage``."""

    @classmethod
    def updated(cls, mp: "MessageProcessor") -> Self:
        """Poke a surface to refetch the turn's block. Carries no broadcast gate
        of its own — callers emit through ``mp.push_websocket``, the one pre-flight
        that drops the frame for a non-broadcasting or internal-channel turn:
        ``mp.push_websocket(TurnSignal.updated(mp))``."""
        return cls(status="updated", turn_id=mp.turn_id, type=mp.config.type_value())

    @classmethod
    def provider_retry(
        cls,
        turn_id: int | None,
        type_: str | None,
        attempt: int,
        max_attempts: int,
        message: str,
    ) -> Self:
        """Toast that an upstream resend is underway; the turn stays in flight."""
        return cls(
            status="provider_retry",
            turn_id=turn_id,
            type=type_,
            attempt=attempt,
            max_attempts=max_attempts,
            message=message,
        )

    @classmethod
    def context_usage(
        cls,
        mp: "MessageProcessor",
        tokens_input: int,
        context_window: int,
    ) -> Self:
        """Report how full the request just sent was against its window — the
        context meter's live feed, emitted once per CHAT provider call because
        that is exactly when the fraction moves. Carries the measured count
        rather than a ratio: the surface owns the rendering. Like ``updated``,
        it holds no broadcast gate of its own — callers emit through
        ``mp.push_websocket``."""
        return cls(
            status="context_usage",
            turn_id=mp.turn_id,
            type=mp.config.type_value(),
            tokens_input=tokens_input,
            context_window=context_window,
        )
