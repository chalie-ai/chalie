"""TurnHandoverService — the turn's own hand-over, captured before compaction.

Compaction erases the in-flight turn's act trail: the marker ``tool_calls`` row
records unconditionally and the trail render drops every call at or below it, so
the tool work, thinking and interim prose of the turn that triggered compaction
vanish with nothing left in their place. ``chat_history_compactor`` cannot fill
that gap — it reads ``transcript_service.read()``, which floors every turn at its
``settle0`` and therefore skips the unsettled turn entirely.

This pass owns exactly that gap. It reads the turn currently in flight — its
transcript rows plus the RAW tool-call results, deliberately unfiltered by the
compaction watermark, because the watermark marks precisely what is about to be
erased — and asks the model to write itself a first-person hand-over.

Never raises and never returns empty: any failure, and any empty model output,
falls back to a pointer at the last input row so the erasure is never left
standing alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class TurnHandoverService:
    """Captures the in-flight turn's hand-over immediately before compaction."""

    @staticmethod
    def capture(mp: "MessageProcessor") -> str:
        """The turn's hand-over, or the fallback pointer. Never raises, never empty.

        The caller writes the returned text before the compaction marker lands —
        the erasure must never happen with nothing written in its place.
        """
        from abilities._turn_handover_config import TurnHandoverConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        try:
            body = TurnHandoverService._fit(mp)
            if body is None:
                logger.warning(
                    "[turn_handover] nothing to hand over on channel=%s turn=%s",
                    mp.channel, mp.turn_id,
                )
            else:
                handover = (MessageProcessor.process(
                    TurnHandoverConfig(), raw_input=body,
                ).result() or "").strip()
                if handover:
                    return handover
                logger.warning(
                    "[turn_handover] empty hand-over on channel=%s turn=%s",
                    mp.channel, mp.turn_id,
                )
        except Exception:
            logger.exception(
                "[turn_handover] capture failed on channel=%s turn=%s",
                mp.channel, mp.turn_id,
            )
        return TurnHandoverService._fallback(mp)

    @staticmethod
    def _fit(mp: "MessageProcessor") -> str | None:
        """Build the hand-over request body and shrink it to fit the cap.

        The body is the in-flight turn's transcript rows followed by its raw
        tool-call results. Sizing mirrors
        ``ChatHistoryCompactor._fit_compaction_input``: the bare request almost
        always fits, so the drop loop is the EXTREMELY RARE fallback — while the
        {system + body} pair exceeds the cap, drop the OLDEST tool result one at
        a time, keeping the newest, with a floor of one surviving result. Only
        tool results are droppable; the turn's prose is what the hand-over is
        for. Returns None when the turn has nothing to hand over.

        The compaction marker itself is skipped: a previous compaction in this
        same turn leaves a ``chat_history_compactor`` row whose result is
        bookkeeping, not turn work.
        """
        from abilities._turn_handover_config import TurnHandoverConfig  # noqa: PLC0415
        from abilities.chat_history_compactor import ChatHistoryCompactor  # noqa: PLC0415
        from configs.enums.thinking_level import ThinkingLevel  # noqa: PLC0415
        from models.provider_request import ProviderRequest  # noqa: PLC0415

        lines: list[str] = []
        for row in mp.transcript_service.turn_rows():
            fields = row.to_dict()
            content = cast("str", fields.get("content") or "").replace("\n", " ").strip()
            if content:
                lines.append(f"{cast('str', fields.get('role') or 'unknown')}: {content}")

        tools = [
            f"[{call.tool_name}] {call.params} → {call.result or ''}"
            for call in mp.tool_call_service.by_turn()
            if call.tool_name != ChatHistoryCompactor.NAME
        ]
        if not lines and not tools:
            return None

        system = TurnHandoverConfig().system_prompt
        window = mp.provider_service.context_limit()
        cap = window - max(int(0.10 * window), 8000) if window else 0

        drop = 0
        while True:
            body = "\n".join(lines + tools[drop:])
            if cap <= 0 or drop >= len(tools) - 1:
                return body   # cannot shrink further (no window, or one result left)
            candidate_dto = ProviderRequest(
                system=system,
                messages=[{"role": "user", "content": body}],
                tools=None,
                thinking_mode=ThinkingLevel.LOW,
                cache_prefix=False,
            )
            if mp.provider_service.measure(candidate_dto) <= cap:
                return body
            drop += 1

    @staticmethod
    def _fallback(mp: "MessageProcessor") -> str:
        """A pointer at the last input row, for when the hand-over pass yields nothing.

        Anchored on the newest row in the channel that is neither ``assistant``
        nor ``compaction`` — never ``role='user'``, which is only one of the many
        input roles a channel can carry (a scheduled turn's input row is one).
        Defensive throughout: this is the path taken when something already
        failed, so it must not fail in turn.
        """
        from models.transcript import Transcript  # noqa: PLC0415

        last = ""
        try:
            row = (
                Transcript.filter("channel", mp.channel)
                .filter("role", "assistant", "!=")
                .filter("role", "compaction", "!=")
                .order_by("id DESC")
                .first()
            )
            if row is not None:
                last = cast("str", row.to_dict().get("content") or "").strip()
        except Exception:
            logger.exception("[turn_handover] fallback anchor lookup failed")
        return (
            f'You are continuing a session where the last user message was "{last}". '
            "Your session got cut-off. Use memory and other tools to pick up where you left off."
        )
