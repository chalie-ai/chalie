# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ToolCallService — read/write the ``tool_calls`` trail for one ACT loop and
emit its single live WS frame.

A tool call anchors ONLY to the transcript input row that drove it
(``transcript_id``, resolved off ``self.mp`` — never a parameter); its turn is
derived by joining ``transcript`` on (channel, turn_id) inside
``ToolCall.by_turn``, so ``tool_calls`` carries no turn_id/channel column of its
own. Opening or recording a call un-settles the owning transcript row (§6.9)
unless the tool's ability opts out via ``counts_as_settle=False`` — a settling
tool demotes the turn's settle0 back to in-progress. Every write
emits the row's WS-safe projection (``ToolCall.to_json``, §6.2 — params/result
never cross the wire) gated by ``self.mp.config.BROADCASTS_STATE`` and silenced
outright for the turn-zero memory seed (§6.10).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from abilities._registry import AbilityRegistry
from models.tool_call import ToolCall
from services.time_utils import utc_now

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class ToolCallService:
    """CRUD + emission for ``tool_calls``, keyed off ``self.mp``."""

    def __init__(self, mp: "MessageProcessor") -> None:
        self.mp = mp

    def start(
        self, tool_name: str, params: dict[str, object], summary: "str | None" = None
    ) -> "int | None":
        """Open a row (state=STARTED) the instant a call begins executing,
        un-settle the owning transcript row (unless ``tool_name``'s ability
        opts out via ``counts_as_settle=False``, §6.9), emit the live frame,
        and return the new row's id; :meth:`finish` writes the terminal state
        when the call returns. Skips (returns None) with no anchor row to
        attach to — a delegate with no input row has nothing to record
        against. Both writes run in one atomic block; a failure of either
        logs and yields None, and the turn continues unrecorded."""
        transcript_id = self._transcript_id()
        if transcript_id is None:
            logger.debug("[ToolCallService.start] skipping (no transcript_id): tool=%s", tool_name)
            return None
        try:
            with self.mp.db.transaction():
                call = ToolCall(
                    transcript_id=transcript_id,
                    tool_name=tool_name,
                    params=json.dumps(params),
                    result="",
                    summary=summary or "",
                    created_at=utc_now().isoformat(),
                    ended_at=None,
                    state=ToolCall.STARTED,
                ).save()
                if self._settles(tool_name):
                    self.mp.transcript_service.unsettle()
        except Exception as exc:
            logger.warning(
                "[ToolCallService.start] write failed (non-fatal): tool=%s transcript=%s: %s",
                tool_name, transcript_id, exc,
            )
            return None
        self._emit(call)
        # tool_calls PK is INTEGER autoincrement; the str arm of Model.id
        # exists only for TEXT-UUID models (episodes).
        return cast("int | None", call.id)

    def finish(self, call_id: "int | None", result: str, state: str) -> None:
        """Write the terminal ``result``/``state`` onto a row :meth:`start`
        opened, stamp ``ended_at``, and emit the terminal frame. No-op when
        ``call_id`` is None (the call was never recorded — no anchor). A
        write failure logs and is non-fatal."""
        if call_id is None:
            return
        try:
            call = ToolCall.filter("id", call_id).first()
            if call is None:
                return
            call.result = result
            call.state = state
            call.ended_at = utc_now().isoformat()
            call.save()
        except Exception as exc:
            logger.warning(
                "[ToolCallService.finish] write failed (non-fatal): call=%s: %s", call_id, exc,
            )
            return
        self._emit(call)

    def record(
        self,
        tool_name: str,
        params: dict[str, object],
        result: str,
        state: str = ToolCall.DONE,
        summary: "str | None" = None,
        emit: bool = True,
    ) -> None:
        """One-shot terminal row for an outcome that never enters live
        execution (a denied/pre-validation-failed call, an unknown tool, or
        the turn-zero memory seed): written whole with its terminal ``state``
        and ``ended_at`` set, then un-settles the owning transcript row
        (unless ``tool_name``'s ability opts out via
        ``counts_as_settle=False``, §6.9). ``emit=False`` persists the audit
        row without a WS frame — an unknown tool must still land on the
        trail but fires no live pill. Both writes run in one atomic block; a
        failure of either logs and is non-fatal. Skips silently with no
        anchor row."""
        transcript_id = self._transcript_id()
        if transcript_id is None:
            logger.debug("[ToolCallService.record] skipping (no transcript_id): tool=%s", tool_name)
            return
        try:
            now = utc_now().isoformat()
            with self.mp.db.transaction():
                call = ToolCall(
                    transcript_id=transcript_id,
                    tool_name=tool_name,
                    params=json.dumps(params),
                    result=result,
                    summary=summary or "",
                    created_at=now,
                    ended_at=now,
                    state=state,
                ).save()
                if self._settles(tool_name):
                    self.mp.transcript_service.unsettle()
        except Exception as exc:
            logger.warning(
                "[ToolCallService.record] write failed (non-fatal): tool=%s transcript=%s: %s",
                tool_name, transcript_id, exc,
            )
            return
        if emit:
            self._emit(call)

    def by_turn(self) -> "list[ToolCall]":
        """Every tool call of the current turn, derived by joining
        ``transcript`` on (channel, turn_id) — a turn's calls span an async
        result or delegate re-entry, each its own input row under one
        turn_id."""
        return ToolCall.by_turn(self.mp.channel, self.mp.turn_id)

    def by_exchange(self) -> "list[ToolCall]":
        """The current exchange's tool calls — those of this MP recursion
        instance, floored at its input row (``self.mp.uid``) so a reply or
        async re-entry sharing the turn never re-loads a prior exchange's raw
        trail into context. Falls back to :meth:`by_turn` when this turn has no
        input row (``skip_input_row`` channels): they never fork, so the turn
        and the exchange coincide."""
        if self.mp.uid is None:
            return self.by_turn()
        return ToolCall.by_exchange(self.mp.channel, self.mp.turn_id, self.mp.uid)

    def by_transcript(self) -> "list[ToolCall]":
        """Every tool call anchored to the current input row alone — the
        narrow single-anchor read. Empty with no anchor row."""
        transcript_id = self._transcript_id()
        if transcript_id is None:
            return []
        return ToolCall.by_transcript(transcript_id)

    def _transcript_id(self) -> "int | None":
        """The input row this call anchors to: the row the turn is currently
        writing against, falling back to the turn's own opening row."""
        current = self.mp.current_transcript_id
        return cast("int | None", current) if current is not None else self.mp.uid

    def _settles(self, tool_name: str) -> bool:
        """Whether opening/recording a call for ``tool_name`` demotes the
        owning transcript's settle0 (§6.9) — the registered ability's
        ``counts_as_settle`` flag (default True; ``chat_history_compactor``
        opts out so it never demotes a settle0). An unregistered name (an
        MCP proxy, or a tool the registry has no entry for) defaults True —
        it was never a member of the pre-rewrite exclusion set either."""
        try:
            return AbilityRegistry.get(tool_name).counts_as_settle
        except KeyError:
            return True

    def _emit(self, call: ToolCall) -> None:
        """The sole tool-frame emitter. Silenced outright while the turn-zero
        seed phase runs (§6.10, ``mp.seeding_turn_zero``) — the seed must never
        surface a live pill; the broadcast-state gate itself lives in
        ``mp.push_websocket``. Sets the transient envelope (``type``/``turn_id``)
        on the row before pushing; ``Websocket`` serializes it via
        ``ToolCall.to_json`` (§6.2)."""
        if self.mp.seeding_turn_zero:
            return
        config_type = self.mp.config.type()
        call.type = config_type.value if config_type is not None else ""
        call.turn_id = self.mp.turn_id
        self.mp.push_websocket(call)
