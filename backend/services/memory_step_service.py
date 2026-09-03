# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Memory step service — settle-triggered, ephemeral memory consolidation.

After a turn settles on an in-scope channel, a background MessageProcessor on
the SAME channel reconciles memories using only the four memory tools (Recall,
SaveGraph, SaveMap, DeleteGraph). The step is ephemeral: it runs once per settle
window, single-flight per channel with trailing coalesce.

It runs on :class:`~configs.channels.memory_step.MemoryStepConfig` — one config
for every in-scope channel, carrying its own system prompt rather than the
settling channel's. Only what must follow the channel is carried across:
read_channel, policy_channel and external_turn_id.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from configs.channels.memory_step import MemoryStepConfig
from configs.enums.channels import Channel
from models.transcript import Transcript
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

def _in_scope(channel: str) -> bool:
    """Return True when *channel* is in-scope for memory stepping.

    Covers the three fixed channels (user, schedule, discovery) and the
    dynamic external-agent prefix, whose concrete values are
    ``external-agent:<name>``.
    """
    if channel in (Channel.USER.value, Channel.SCHEDULE.value, Channel.DISCOVERY.value):
        return True
    return channel.startswith(Channel.EXTERNAL_AGENT.value + ":")


def memory_step_config(
    cfg: ProcessorConfig,
    source_transcript_ids: list[int],
) -> MemoryStepConfig:
    """Build the memory step's config for the channel *cfg* settled on.

    The step no longer clones the settling config: it declares its own system
    prompt, tools and history window, so only the fields that must follow the
    channel are read off *cfg* — where rows are written and read
    (``channel``/``read_channel``), which policy gates the tool calls, and who
    owns the turn id. Everything else is
    :class:`~configs.channels.memory_step.MemoryStepConfig`'s declaration.
    """
    return MemoryStepConfig(
        channel=cfg.channel,
        policy_channel=cfg.policy_channel,
        read_channel=cfg.read_channel,
        external_turn_id=cfg.external_turn_id,
        source_transcript_ids=source_transcript_ids,
    )


@dataclass
class _Pending:
    """Accumulated pending memory step for a channel."""

    config: ProcessorConfig
    turn_id: int
    ids: set[int] = field(default_factory=set)


class MemoryStepService:
    """Per-channel single-flight memory step service with trailing coalesce.

    At most one step per channel runs at a time. If a settle arrives while a
    step runs, the pending entry is updated (latest config + turn_id + union
    of transcript ids). When the current run finishes, if a pending entry
    exists, a trailing run is scheduled to absorb all accumulated settles.
    """

    _instance: "MemoryStepService | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._pending: dict[str, _Pending] = {}

    @classmethod
    def instance(cls) -> "MemoryStepService":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def on_settle(self, mp: "MessageProcessor") -> None:
        """Called from the settling MP's drive thread; must be quick and
        never raise.

        Gates (silent return): not in scope, role is already "memory", or
        skip_transcript is True.
        """
        channel = mp.channel
        if not _in_scope(channel):
            return
        if mp.config.role == "memory":
            return
        if mp.config.skip_transcript:
            return

        ids = Transcript.ids_for_turn(channel, mp.turn_id)
        if not ids:
            return

        with self._lock:
            if channel in self._running:
                if channel in self._pending:
                    pending = self._pending[channel]
                    pending.config = mp.config
                    pending.turn_id = mp.turn_id
                    pending.ids.update(ids)
                else:
                    self._pending[channel] = _Pending(
                        config=mp.config,
                        turn_id=mp.turn_id,
                        ids=set(ids),
                    )
            else:
                self._running.add(channel)
                threading.Thread(
                    target=self._run,
                    args=(channel, mp.config, mp.turn_id, ids),
                    daemon=True,
                    name=f"memory-step-{channel}",
                ).start()

    def _run(
        self,
        channel: str,
        config: ProcessorConfig,
        turn_id: int,
        ids: list[int],
    ) -> None:
        """Run one memory step for *channel*, then release and check pending.

        No raw input: the instruction is the config's system prompt and the body
        is the transcript window PromptService reads, so there is nothing for a
        user message to carry. The empty input row still exists — it anchors the
        act trail.
        """
        try:
            step_config = memory_step_config(config, ids)
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415

            mp = MessageProcessor.process(
                step_config,
                raw_input="",
                turn_id=turn_id if step_config.external_turn_id else -1,
            )
            mp.result()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[MemoryStep] step failed on channel=%s: %s",
                channel,
                exc,
                exc_info=True,
            )
        finally:
            with self._lock:
                if channel in self._pending:
                    next_pending = self._pending.pop(channel)
                    # channel stays in _running; start a new watcher.
                    threading.Thread(
                        target=self._run,
                        args=(
                            channel,
                            next_pending.config,
                            next_pending.turn_id,
                            list(next_pending.ids),
                        ),
                        daemon=True,
                        name=f"memory-step-{channel}",
                    ).start()
                else:
                    self._running.discard(channel)
