# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Memory step service — settle-triggered, ephemeral memory consolidation.

After a turn settles on an in-scope channel, a background MessageProcessor on
the SAME channel with the SAME config class reconciles memories using only the
four memory tools (Recall, SaveGraph, SaveMap, DeleteGraph). The step is
ephemeral: it runs once per settle window, single-flight per channel with
trailing coalesce, and inherits the parent turn's config class so system
prompt, persona, read_channel, and external_turn_id are preserved.
"""
from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.enums.channels import Channel
from models.transcript import Transcript
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

MEMORY_STEP_PROMPT = (
    "Record what has transpired in this thread since the last memory update.\n"
    "Start with `recall` to see what is already stored. Then create, update, or delete memories so they match the current state of things — reuse the exact subject recall returned instead of minting a new one, and do not re-store a fact that is already recorded unchanged.\n"
    "Use `save_graph` for concrete facts: names, dates, locations, schedules, decisions, preferences.\n"
    "Use `save_map` to describe what happened — the narrative of this thread.\n"
    "Use `delete_graph` to remove facts that no longer hold."
)


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
) -> ProcessorConfig:
    """Clone *cfg* and override only the fields required for a memory step.

    Mirrors the copy.copy + object.__setattr__ pattern from
    :meth:`~services.processor_config.ProcessorConfig.with_hidden_input`.

    ``skip_input_row`` is deliberately False: the input row assigns the uid
    the act trail anchors on, and we must not skip it. ``recall_k`` is raised
    to 10 so the recall-first pass surfaces enough existing memory for
    distillation. ``BROADCASTS_STATE`` is shadowed to False on the
    instance so the background step stays off the client's WS frames (the
    gate reads the instance attribute, not the ClassVar). ``USAGE_TYPE`` is
    shadowed to "background" for the same reason DiscoveryConfig declares it:
    the user never asked for this pass and never sees it, so its spend must
    not bill as foreground (the usage log reads the instance attribute too).
    """
    clone = copy.copy(cfg)
    object.__setattr__(
        clone,
        "always_available",
        [Recall.NAME, SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME],
    )
    object.__setattr__(clone, "role", "memory")
    object.__setattr__(clone, "skip_transcript", True)
    object.__setattr__(clone, "skip_input_row", False)
    object.__setattr__(clone, "memory_seed", False)
    object.__setattr__(clone, "recall_k", 10)
    object.__setattr__(clone, "_source_transcript_ids", list(source_transcript_ids))
    object.__setattr__(clone, "BROADCASTS_STATE", False)
    object.__setattr__(clone, "USAGE_TYPE", "background")
    return clone


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
        """Run one memory step for *channel*, then release and check pending."""
        try:
            step_config = memory_step_config(config, ids)
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415

            mp = MessageProcessor.process(
                step_config,
                raw_input=MEMORY_STEP_PROMPT,
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
