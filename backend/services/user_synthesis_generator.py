"""UserSynthesisGenerator — compaction-fired background distillation of the user.

When chat-history compaction folds the USER channel, a conversation window has
closed and is worth distilling. The compactor hands over its own tool-call-free
``previous_messages()`` render — captured BEFORE ``Compaction.write`` advanced
the watermark, because afterwards the folded rows can no longer be read — and
``on_compaction`` runs one background :class:`UserSynthesisConfig` turn over
it. The model reads the current synthesis, cross-checks with ``recall``, and
persists a new version via ``save_synthesis``.

Single-flight per process, mirroring MemoryStepService: while a run is live,
further folded blocks APPEND to a pending buffer (a dropped window is lost
observation — accumulate, never replace) and one trailing run absorbs the
accumulation.
"""

from __future__ import annotations

import logging
import threading

from configs.channels.user_synthesis import UserSynthesisConfig
from configs.enums.channels import Channel
from models.user_synthesis import UserSynthesisRow

logger = logging.getLogger(__name__)


class UserSynthesisGenerator:
    """Process-wide single-flight runner for user-synthesis turns."""

    _instance: "UserSynthesisGenerator | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._pending: list[str] = []

    @classmethod
    def instance(cls) -> "UserSynthesisGenerator":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def on_compaction(self, channel: str, folded_block: str) -> None:
        """Called from the compacting parent's drive thread; must be quick and
        never raise. Gates (silent return): not the user channel, or an
        empty folded block."""
        if channel != Channel.USER.value or not folded_block.strip():
            return
        with self._lock:
            if self._running:
                self._pending.append(folded_block)
                return
            self._running = True
            threading.Thread(
                target=self._run,
                args=(folded_block,),
                daemon=True,
                name="user-synthesis-generator",
            ).start()

    def _run(self, block: str) -> None:
        """Run one generator turn over *block*, then release and drain pending."""
        try:
            current = UserSynthesisRow.latest_content() or "(none — first conversation)"
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415

            MessageProcessor.process(
                UserSynthesisConfig(),
                raw_input=f"## Current synthesis\n{current}\n\n## Transcript\n{block}",
            ).result()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[UserSynthesis] generator turn failed: %s", exc, exc_info=True
            )
        finally:
            with self._lock:
                if self._pending:
                    next_block = "\n\n".join(self._pending)
                    self._pending.clear()
                    # _running stays True; the trailing run absorbs the buffer.
                    threading.Thread(
                        target=self._run,
                        args=(next_block,),
                        daemon=True,
                        name="user-synthesis-generator",
                    ).start()
                else:
                    self._running = False
