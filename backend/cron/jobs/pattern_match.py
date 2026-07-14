"""Pattern-match cron job — idle-gated single-pass LLM pattern matcher.

Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_pattern_match``
so the step stays self-contained when the source worker is deleted. The
``pattern_match_cursor`` watermark makes it self-healing regardless of ordering;
the idle-window and min-interval gates are inherited from ``IdleGatedJob``.
"""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class PatternMatchJob(IdleGatedJob):
    """Idle-gated single-pass LLM pattern matcher over a transcript-id window.

    Reads/advances the ``pattern_match_cursor`` watermark and fires only when
    the unprocessed delta clears ``_MIN_DELTA``; the base ``IdleGatedJob``
    applies a 30-min idle window and 5-min min-interval.
    """

    name = "pattern_match"

    def _run(self) -> str:
        """Step 4 — single-pass LLM pattern matcher over a transcript-id window."""
        from configs.enums.channels import Channel  # noqa: PLC0415
        from models.machine_state import MachineStateRow  # noqa: PLC0415
        _DG_KEY_CURSOR = "pattern_match_cursor"
        _MIN_DELTA = 50

        # 1. Read cursor — newest active row wins (MachineStateRow.newest_active_by_key
        # owns the deterministic ORDER BY id DESC that defends against historical
        # / concurrent writes leaving more than one active row).
        cursor = 0
        row = MachineStateRow.newest_active_by_key(_DG_KEY_CURSOR)
        if row and row.value:
            try:
                cursor = int(row.value)
            except (TypeError, ValueError):
                cursor = 0

        # 2. Read latest transcript id.
        # The cursor must count only rows the pattern LOAD window (pattern.py)
        # actually reads — user-behaviour channels, no compaction rows. Counting
        # background-loop rows (dmn writes many) would advance the delta past the
        # _MIN_DELTA trigger and fire spurious pattern passes the load discards.
        from models.transcript import Transcript  # noqa: PLC0415
        latest = Transcript.latest_id([Channel.USER.value]) or 0

        delta = latest - cursor
        if delta < _MIN_DELTA:
            logger.info(
                f"{LOG_PREFIX} pattern_match_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # 3. Fire the pattern pass via the canonical entry point. The skill-
        # personalisation sync (PatternSkillSyncHook) runs inside the turn's
        # post_turn_hooks, keyed off the patterns it touched — both that set and
        # the confidence-decay sweep are derived from the turn's durable rows, so
        # nothing needs to be inspected on the processor after it returns.
        from configs.channels import PatternConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        MessageProcessor.process(PatternConfig(cursor, latest))

        # 4. Advance cursor on success.
        # Cursor write race / known risk: a crash here leaves the cursor
        # pinned to its previous value while the processor has already
        # decayed every untouched pattern. The next tick re-fires the same
        # window and re-decays — visible in tests as confidence drifting
        # below the "−0.005 per cycle" expectation. Spec accepts this as a
        # minor double-fire risk; logging at WARNING so an unexpected
        # cursor-stuck pattern is observable in operator logs.
        try:
            MachineStateRow.store(
                key=_DG_KEY_CURSOR,
                value=str(latest),
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} pattern_match cursor write failed "
                f"cursor={cursor}->{latest} — next tick will re-fire same "
                f"window and re-decay untouched patterns: {exc}"
            )
            raise
        return f"fired cursor={cursor}->{latest} delta={delta}"
