"""Pattern-match cron job — idle-gated single-pass LLM pattern matcher.

Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_pattern_match``
so the step stays self-contained when the source worker is deleted. The
``pattern_match_cursor`` watermark makes it self-healing regardless of ordering;
the idle-window and min-interval gates are inherited from ``IdleGatedJob``.
"""

from __future__ import annotations

from cron.jobs._cursor_gated import CursorGatedJob


class PatternMatchJob(CursorGatedJob):
    """Idle-gated single-pass LLM pattern matcher over a transcript-id window.

    Reads/advances the ``pattern_match_cursor`` watermark and fires only when
    the unprocessed delta clears ``min_delta``; the base ``CursorGatedJob``
    inherits idle-window (30 min) and min-interval (5 min) gates from
    ``IdleGatedJob``.

    Cursor write race / known risk: a crash during cursor advance leaves the
    cursor pinned to its previous value while the processor has already
    decayed every untouched pattern. The next tick re-fires the same window
    and re-decays — visible in tests as confidence drifting below the
    "−0.005 per cycle" expectation. Spec accepts this as a minor double-fire
    risk; logging at WARNING so an unexpected cursor-stuck pattern is
    observable in operator logs.
    """

    name = "pattern_match"
    cursor_key = "pattern_match_cursor"
    min_delta = 50

    def _latest_id(self) -> int:
        from models.transcript import Transcript  # noqa: PLC0415
        from configs.enums.channels import Channel  # noqa: PLC0415

        # The cursor must count only rows the pattern LOAD window
        # (pattern.py) actually reads — user-behaviour channels, no compaction
        # rows. Counting background-loop rows (dmn writes many) would advance
        # the delta past the min-delta trigger and fire spurious pattern passes
        # the load discards.
        return Transcript.latest_id([Channel.USER.value]) or 0

    def _make_config(self, window_start: int, window_end: int):
        from configs.channels import PatternConfig  # noqa: PLC0415

        return PatternConfig(window_start, window_end)
