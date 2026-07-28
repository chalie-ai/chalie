# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The read-before-modify guard shared by every ability that changes a file.

One rule, one implementation: **the model's most recent ``read`` of the target
must be newer than the most recent successful change to that target on this
turn.** A modification the model cannot see is a blind modification, whether it
is blind because it never read the file or because its own earlier edit made the
read stale.

Turn-scoped alone is not enough, and that gap is what this module exists to
close. ``read -> edit(ok) -> edit(blind)`` passed the old guard: any read of the
path anywhere in the turn satisfied it, even one the first edit had already
invalidated. In practice the second edit anchors on text the file no longer
contains, comes back ``not-found``, and the model retries the same call until the
runaway backstop kills the turn — 14 calls and 13 errors against one file in a
single observed session, ending in a half-edited file.

``read`` rows are matched on the resolved path, so ``~/notes.txt`` and
``/home/u/notes.txt`` are the same file: ``read`` expands ``~`` before reading
but records the raw argument.

Two deliberate bypasses, both returning ``None`` (allow): no live processor, and
a processor whose turn has not been allocated yet. The act-trail cannot be
inspected in either case, and a guard that cannot see the evidence must not
invent a refusal.

``truncated-read`` is opt-in per caller (``refuse_truncated``). A truncated read
is fatal to a WHOLE-FILE overwrite — it would destroy the tail the model never
saw — but harmless to an anchored edit, which only touches text the model quoted
back verbatim. That asymmetry is the reason this is a parameter and not a rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from abilities._result import ToolResult
from models.tool_call import ToolCall

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

# The param keys a file-touching tool records its target under, in precedence
# order — ``read`` accepts all three, the mutators only ``path``. One chain for
# both so there is a single way a row's target is resolved.
_TARGET_KEYS = ("source", "path", "url")


def read_guard(
    mp: "MessageProcessor | None", target: Path, *, refuse_truncated: bool,
) -> ToolResult | None:
    """The refusal to return, or ``None`` when the modification may proceed.

    *target* must already be resolved — the caller owns path resolution, this
    guard only compares. *refuse_truncated* adds the whole-file-overwrite check
    described in the module docstring.
    """
    if mp is None:
        return None
    # ``turn_id`` stays -1 until ``MessageProcessor.begin()`` allocates one
    # (message_processor.py:152, :258). Before that the turn has no act trail to
    # inspect — not an empty one, an unreadable one — and a guard that cannot see
    # the evidence must not invent a refusal. ``channel`` needs no such check: it
    # is assigned unconditionally in __init__ (:164) and typed ``str``.
    if mp.turn_id < 0:
        logger.warning("read guard: processor has no allocated turn — guard bypassed")
        return None

    from abilities.edit_file import EditFileAbility  # noqa: PLC0415
    from abilities.file_write import FileWriteAbility  # noqa: PLC0415
    from abilities.read import ReadAbility  # noqa: PLC0415

    mutators = (FileWriteAbility.NAME, EditFileAbility.NAME)
    last_read: ToolCall | None = None
    stale_by: ToolCall | None = None
    # ``by_turn`` yields the turn's calls in autoincrement-id order, so "newer
    # than the last read" is simply "seen after it": a read clears the staleness,
    # a later change re-arms it. No id arithmetic, no timestamp granularity.
    for row in ToolCall.by_turn(mp.channel, mp.turn_id):
        is_read = row.tool_name == ReadAbility.NAME
        # Only a SUCCEEDED change invalidates a read: a refused or failed edit
        # wrote nothing, so the model's view of the file still holds. The
        # in-flight call is still ``started``, so it can never invalidate itself.
        is_change = row.tool_name in mutators and row.state == ToolCall.DONE
        if (not is_read and not is_change) or not _row_matches(row, target):
            continue
        if is_read:
            last_read, stale_by = row, None
        else:
            stale_by = row

    if last_read is None:
        return ToolResult.err(
            f"You must read {target} before modifying it.",
            code="read-required",
            hint=f"call the '{ReadAbility.NAME}' tool on {target} first, then retry",
        )

    if stale_by is not None:
        return ToolResult.err(
            f"{target} has changed since you last read it — your own "
            f"'{stale_by.tool_name}' call modified it earlier in this turn.",
            code="read-required",
            hint=f"call the '{ReadAbility.NAME}' tool on {target} again to see the "
                 "current content, then retry",
        )

    # Truncation is flagged in the envelope OPEN TAG (its first line) only —
    # matching the whole result would false-positive on file content that
    # happens to contain the marker.
    if refuse_truncated and "truncated=true" in (last_read.result or "").split("\n", 1)[0]:
        return ToolResult.err(
            f"Your last read of {target} was truncated — a full overwrite would "
            "destroy the unread portion.",
            code="truncated-read",
            hint=f"use {EditFileAbility.NAME} for targeted changes, or re-read with a "
                 "higher max_chars to get the full file first.",
        )
    return None


def _row_matches(row: ToolCall, target: Path) -> bool:
    """Whether a recorded tool call acted on *target*.

    ``read`` expands ``~`` before reading but records the RAW argument, so the
    guard must expanduser too or a tilde-path read would never match the
    resolved target it satisfies."""
    try:
        params = json.loads(row.params)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(params, dict):
        return False
    raw = next((params[k] for k in _TARGET_KEYS if params.get(k)), None)
    if not isinstance(raw, str):
        return False
    try:
        return Path(raw).expanduser().resolve() == target
    except (ValueError, OSError):
        # An unresolvable argument (embedded NUL, over-long path) still matches
        # when it is byte-identical to the target — carried over verbatim.
        return raw == str(target)
