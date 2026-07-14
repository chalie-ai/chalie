# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ToolCallGcService — the off-turn Decayable that garbage-collects orphaned
``tool_calls`` rows.

Retention purges ``transcript`` rows out from under the tool calls anchored to
them, leaving dangling ``transcript_id``s no live read can ever reach (every
tool-call read joins ``transcript``). This is the tool-call *retention sweep*:
it registers on the :class:`~orchestrators.decay_engine.DecayEngine` and, once
those orphans age past the model's window, deletes them — the one GC that keeps
``tool_calls`` from growing without bound.

A thin facade, mirroring the memory-lane decayables (``MiscService`` et al.):
mp-less (T2 — decay is off-turn), it owns no SQL and delegates the whole sweep
to :meth:`~models.tool_call.ToolCall.decay`. It is a separate class from the
turn-scoped ``ToolCallService`` on purpose: that service is intrinsically
``mp``-bound (every method reads ``self.mp``), whereas this maintenance runs
with no turn at all.
"""

from __future__ import annotations

from models.tool_call import ToolCall


class ToolCallGcService:
    """Off-turn GC of orphaned ``tool_calls`` rows — the tool-call retention sweep."""

    def decay(self) -> int:
        """Off-turn Decayable entry point: hard-delete tool_calls whose anchoring
        transcript row is gone once they age past the model's orphan-GC window,
        returning the number of rows deleted (``0`` = ran, nothing to sweep)."""
        return ToolCall.decay()
