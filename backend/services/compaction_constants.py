"""
Compaction tuning constants.

All constants are module-level literals so the meta-harness can diff-patch
them mechanically. Do NOT read these from config/env at import time.

Single source of truth — every consumer (the proactive compaction trigger in
MessageProcessor._loop, the render/compaction-input bound in
MessageProcessor.get_previous_messages, tests, docs) imports from here. Never
hardcode the numeric value at a call site; that reintroduces drift.
"""

# Compaction window, measured in transcript rows since the watermark.
#
# Two roles, one value:
#   1. Proactive trigger — _loop compacts when the UNCAPPED backlog
#      (len(_previous_rows())) exceeds this many rows.
#   2. Render / compaction-input bound — get_previous_messages() renders at most
#      this many most-recent rows, so both the parent prompt and the compaction
#      input stay bounded even when the watermark is 0 (e.g. the post-migration
#      cut-over). See docs/13-MESSAGE-FLOW.md "Mid-ACT Compaction".
COMPACTION_ROW_WINDOW = 50
