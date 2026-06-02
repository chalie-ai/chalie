# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from services.processor_config import ProcessorConfig

def _compaction_system_prompt(_mp: object) -> str:
    """System prompt for continuity (history) compaction.  §3a / §4a."""
    from services.system_message_prompt import ContinuityCompactionSystemPrompt
    return ContinuityCompactionSystemPrompt().get_prompt()


COMPACTION_CONFIG = ProcessorConfig(
    channel="compaction",
    role="compaction",
    usage_class="subconscious",
    build_user_prompt=lambda mp: mp._raw_input,
    build_user_definition=lambda _mp: "",
    build_system_prompt=_compaction_system_prompt,
    always_available=[],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=30,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Continuity compaction — bounded loop, no tools, no transcript writes.  §3a."""
