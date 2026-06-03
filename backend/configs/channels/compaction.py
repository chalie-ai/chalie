from __future__ import annotations

from services.processor_config import ProcessorConfig

def _compaction_system_prompt(_mp: object) -> str:
    """System prompt for continuity (history) compaction.  §3a / §4a."""
    from services.system_message_prompt import ContinuityCompactionSystemPrompt
    return ContinuityCompactionSystemPrompt().get_prompt()


COMPACTION_CONFIG = ProcessorConfig(
    channel="compaction",
    role="compaction",
    policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
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
