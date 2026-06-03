from __future__ import annotations

from services.processor_config import ProcessorConfig

# ── Episode encoder prompt builders ──────────────────────────────────────────


def _episode_encoder_build_user_definition(_mp: object) -> str:
    return (
        "The user is 'episode_encoder' — a background process that "
        "summarises transcript windows into memory snapshots."
    )


def _episode_encoder_build_user_prompt(mp: object) -> str:
    """Episode encoder user-prompt: transcript window + referenced episodes.

    Reads _window and _referenced from the mp instance (set by the caller
    before calling MessageProcessor.process()).
    """
    window = getattr(mp, "_window", "") or ""
    referenced = getattr(mp, "_referenced", "") or ""
    parts = [
        "Transcript window — each line is `[id] (timestamp) role: content`:",
        "",
        window,
    ]
    if referenced:
        parts.extend([
            "",
            "Episodes referenced during these turns (candidates for update / delete):",
            "",
            referenced,
        ])
    return "\n".join(parts)


def _episode_encoder_build_system_prompt(_mp: object) -> str:
    """Episode encoder system prompt from EpisodeEncoderSystemPrompt."""
    from services.system_message_prompt import EpisodeEncoderSystemPrompt  # noqa: PLC0415
    return EpisodeEncoderSystemPrompt().get_prompt()


EPISODE_ENCODER_CONFIG = ProcessorConfig(
    channel="episode_encoder",
    role="episode_encoder",
    policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
    build_user_prompt=_episode_encoder_build_user_prompt,
    build_user_definition=_episode_encoder_build_user_definition,
    build_system_prompt=_episode_encoder_build_system_prompt,
    always_available=[],
    discoverable=[],
    blocked=frozenset(),
    max_iterations=1,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""Episode encoder — one-shot, no tools, no transcript writes.  §3a."""
