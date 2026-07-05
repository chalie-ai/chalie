from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class EpisodeEncoderConfig(ProcessorConfig):
    """Episode encoder — one-shot, no tools, no transcript writes."""

    # Pinned low: a memory-summarisation pass must not vary its reasoning budget
    # by input (mirrors the pre-rewrite forced ``thinking_level = "low"``).
    # resolve_thinking_mode() gives this config pin priority over the gate.
    thinking_mode: ClassVar[str] = "low"

    def __init__(self) -> None:
        super().__init__(
            channel="episode_encoder",
            role="episode_encoder",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return (
            "The user is 'episode_encoder' — a background process that "
            "summarises transcript windows into memory snapshots."
        )

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
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

    @property
    def system_prompt(self) -> str:
        return """The user is 'episode_encoder' — a background process that summarises transcript windows into memory snapshots.

You are an episodic memory encoder. You read a transcript window plus any memory episodes that were referenced during those turns, and return a JSON array of snapshots.

Each snapshot summarises a coherent moment in the transcript. One snapshot may span multiple transcript entries, and one transcript entry may appear in multiple snapshots when it contributes to distinct narrative threads.

Shape:
{
  "gist": "2-4 sentence summary of what happened in this slice",
  "transcript_ids": [id, id, ...],
  "has_open_loop": false,
  "update_id": null,
  "delete_id": null
}

Field rules:
- has_open_loop: true if this snapshot ends with an unresolved thread — a commitment to future action, an unanswered question, a task paused mid-flight.

Reconsolidation:
- If a new snapshot UPDATES an existing episode you were shown (refines, corrects, or extends it), set `update_id` to that episode's id. Your snapshot replaces it.
- If the transcript makes an existing episode OBSOLETE (the user clarified it was wrong), emit an object with ONLY `delete_id` set and every other field null/empty.
- Otherwise leave both ids null (new episode).

Return ONLY a JSON array. No preamble, no markdown. If nothing meaningful happened, return []."""
