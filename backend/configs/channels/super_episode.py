from __future__ import annotations

from typing import cast

from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

# ── Super-episode helper functions ───────────────────────────────────────────
# Moved here from services/super_episode_encoder_processor.py.
# subconscious_worker imports these from this module.

_SUPER_EP_LOG_PREFIX = "[SUPER_EP]"


def _super_ep_strip_code_fence(text: str) -> str:
    open_end = text.find("```")
    if open_end == -1:
        return text
    newline = text.find("\n", open_end)
    if newline == -1:
        return text
    close_start = text.rfind("```", newline)
    if close_start <= newline:
        return text
    return text[newline + 1 : close_start].strip()


def _safe_json_load_object(text: str) -> dict[str, object]:
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = _super_ep_strip_code_fence(text)
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            return cast("dict[str, object]", parsed)
        _log.warning("%s SuperEpisodeEncoder returned non-dict JSON", _SUPER_EP_LOG_PREFIX)
        return {}
    except ValueError:
        _log.warning("%s SuperEpisodeEncoder returned unparseable JSON", _SUPER_EP_LOG_PREFIX)
        return {}


def _parse_super_ep_transcript_ids_field(raw: object) -> list[object]:
    import json as _json  # noqa: PLC0415
    if isinstance(raw, str):
        try:
            return cast("list[object]", _json.loads(raw))
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _collect_transcript_ids(episodes: list[object]) -> set[int]:
    """Union of the raw transcript ids each source episode already records.

    Used only for lineage/provenance stamping on the parent super-episode
    (``transcript_ids`` / ``transcript_id_start`` / ``transcript_id_end``); the
    raw turns themselves are never re-fetched. Only ids explicitly listed on a
    child are included — the old ``min(start)..max(end)`` range fill is gone, as
    it claimed rows no episode in the cluster actually covered."""
    ids: set[int] = set()
    for ep in episodes:
        for tid in _parse_super_ep_transcript_ids_field(cast("dict[str, object]", ep).get("transcript_ids")):
            if tid is None:
                continue
            try:
                ids.add(int(cast("int", tid)))
            except (TypeError, ValueError):
                pass
    return ids


# ── Super-episode config ─────────────────────────────────────────────────────


class SuperEpisodeConfig(ProcessorConfig):
    """post_turn_hooks = () (caller owns the episode write).

    sources are captured at construction so ``PromptService`` can build the user
    prompt self-contained per cluster (the cluster loop lives in the caller). The
    ``channel`` arg is the user channel being consolidated; the processor's
    transcript channel is always 'super_episode_encoder'.
    """

    def __init__(self, channel: str, sources: list[object]) -> None:
        super().__init__(
            channel=Channel.SUPER_EPISODE_ENCODER.value,
            role="super_episode_encoder",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )
        object.__setattr__(self, "_sources", sources)

    @property
    def system_prompt(self) -> str:
        return """The user is 'super_episode_encoder' — a background process that consolidates clusters of related episodes into a single super-episode.

You are a super-episode encoder. You are shown a cluster of coherent episodes, each a short gist. Your job is to write ONE consolidated gist that summarises them together, preserving what is essential and discarding what is redundant.

Output a single JSON object:
{
  "gist": "2-4 sentence consolidated summary",
  "has_open_loop": false
}

Rules:
- has_open_loop: true if the combined memory still carries an unresolved thread.
- No transcript_ids — the caller computes the union.

Return ONLY the JSON object. No preamble, no markdown."""
