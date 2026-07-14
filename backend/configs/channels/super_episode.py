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
    """Also expands transcript_id_start..transcript_id_end so overlap rows
    are always included."""
    ids: set[int] = set()
    for ep in episodes:
        for tid in _parse_super_ep_transcript_ids_field(cast("dict[str, object]", ep).get("transcript_ids")):
            if tid is None:
                continue
            try:
                ids.add(int(cast("int", tid)))
            except (TypeError, ValueError):
                pass

    eps = [cast("dict[str, object]", ep) for ep in episodes]
    starts = [cast("int", ep["transcript_id_start"]) for ep in eps if ep.get("transcript_id_start") is not None]
    ends = [cast("int", ep["transcript_id_end"]) for ep in eps if ep.get("transcript_id_end") is not None]
    if starts and ends:
        ids.update(range(min(starts), max(ends) + 1))

    return ids


def _fetch_transcript_spans(t_ids: set[int]) -> str:
    """Returns '' if no transcript IDs found."""
    import logging as _logging  # noqa: PLC0415
    from models.transcript import Transcript
    _log = _logging.getLogger(__name__)
    if not t_ids:
        return ""

    try:
        lines = []
        for r in Transcript.by_ids(sorted(t_ids)):
            if r["tool_name"]:
                lines.append(f"[{r['id']}] ({r['created_at']}) {r['role']} [{r['tool_name']}]: {r['content']}")
            else:
                lines.append(f"[{r['id']}] ({r['created_at']}) {r['role']}: {r['content']}")
        return "\n".join(lines)
    except Exception as exc:
        _log.warning("%s _fetch_transcript_spans failed: %s", _SUPER_EP_LOG_PREFIX, exc)
        return ""


def _spans_for_level(t_ids: set[int], level: int) -> str:
    """Raw transcript spans ground the synthesis only at the leaf level.

    Level 1 (leaf -> super) fetches the raw turns the episode gists were lossily
    summarised from, so the super-gist stays true to what happened. Level >=2
    distils the child gists alone: re-hydrating raw spans there would re-expand
    the whole subtree into the encoder prompt (inverting the distillation and
    blowing the token cap), so we return '' and let each level contract the one
    below it. This is the single source of truth for the level decision; the
    consolidation caller passes the result straight to ``SuperEpisodeConfig``.
    """
    return _fetch_transcript_spans(t_ids) if level == 1 else ""


# ── Super-episode config ─────────────────────────────────────────────────────


class SuperEpisodeConfig(ProcessorConfig):
    """post_turn_hooks = () (caller owns the episode write).

    sources and spans are captured at construction so ``PromptService`` can
    build the user prompt self-contained per cluster (the cluster loop lives
    in the caller). The ``channel`` arg is the user channel being consolidated;
    the processor's transcript channel is always 'super_episode_encoder'.
    """

    def __init__(self, channel: str, sources: list[object], spans: object) -> None:
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
        object.__setattr__(self, "_spans", spans)

    @property
    def system_prompt(self) -> str:
        return """The user is 'super_episode_encoder' — a background process that consolidates clusters of related episodes into a single super-episode.

You are a super-episode encoder. You are shown a cluster of coherent episodes, and sometimes the raw transcript spans that produced them. Your job is to write ONE consolidated gist that summarises them together, preserving what is essential and discarding what is redundant.

Output a single JSON object:
{
  "gist": "2-4 sentence consolidated summary",
  "has_open_loop": false
}

Rules:
- has_open_loop: true if the combined memory still carries an unresolved thread.
- No transcript_ids — the caller computes the union.

Return ONLY the JSON object. No preamble, no markdown."""
