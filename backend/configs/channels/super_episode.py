from __future__ import annotations

import json
import logging

from typing import TYPE_CHECKING, cast

from services.processor_config import ProcessorConfig
from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt
from services.transcript_service import Transcript

if TYPE_CHECKING:
    from services.database_service import DatabaseService
    from services.message_processor import MessageProcessor

# ── Super-episode helper functions ───────────────────────────────────────────
# Moved here from services/super_episode_encoder_processor.py (§1 "Killed" table).
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
    _log = logging.getLogger(__name__)
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = _super_ep_strip_code_fence(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return cast("dict[str, object]", parsed)
        _log.warning("%s SuperEpisodeEncoder returned non-dict JSON", _SUPER_EP_LOG_PREFIX)
        return {}
    except ValueError:
        _log.warning("%s SuperEpisodeEncoder returned unparseable JSON", _SUPER_EP_LOG_PREFIX)
        return {}


def _parse_super_ep_transcript_ids_field(raw: object) -> list[object]:
    if isinstance(raw, str):
        try:
            return cast("list[object]", json.loads(raw))
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


def _fetch_transcript_spans(t_ids: set[int], db: "DatabaseService") -> str:
    """Returns '' if no transcript IDs found."""
    _log = logging.getLogger(__name__)
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


# ── Super-episode config ─────────────────────────────────────────────────────

if TYPE_CHECKING:
    class _WithSourcesSpans:
        _sources: list[object]
        _spans: object


class SuperEpisodeConfig(ProcessorConfig):
    """§3b / O2 — post_turn_hooks = () (caller owns the episode write).

    sources and spans are captured at construction so get_user_prompt is
    self-contained per cluster (§3c: cluster loop moves to caller). The
    ``channel`` arg is the user channel being consolidated; the processor's
    transcript channel is always 'super_episode_encoder'.
    """

    def __init__(self, channel: str, sources: list[object], spans: object) -> None:
        super().__init__(
            channel="super_episode_encoder",
            role="super_episode_encoder",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )
        object.__setattr__(self, "_sources", sources)
        object.__setattr__(self, "_spans", spans)

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return (
            "The user is 'super_episode_encoder' — a background process that "
            "consolidates clusters of related episodes into a single super-episode."
        )

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        _self = cast("_WithSourcesSpans", self)
        src = "\n\n".join(
            f"[{cast('dict[str, object]', e)['id']}] {cast('dict[str, object]', e)['gist']}"
            for e in _self._sources
        )
        return (
            f"Source episodes:\n\n{src}\n\n"
            f"Raw transcript spans covering these episodes:\n\n{_self._spans}"
        )

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        """OLD assembly ``f"{user_def}\n\n{body}"``."""
        return f"{self.get_user_definition(mp)}\n\n{SuperEpisodeEncoderSystemPrompt().get_prompt()}"
