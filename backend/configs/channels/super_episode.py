from __future__ import annotations

from typing import Any

from services.processor_config import ProcessorConfig

# ── Super-episode helper functions ───────────────────────────────────────────
# Moved here from services/super_episode_encoder_processor.py (§1 "Killed" table).
# subconscious_worker imports these from this module.

_SUPER_EP_LOG_PREFIX = "[SUPER_EP]"


def _super_ep_strip_code_fence(text: str) -> str:
    """Remove a single markdown code fence wrapper (```[lang]...```) from text."""
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


def _safe_json_load_object(text: str) -> dict:
    """Parse a JSON object from LLM response text. Returns {} on any failure."""
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
            return parsed
        _log.warning("%s SuperEpisodeEncoder returned non-dict JSON", _SUPER_EP_LOG_PREFIX)
        return {}
    except ValueError:
        _log.warning("%s SuperEpisodeEncoder returned unparseable JSON", _SUPER_EP_LOG_PREFIX)
        return {}


def _parse_super_ep_transcript_ids_field(raw) -> list:
    """Normalize an episode.transcript_ids field (JSON string or list) to a list."""
    import json as _json  # noqa: PLC0415
    if isinstance(raw, str):
        try:
            return _json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    return []


def _collect_transcript_ids(episodes: list) -> set:
    """Return the union of all transcript IDs referenced by *episodes*.

    Handles both JSON-string and list forms of the ``transcript_ids`` field,
    and also expands the ``transcript_id_start``–``transcript_id_end`` range so
    overlap rows are always included.
    """
    ids: set = set()
    for ep in episodes:
        for tid in _parse_super_ep_transcript_ids_field(ep.get("transcript_ids")):
            if tid is None:
                continue
            try:
                ids.add(int(tid))
            except (TypeError, ValueError):
                pass

    starts = [ep["transcript_id_start"] for ep in episodes if ep.get("transcript_id_start") is not None]
    ends = [ep["transcript_id_end"] for ep in episodes if ep.get("transcript_id_end") is not None]
    if starts and ends:
        ids.update(range(min(starts), max(ends) + 1))

    return ids


def _fetch_transcript_spans(t_ids: set, db) -> str:
    """Fetch and format the raw transcript rows for the given transcript IDs.

    Fetches those rows and formats them as ``[id] (ts) role: content`` lines.

    Returns formatted string oldest-first, or '' if no transcript IDs found.
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    if not t_ids:
        return ""

    try:
        placeholders = ",".join("?" * len(t_ids))
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, role, content, tool_name, created_at
                FROM transcript
                WHERE id IN ({placeholders})
                ORDER BY id ASC
                """,
                list(t_ids),
            )
            rows = cursor.fetchall()
            cursor.close()

        lines = []
        for r in rows:
            entry_id, role, content, tool_name, created_at = r
            if tool_name:
                lines.append(f"[{entry_id}] ({created_at}) {role} [{tool_name}]: {content}")
            else:
                lines.append(f"[{entry_id}] ({created_at}) {role}: {content}")
        return "\n".join(lines)

    except Exception as exc:
        _log.warning("%s _fetch_transcript_spans failed: %s", _SUPER_EP_LOG_PREFIX, exc)
        return ""


# ── Super-episode prompt builders ────────────────────────────────────────────


def _super_episode_build_system_prompt(_mp: object) -> str:
    """Super-episode system prompt from SuperEpisodeEncoderSystemPrompt."""
    from services.system_message_prompt import SuperEpisodeEncoderSystemPrompt  # noqa: PLC0415
    return SuperEpisodeEncoderSystemPrompt().get_prompt()


def make_super_episode_config(
    channel: str,
    sources: list[Any],
    spans: Any,
) -> ProcessorConfig:
    """Super-episode encoder config — per-cluster episode synthesis.

    channel/role='super_episode_encoder', suppress_history=True, max_iterations=1.
    post_turn = None (caller owns episode write — §3b / O2).

    sources and spans are captured at factory time so the build_user_prompt
    closure is self-contained per cluster (§3c: cluster loop moves to caller).

    Args:
        channel: The user channel being consolidated (e.g. 'user').
        sources: List of episode dicts for this cluster (each with 'id', 'gist').
        spans:   Raw transcript spans string covering these episodes.
    """
    _sources = sources
    _spans = spans

    def _build_user_prompt(_mp: object) -> str:
        src = "\n\n".join(f"[{e['id']}] {e['gist']}" for e in _sources)
        return (
            f"Source episodes:\n\n{src}\n\n"
            f"Raw transcript spans covering these episodes:\n\n{_spans}"
        )

    def _build_user_definition(_mp: object) -> str:
        return (
            "The user is 'super_episode_encoder' — a background process that "
            "consolidates clusters of related episodes into a single super-episode."
        )

    return ProcessorConfig(
        channel="super_episode_encoder",
        role="super_episode_encoder",
        usage_class="subconscious",
        build_user_prompt=_build_user_prompt,
        build_user_definition=_build_user_definition,
        build_system_prompt=_super_episode_build_system_prompt,
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
