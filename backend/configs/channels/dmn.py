# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from services.processor_config import ProcessorConfig

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE, DEFAULT_DISCOVERABLE

# ── DMN prompt builders ───────────────────────────────────────────────────────

_EPISODE_RETRIEVAL_WEIGHT_FLOOR = 0.3
_DMN_EPISODE_LOOKBACK_DAYS = 30
_DMN_EPISODE_LIMIT = 50


def _dmn_build_user_definition(_mp: object) -> str:
    """DMN runs as a background process — no human user definition needed."""
    return (
        "The user is 'proactive_thought' — a special background process "
        "that represents your own reflections on recent activity."
    )


def _dmn_fetch_user_synthesis() -> str:
    """Read user synthesis from data_graph.

    Prefers user_summary_long for richer DMN reflection context.
    Falls back to user_summary. Returns '' when neither row exists.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM data_graph "
                "WHERE kind = 'system' "
                "  AND key IN ('user_summary', 'user_summary_long') "
                "  AND active = 1 AND deleted_at IS NULL",
            ).fetchall()
        by_key = {row[0]: row[1] for row in rows if row[1]}
        return by_key.get("user_summary_long") or by_key.get("user_summary") or ""
    except Exception as exc:
        _log.warning("[DMN_CONFIG] _dmn_fetch_user_synthesis failed: %s", exc)
        return ""


def _dmn_fetch_recent_episodes() -> str:
    """Retrieve recent, non-decayed user-channel episodes as a numbered list."""
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, gist, salience, created_at "
                "FROM episodes "
                "WHERE deleted_at IS NULL "
                "  AND channel = 'user' "
                "  AND retrieval_weight >= ? "
                "  AND ( "
                "      last_accessed_at >= datetime('now', ?) "
                "      OR created_at >= datetime('now', ?) "
                "  ) "
                "ORDER BY retrieval_weight DESC, created_at DESC "
                "LIMIT ?",
                (
                    _EPISODE_RETRIEVAL_WEIGHT_FLOOR,
                    f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                    f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                    _DMN_EPISODE_LIMIT,
                ),
            ).fetchall()
        lines = []
        for i, (ep_id, gist, salience, created_at) in enumerate(rows, 1):
            ts = (created_at or "")[:16].replace("T", " ")
            lines.append(f"{i}. [{ts}] (salience={salience}) {gist or ''}")
        return "\n".join(lines)
    except Exception as exc:
        _log.warning("[DMN_CONFIG] _dmn_fetch_recent_episodes failed: %s", exc)
        return ""


def _dmn_build_user_prompt(mp: object) -> str:
    """DMN user-message: user synthesis + filtered recent episodes + ACT trail."""
    parts: list[str] = []
    synthesis = _dmn_fetch_user_synthesis()
    if synthesis:
        parts.append(f"## About the User\n{synthesis}")
    episodes_text = _dmn_fetch_recent_episodes()
    if episodes_text:
        parts.append(f"## Episodes\n{episodes_text}")
    try:
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        if trail and isinstance(trail, str):
            parts.append(trail)
    except Exception:
        pass
    return "\n\n".join(parts)


def _dmn_build_system_prompt(_mp: object) -> str:
    """DMN system prompt from DMNSystemMessagePrompt."""
    from services.system_message_prompt import DMNSystemMessagePrompt  # noqa: PLC0415
    return DMNSystemMessagePrompt().get_prompt()


DMN_CONFIG = ProcessorConfig(
    channel="dmn",
    role="proactive_thought",
    usage_class="subconscious",
    build_user_prompt=_dmn_build_user_prompt,
    build_user_definition=_dmn_build_user_definition,
    build_system_prompt=_dmn_build_system_prompt,
    always_available=DEFAULT_ALWAYS_AVAILABLE,
    discoverable=DEFAULT_DISCOVERABLE,
    # DMN is a background reflection loop — it must never spawn delegate work
    # (the old single ``subagent`` tool was retired in favour of the four
    # delegate tools; block all of them to preserve that intent).
    blocked=frozenset({"web_search", "research", "web_browse", "summariser"}),
    max_iterations=100,
    skip_transcript=False,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
    post_turn=None,
)
"""DMN background channel.  §3a / §8b.  post_turn=None (metrics moved to gateway §4e)."""
