from __future__ import annotations

from services.processor_config import ProcessorConfig

# ── User-summary prompt builders and post_turn ────────────────────────────────

_MAX_TRAIT_ROWS = 200
_MAX_PATTERN_ROWS = 25


def _format_pattern_line(content: dict) -> str:
    """Render a single behavioral_pattern content dict as a compact one-liner."""
    name = content.get("name", "unknown")
    freq = content.get("frequency", "?")
    anchor = content.get("time_anchor") or ""
    summary = content.get("summary", "")
    confidence = content.get("confidence", 0)
    last_seen = (content.get("last_seen_at") or "")[:10] or "?"
    anchor_part = f" @ {anchor}" if anchor else ""
    return (
        f"{name} ({freq}{anchor_part}): {summary} "
        f"[confidence={confidence}, last {last_seen}]"
    )


def _user_summary_build_user_prompt(_mp: object) -> str:
    """User-summary user-prompt: user_specific traits + behavioral_patterns."""
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)

    # Section 1: user_specific traits
    try:
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        rows = get_data_graph_service().fetch(
            kinds=["user_specific"],
            limit=_MAX_TRAIT_ROWS,
            order_by="retrieval_weight DESC",
        )
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] trait fetch failed: %s", exc)
        rows = []

    if not rows:
        facts_section = "Facts:\n(no facts available)"
    else:
        lines = [
            f"{r['key']}: {r['value']}"
            for r in rows
            if r.get("key") and r.get("value")
        ]
        facts_section = "Facts:\n" + "\n".join(lines) if lines else "Facts:\n(no facts available)"

    # Section 2: active behavioral_patterns
    active_patterns = []
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            pattern_rows = conn.execute(
                """
                SELECT value
                FROM data_graph
                WHERE kind = 'behavioral_pattern'
                  AND active = 1
                  AND deleted_at IS NULL
                ORDER BY last_confirmed_at DESC
                LIMIT ?
                """,
                (_MAX_PATTERN_ROWS,),
            ).fetchall()
        for (value_json,) in pattern_rows:
            try:
                content = _json.loads(value_json)
                if content:
                    active_patterns.append(content)
            except Exception:
                continue
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] active pattern fetch failed: %s", exc)

    if not active_patterns:
        return facts_section

    pattern_lines = [_format_pattern_line(p) for p in active_patterns]
    patterns_section = (
        "## Behavioural patterns (frequency, last seen)\n"
        + "\n".join(f"- {line}" for line in pattern_lines)
    )
    return facts_section + "\n\n" + patterns_section


def _user_summary_build_system_prompt(_mp: object) -> str:
    """User-summary system prompt from UserSummarySystemPrompt."""
    from services.system_message_prompt import UserSummarySystemPrompt  # noqa: PLC0415
    return UserSummarySystemPrompt().get_prompt()


def _user_summary_post_turn(_mp: object, response_text: str) -> None:
    """Parse JSON {short, long} from response_text and write to data_graph.

    AC-29: receives response_text directly — no on_store hook, no _last_response cache.
    §3b: post_turn(mp, response_text) is the only callback.
    """
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)

    text = (response_text or "").strip()
    if not text:
        _log.warning("[USER_SUMMARY_CONFIG] post_turn: empty LLM response — skipping write")
        return

    # Strip markdown code fences if the model wrapped the JSON.
    stripped = text.removeprefix("```json").removeprefix("```").lstrip()
    stripped = stripped.removesuffix("```").rstrip()

    try:
        parsed = _json.loads(stripped)
    except _json.JSONDecodeError as exc:
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: JSON parse failed (%s) — skipping write. raw=%r",
            exc,
            text[:200],
        )
        return

    if not isinstance(parsed, dict):
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: parsed value is not a dict — skipping write"
        )
        return

    short = (parsed.get("short") or "").strip()
    long_ = (parsed.get("long") or "").strip()

    if not short or not long_:
        _log.warning(
            "[USER_SUMMARY_CONFIG] post_turn: 'short' or 'long' missing/empty — skipping write"
        )
        return

    try:
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        dgs = get_data_graph_service()
        # Write user_summary_long FIRST so crash recovery works correctly.
        # See UserSummaryProcessor.post_turn() docstring for the ordering rationale.
        dgs.store(
            kind="system",
            key="user_summary_long",
            value=long_,
            source="user_summary_config",
        )
        dgs.store(
            kind="system",
            key="user_summary",
            value=short,
            source="user_summary_config",
        )
        _log.info(
            "[USER_SUMMARY_CONFIG] post_turn: wrote user_summary (%d chars) and "
            "user_summary_long (%d chars)",
            len(short),
            len(long_),
        )
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] post_turn: data_graph write failed: %s", exc)


def _should_synthesise() -> bool:
    """Return True when re-synthesis is warranted.

    Logic:
    - latest_trait_ts = MAX(last_confirmed_at) WHERE kind IN
      ('user_specific', 'behavioral_pattern') AND active=1
    - current_summary = row WHERE kind='system' AND key='user_summary' AND active=1

    Cases:
    - No traits or patterns at all → False (nothing to synthesise).
    - Traits/patterns exist, no summary → True (first synthesis needed).
    - Both exist → True only when latest trait/pattern is newer than the summary row.
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.time_utils import parse_utc  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(last_confirmed_at)
                FROM data_graph
                WHERE kind IN ('user_specific', 'behavioral_pattern')
                  AND active = 1
                  AND deleted_at IS NULL
                """
            ).fetchone()
            latest_trait_ts = row[0] if row else None
            if latest_trait_ts is None:
                return False
            summary_row = conn.execute(
                """
                SELECT last_confirmed_at
                FROM data_graph
                WHERE kind = 'system'
                  AND key = 'user_summary'
                  AND active = 1
                  AND deleted_at IS NULL
                LIMIT 1
                """
            ).fetchone()
        if summary_row is None:
            return True
        try:
            trait_dt = parse_utc(latest_trait_ts)
            summary_dt = parse_utc(summary_row[0])
        except Exception as exc:
            _log.error(
                "[USER_SUMMARY_CONFIG] _should_synthesise: parse_utc failed "
                "trait_ts=%r summary_ts=%r: %s",
                latest_trait_ts,
                summary_row[0],
                exc,
            )
            return False
        return trait_dt > summary_dt
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] _should_synthesise failed: %s", exc)
        return False


class UserSummaryConfig(ProcessorConfig):
    """User-summary config — one-shot user synthesis.

    channel/role='user_summary', suppress_history=True, max_iterations=1.
    post_turn parses {short, long} → data_graph (§3b / AC-29).

    The caller gates on _should_synthesise() BEFORE calling
    MessageProcessor.process() — §3c / O1.
    """

    def __init__(self) -> None:
        super().__init__(
            channel="user_summary",
            role="user_summary",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            build_user_prompt=_user_summary_build_user_prompt,
            build_user_definition=lambda _mp: "You are a synthesiser. The user is a real human whose traits you are distilling.",
            build_system_prompt=_user_summary_build_system_prompt,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=_user_summary_post_turn,
        )
