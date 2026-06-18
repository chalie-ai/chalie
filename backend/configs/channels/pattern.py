from __future__ import annotations

from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig

# ── Pattern-match prompt builders and post_turn ───────────────────────────────

_TOP_PATTERN_CAP = 50


def _pattern_existing_patterns_block() -> str:
    """Return the top-confidence active behavioral_pattern rows as JSON."""
    import json as _json  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT value FROM data_graph "
                "WHERE kind='behavioral_pattern' AND active=1 "
                "AND deleted_at IS NULL "
                "AND json_extract(value, '$.confidence') IS NOT NULL "
                "ORDER BY CAST(json_extract(value, '$.confidence') AS REAL) "
                "DESC LIMIT ?",
                (_TOP_PATTERN_CAP,),
            ).fetchall()
        if not rows:
            return "(none yet)"
        patterns = {}
        for (val,) in rows:
            try:
                d = _json.loads(val) or {}
                name = d.get("name")
                if name:
                    patterns[name] = d.get("summary", "")
            except Exception:
                continue
        return _json.dumps(patterns, indent=2) if patterns else "(none yet)"
    except Exception as exc:
        _log.warning("[PATTERN_CONFIG] existing_patterns_block failed: %s", exc)
        return "(none yet)"


class PatternDecayHook(PostTurnHook):
    """Confidence decay sweep: -0.005 on untouched active rows; soft-delete at <=0.

    §3b / §4e / §4.8 — no metrics recorded here (metrics moved to send gateway).
    """

    def run(self, mp, response_text: str) -> None:
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger(__name__)
        try:
            touched_ids: set = getattr(mp, "_touched_pattern_ids", set()) or set()
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            from services.time_utils import utc_now  # noqa: PLC0415
            db = get_shared_db_service()
            with db.connection() as conn:
                # Decrement confidence on untouched rows.
                params: list = []
                touched_filter = ""
                if touched_ids:
                    placeholders = ",".join("?" * len(touched_ids))
                    touched_filter = f"AND id NOT IN ({placeholders})"
                    params = list(touched_ids)
                conn.execute(
                    f"""
                    UPDATE data_graph
                    SET value = json_set(
                          value,
                          '$.confidence',
                          MAX(0.0, CAST(json_extract(value, '$.confidence') AS REAL) - 0.005)
                        )
                    WHERE kind = 'behavioral_pattern'
                      AND active = 1
                      AND deleted_at IS NULL
                      AND json_extract(value, '$.confidence') IS NOT NULL
                      {touched_filter}
                    """,
                    params,
                )
                # Soft-delete rows whose confidence dropped to <=0.
                conn.execute(
                    """
                    UPDATE data_graph
                    SET active = 0
                    WHERE kind = 'behavioral_pattern'
                      AND active = 1
                      AND CAST(json_extract(value, '$.confidence') AS REAL) <= 0.0
                    """,
                )
            save_pattern_calls = getattr(mp, "_save_pattern_calls", 0)
            save_graph_calls = getattr(mp, "_save_graph_calls", 0)
            now_iso = utc_now().isoformat()
            _log.info(
                "[PATTERN_CONFIG] done save_pattern=%d save_graph=%d "
                "touched=%d at=%s",
                save_pattern_calls,
                save_graph_calls,
                len(touched_ids),
                now_iso,
            )
        except Exception as exc:
            _log.warning("[PATTERN_CONFIG] decay sweep failed: %s", exc)


def _pattern_init_instance_state(mp: object) -> None:
    """Initialise per-instance counter/state attrs that SavePattern/SaveGraph read."""
    if not hasattr(mp, "_save_pattern_calls"):
        mp._save_pattern_calls = 0  # type: ignore[attr-defined]
    if not hasattr(mp, "_save_graph_calls"):
        mp._save_graph_calls = 0  # type: ignore[attr-defined]
    if not hasattr(mp, "_save_graph_seen"):
        mp._save_graph_seen = set()  # type: ignore[attr-defined]
    if not hasattr(mp, "_touched_pattern_ids"):
        mp._touched_pattern_ids = set()  # type: ignore[attr-defined]


class PatternConfig(ProcessorConfig):
    """Pattern-match config — per-window background pattern recognition.

    channel/role='pattern_match', suppress_history=True, max_iterations=100.
    post_turn_hooks = (PatternDecayHook(),) — confidence decay sweep (§3b).

    Counter/state attrs are lazily initialised by get_user_prompt on the first
    call so the caller does not need to pre-set them.
    """

    def __init__(self, window_start: int, window_end: int) -> None:
        super().__init__(
            channel="pattern_match",
            role="pattern_match",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["save_pattern", "save_graph"],
            max_iterations=100,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn_hooks=(PatternDecayHook(),),
        )
        object.__setattr__(self, "_window_start", window_start)
        object.__setattr__(self, "_window_end", window_end)

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        """Pattern-match user-prompt: transcripts from window + existing patterns + trail."""
        # Lazy-init per-instance state so SavePattern/SaveGraph find it.
        _pattern_init_instance_state(mp)
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger(__name__)
        # Bidirectional dependency: the per-source allowlist lives in
        # services/source_profiles.py; this is the pattern-window consumer.
        from services.source_profiles import (  # noqa: PLC0415
            pattern_user_channels_sql,
            non_compaction_sql,
        )
        try:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, role, content, created_at FROM transcript "
                    "WHERE id > ? AND id <= ? "
                    "AND content IS NOT NULL AND content != '' "
                    # Only user-behaviour channels feed the pattern window.
                    # Background loops (dmn, delegate:*, external-agent:*,
                    # skills_building) and compaction rows are internal, not
                    # observed user behaviour — the allowlist excludes them.
                    f"AND {pattern_user_channels_sql()} "
                    f"AND {non_compaction_sql()} "
                    "ORDER BY id ASC",
                    (self._window_start, self._window_end),
                ).fetchall()
            if not rows:
                return "(no transcripts in window)"
            transcript_block = "\n".join(
                f"[id={r[0]} | {r[1]} | {r[3]}] {r[2]}" for r in rows
            )
        except Exception as exc:
            _log.warning("[PATTERN_CONFIG] transcript fetch failed: %s", exc)
            transcript_block = "(transcript fetch failed)"

        existing = _pattern_existing_patterns_block()
        parts = [f"Existing patterns:\n{existing}", transcript_block]
        try:
            trail = mp._render_act_trail()  # type: ignore[attr-defined]
            if trail:
                parts.append(trail)
        except Exception:
            pass
        return "\n\n".join(parts)

    def get_system_prompt(self, mp) -> str:
        """Pattern-match system prompt (inlined from PatternMatchProcessor.get_system_prompt)."""
        return (
            "You are analysing the user's recent transcripts to detect "
            "repeating behavioural patterns and surface durable life-graph "
            "facts.\n\n"
            "You have ONE forward pass. Emit ALL tool calls in parallel. Do "
            "NOT loop — results are intentionally minimal.\n\n"
            "save_pattern summaries must be 1 sentence and concise. "
            "Examples:\n"
            "- User walks to work every morning at 09:00\n"
            "- User prefers cold-beverages\n"
            "- User goes to the gym daily, except Sundays\n"
            "- User's gym schedule is: Monday weights, Tuesday boxing\n"
            "Do NOT write narratives, episode summaries, or date-stamped "
            "event logs. The summary is a distilled habit, not a story.\n\n"
            "Rules:\n"
            "- save_pattern requires >=2 evidence rows in this batch.\n"
            "- Mirror existing pattern names exactly when reinforcing — "
            "case-sensitive.\n"
            "- Do NOT use save_graph for repeating behaviours — use "
            "save_pattern.\n"
            "- Skip noise. Skip one-offs. Skip ambiguous interpretations.\n"
            "- Emit everything in this single pass."
        )
