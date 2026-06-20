from __future__ import annotations

from typing import TYPE_CHECKING, cast

from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

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


def _touched_pattern_names(mp: "MessageProcessor") -> set[str]:
    """Pattern names (= data_graph keys) save_pattern wrote this turn, from the DB."""
    import json as _json  # noqa: PLC0415
    from services.act_trail import ActTrail  # noqa: PLC0415
    turn_id = getattr(mp, "turn_id", None)
    if turn_id is None:
        return set()
    names: set[str] = set()
    for row in ActTrail().fetch_by_turn(cast("ProcessorConfig", mp.config).channel, turn_id):
        if row.get("tool_name") != "save_pattern":
            continue
        try:
            name = (_json.loads(cast("str", row.get("params") or "{}")) or {}).get("name")
        except Exception:
            continue
        if name:
            names.add(name)
    return names


class PatternDecayHook(PostTurnHook):
    """Confidence decay sweep: -0.005 on untouched active rows; soft-delete at <=0."""

    def run(self, mp: "MessageProcessor", response_text: str) -> None:
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger(__name__)
        try:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            from services.time_utils import utc_now  # noqa: PLC0415
            touched_names = _touched_pattern_names(mp)
            db = get_shared_db_service()
            with db.connection() as conn:
                # Decrement confidence on untouched rows (excluded by pattern key).
                params: list[object] = []
                touched_filter = ""
                if touched_names:
                    placeholders = ",".join("?" * len(touched_names))
                    touched_filter = f"AND key NOT IN ({placeholders})"
                    params = list(touched_names)
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
            now_iso = utc_now().isoformat()
            _log.info(
                "[PATTERN_CONFIG] done touched=%d at=%s",
                len(touched_names),
                now_iso,
            )
        except Exception as exc:
            _log.warning("[PATTERN_CONFIG] decay sweep failed: %s", exc)


class PatternSkillSyncHook(PostTurnHook):
    """Map the behavioural patterns touched this turn to curated skill playbooks."""

    def run(self, mp: "MessageProcessor", response_text: str) -> None:
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger(__name__)
        try:
            names = _touched_pattern_names(mp)
            if not names:
                return
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
            placeholders = ",".join("?" * len(names))
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT id FROM data_graph "
                    "WHERE kind='behavioral_pattern' AND active=1 "
                    "AND deleted_at IS NULL AND source=? "
                    f"AND key IN ({placeholders})",
                    (cast("ProcessorConfig", mp.config).channel, *names),
                ).fetchall()
            touched_ids = {r[0] for r in rows}
            if not touched_ids:
                return
            from services.skill_association_service import SkillAssociationService  # noqa: PLC0415
            SkillAssociationService().run_pass(touched_ids)
        except Exception as exc:
            _log.warning("[PATTERN_CONFIG] skill association pass failed: %s", exc)


class PatternConfig(ProcessorConfig):
    """Pattern-match config — per-window background pattern recognition."""

    if TYPE_CHECKING:
        _window_start: int
        _window_end: int

    def __init__(self, window_start: int, window_end: int) -> None:
        super().__init__(
            channel="pattern_match",
            role="pattern_match",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["save_pattern", "save_graph"],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn_hooks=(PatternDecayHook(), PatternSkillSyncHook()),
        )
        object.__setattr__(self, "_window_start", window_start)
        object.__setattr__(self, "_window_end", window_end)

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return ""

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """Pattern-match user-prompt: transcripts from window + existing patterns + trail."""
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
            trail = mp._render_act_trail()
            if trail:
                parts.append(trail)
        except Exception:
            pass
        return "\n\n".join(parts)

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
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
