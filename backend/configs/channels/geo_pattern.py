from __future__ import annotations

from services.processor_config import ProcessorConfig

from configs.channels.pattern import _pattern_existing_patterns_block

# ── Geo-pattern prompt builders ───────────────────────────────────────────────


def _geo_pattern_load_transcript_block(window_start: int, window_end: int) -> str:
    """Fetch location-tagged transcripts from the window and format them."""
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    # Bidirectional dependency: the per-source allowlist lives in
    # services/source_profiles.py; this is the geo-window consumer.
    from services.source_profiles import geo_user_channels_sql, non_compaction_sql  # noqa: PLC0415
    try:
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at, "
                "location_lat, location_lon, location_name "
                "FROM transcript "
                "WHERE id > ? AND id <= ? "
                "AND location_lat IS NOT NULL AND location_lon IS NOT NULL "
                "AND content IS NOT NULL AND content != '' "
                f"AND {geo_user_channels_sql()} "
                f"AND {non_compaction_sql()} "
                "ORDER BY id ASC",
                (window_start, window_end),
            ).fetchall()
        if not rows:
            return "(no location-tagged transcripts in window)"
        lines = []
        for r in rows:
            row_id, role, content, created_at, lat, lon, place_name = r
            location_str = f"{lat},{lon}"
            if place_name:
                location_str = f"{lat},{lon} {place_name}"
            lines.append(
                f"[id={row_id} | {role} | {created_at} | {location_str}] {content}"
            )
        return "\n".join(lines)
    except Exception as exc:
        _log.warning("[GEO_CONFIG] transcript block failed: %s", exc)
        return "(transcript fetch failed)"


class GeoConfig(ProcessorConfig):
    """Geo-pattern config — per-window background geo recognition."""

    def __init__(self, window_start: int, window_end: int) -> None:
        super().__init__(
            channel="geo_pattern",
            role="geo_pattern",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=["save_pattern", "save_graph"],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )
        object.__setattr__(self, "_window_start", window_start)
        object.__setattr__(self, "_window_end", window_end)

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        """Geo-pattern user-prompt: location transcripts + existing patterns + trail."""
        block = _geo_pattern_load_transcript_block(self._window_start, self._window_end)
        existing = _pattern_existing_patterns_block()
        parts = [f"Existing patterns:\n{existing}", block]
        try:
            trail = mp._render_act_trail()  # type: ignore[attr-defined]
            if trail:
                parts.append(trail)
        except Exception:
            pass
        return "\n\n".join(parts)

    def get_system_prompt(self, mp) -> str:
        """Geo-pattern system prompt (inlined from GeoPatternProcessor.get_system_prompt)."""
        return (
            "You are analysing the user's recent location-tagged transcripts "
            "to detect geo-spatial behavioural patterns — routines, habits, "
            "and durable facts tied to specific physical places.\n\n"
            "You have ONE forward pass. Emit ALL tool calls in parallel. Do "
            "NOT loop — results are intentionally minimal.\n\n"
            "save_pattern summaries must be 1 sentence and concise. "
            "Examples:\n"
            "- User walks to work every morning at 09:00\n"
            "- User goes to the gym daily, except Sundays\n"
            "- User's gym schedule is: Monday weights, Tuesday boxing\n"
            "Do NOT write narratives, episode summaries, or date-stamped "
            "event logs. The summary is a distilled habit, not a story.\n\n"
            "Rules:\n"
            "- save_pattern requires >=2 evidence rows in this batch.\n"
            "- Mirror existing pattern names exactly when reinforcing — "
            "case-sensitive.\n"
            "- Only emit patterns where physical place is central. Another "
            "processor handles location-independent patterns.\n"
            "- Do NOT use save_graph for repeating behaviours — use "
            "save_pattern.\n"
            "- Skip noise. Skip one-offs. Skip ambiguous interpretations.\n"
            "- Emit everything in this single pass."
        )
