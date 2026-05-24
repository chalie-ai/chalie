"""GeoPatternProcessor — single-pass LLM geo-spatial pattern extractor.

Fires from SubconsciousWorker._step_geo_patterns() when >=30 new
location-tagged transcripts have accumulated since the last cursor. One
forward pass; the model emits save_pattern / save_graph tool calls to
record location-based behavioural patterns (e.g. "gym on Monday evenings",
"Valletta waterfront for evening walks"). ACT loop bounded at 30 iterations.

Intentionally does NOT run a decay sweep — behavioural_pattern decay is
owned by PatternMatchProcessor. This processor only adds / reinforces
patterns discovered from geo-tagged context.
"""
import json
import logging

from services.database_service import get_shared_db_service
from services.message_processor import MessageProcessor
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GEO_PATTERN]"
TOP_PATTERN_CAP = 50  # bound for {existing_patterns_block} in system prompt


class GeoPatternProcessor(MessageProcessor):
    """LLM processor that extracts location-based behavioural patterns.

    Reads only transcripts with location data attached (lat/lon non-null),
    so the LLM never sees rows without spatial context.
    """

    CHANNEL = "geo_pattern"
    ROLE = "background"
    USAGE_CLASS = 'subconscious'
    LOG_LABEL = 'geo_pattern'
    # save_pattern + save_graph are innate — pre-injected every iteration.
    # DISCOVERABLE is empty: GPP never widens its scope via find_tools.
    ALWAYS_AVAILABLE: list[str] = ["save_pattern", "save_graph"]
    DISCOVERABLE: list[str] = []
    MAX_ITERATIONS = 30
    SKIP_TRANSCRIPT_WRITE = True

    def __init__(
        self,
        window_start: int,
        window_end: int,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(raw_input="", metadata=metadata)
        self._window_start = window_start
        self._window_end = window_end
        # Counters read by SavePattern / SaveGraph via current_processor() + getattr.
        self._save_pattern_calls: int = 0
        self._save_graph_calls: int = 0
        self._save_graph_seen: set[tuple] = set()
        self._touched_pattern_ids: set[int] = set()
        self._cached_transcript_block: str | None = None

    # ── Required MessageProcessor overrides ────────────────────────────────

    def get_user_definition(self) -> str:
        return ""

    def get_system_prompt(self) -> str:
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

    def get_user_prompt(self) -> str:
        if self._cached_transcript_block is None:
            self._cached_transcript_block = self._load_transcript_block()
        existing = self._existing_patterns_block()
        trail = self.get_act_loop_trail()
        parts = [f"Existing patterns:\n{existing}", self._cached_transcript_block]
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def _load_transcript_block(self) -> str:
        """Fetch location-tagged transcripts from the window and format them."""
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at, "
                "location_lat, location_lon, location_name "
                "FROM transcript "
                "WHERE id > ? AND id <= ? "
                "AND location_lat IS NOT NULL AND location_lon IS NOT NULL "
                "AND content IS NOT NULL AND content != '' "
                "ORDER BY id ASC",
                (self._window_start, self._window_end),
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

    def post_turn(self) -> None:
        """Log completion counters. Decay is handled by PatternMatchProcessor."""
        now_iso = utc_now().isoformat()
        logger.info(
            f"{LOG_PREFIX} done save_pattern={self._save_pattern_calls} "
            f"save_graph={self._save_graph_calls} "
            f"touched={len(self._touched_pattern_ids)} "
            f"at={now_iso}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _existing_patterns_block(self) -> str:
        """Return a formatted list of the top-confidence active patterns."""
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT value FROM data_graph "
                "WHERE kind='behavioral_pattern' AND active=1 "
                "AND deleted_at IS NULL "
                "AND json_extract(value, '$.confidence') IS NOT NULL "
                "ORDER BY CAST(json_extract(value, '$.confidence') AS REAL) "
                "DESC LIMIT ?",
                (TOP_PATTERN_CAP,),
            ).fetchall()
        if not rows:
            return "(none yet)"
        patterns = {}
        for (val,) in rows:
            try:
                d = json.loads(val) or {}
                name = d.get("name")
                if name:
                    patterns[name] = d.get("summary", "")
            except Exception:
                continue
        return json.dumps(patterns, indent=2) if patterns else "(none yet)"
