"""PatternMatchProcessor — single-pass LLM matcher over a transcript window.

Fires from SubconsciousWorker._step_pattern_match() when >=50 new transcripts
have accumulated since the last cursor. One forward pass; the model emits
save_pattern / save_graph tool_calls in parallel; ACT loop bounded at 30
iterations. Tool results are intentionally minimal so the model has no useful
content to react to in subsequent iterations.

Decay sweep runs on postTurn: every active behavioral_pattern row not touched
this pass loses 0.005 confidence; rows at <=0 are soft-deleted (active=0).
"""
import json
import logging

from services.database_service import get_shared_db_service
from services.message_processor import MessageProcessor
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PATTERN_MATCH]"
TOP_PATTERN_CAP = 50  # bound for {existing_patterns_block} in system prompt


class PatternMatchProcessor(MessageProcessor):
    CHANNEL = "pattern_match"
    ROLE = "background"
    JOB = "frontal-cortex-unified"
    # save_pattern + save_graph are innate to PMP — pre-injected every
    # iteration. No other processor includes them today; processors that
    # need them in future must opt in by listing them in their own
    # ALWAYS_AVAILABLE. DISCOVERABLE is empty: PMP never widens its scope
    # via find_tools.
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
        # Counters and decay-tracking state read by SavePattern / SaveGraph via
        # current_processor() + getattr.
        self._save_pattern_calls: int = 0
        self._save_graph_calls: int = 0
        self._touched_pattern_ids: set[int] = set()

    # ── Required MessageProcessor overrides ────────────────────────────────

    def getUserDefinition(self) -> str:
        return ""

    def getSystemPrompt(self) -> str:
        block = self._existing_patterns_block()
        return (
            "You are analysing the user's recent transcripts to detect "
            "repeating behavioural patterns and surface durable life-graph "
            "facts.\n\n"
            "You have ONE forward pass. Emit ALL tool calls in parallel. Do "
            "NOT loop — results are intentionally minimal.\n\n"
            "Tools:\n\n"
            "1. save_pattern — record a repeating behaviour with at least 2 "
            "evidence rows. Use snake_case names. REUSE existing names "
            "exactly when reinforcing (case-sensitive).\n"
            f"   Existing patterns currently tracked:\n{block}\n\n"
            "2. save_graph — record a durable fact about the user "
            "(preferences, identity, relationships, places). Pick the right "
            "`kind`. Do NOT use save_graph for repeating behaviours — use "
            "save_pattern.\n\n"
            "Rules:\n"
            "- save_pattern requires >=2 evidence rows in this batch.\n"
            "- Mirror existing pattern names exactly when reinforcing — "
            "case-sensitive.\n"
            "- Skip noise. Skip one-offs. Skip ambiguous interpretations.\n"
            "- Emit everything in this single pass."
        )

    def getUserPrompt(self) -> str:
        db = get_shared_db_service()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM transcript "
                "WHERE id > ? AND id <= ? "
                "AND content IS NOT NULL AND content != '' "
                "ORDER BY id ASC",
                (self._window_start, self._window_end),
            ).fetchall()
        if not rows:
            return "(no transcripts in window)"
        trail = self.getActLoopTrail()
        transcript_block = "\n".join(
            f"[id={r[0]} | {r[1]} | {r[3]}] {r[2]}" for r in rows
        )
        if trail:
            return f"{transcript_block}\n{trail}"
        return transcript_block

    def postTurn(self) -> None:
        """Decay sweep: -0.005 confidence on untouched active rows; soft-
        delete (active=0) rows whose confidence drops to <=0."""
        try:
            self._decay_sweep()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} decay sweep failed: {exc}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _existing_patterns_block(self) -> str:
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
        lines = []
        for (val,) in rows:
            try:
                d = json.loads(val) or {}
                lines.append(
                    f"- {d.get('name')} ({d.get('frequency')}): "
                    f"{d.get('summary')}"
                )
            except Exception:
                continue
        return "\n".join(lines) if lines else "(none yet)"

    def _decay_sweep(self) -> None:
        db = get_shared_db_service()
        with db.connection() as conn:
            # Step A: decrement confidence on untouched rows.
            params: list = []
            touched_filter = ""
            if self._touched_pattern_ids:
                placeholders = ",".join("?" * len(self._touched_pattern_ids))
                touched_filter = f"AND id NOT IN ({placeholders})"
                params = list(self._touched_pattern_ids)
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
            # Step B: soft-delete rows whose confidence dropped to <=0.
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
        logger.info(
            f"{LOG_PREFIX} done save_pattern={self._save_pattern_calls} "
            f"save_graph={self._save_graph_calls} "
            f"touched={len(self._touched_pattern_ids)} "
            f"at={now_iso}"
        )
