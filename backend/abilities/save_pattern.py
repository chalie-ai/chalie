"""SavePattern — record a repeating behavioural pattern in the data graph.

Reachable when a processor lists ``"save_pattern"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget + decay-tracking state lives on the calling processor (read via
``self.mp``).  PMP initialises ``_save_pattern_calls = 0`` and
``_touched_pattern_ids = set()`` in ``__init__``; this Ability uses ``getattr``
defaults so it remains usable from any processor that opts it in.
"""
import json
import math
import re
from typing import ClassVar

from abilities._budget import BudgetCappedAbility
from abilities._result import ToolResult
from services.database_service import get_shared_db_service
from services.time_utils import utc_now
from utils.data_utils import parse_json_column

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})

# One-line recovery hint carrying a minimal, fully-valid example so a weak model
# can self-correct without re-reading the schema (parity with save_graph).
_EXAMPLE_HINT = (
    "use a snake_case name, e.g. name=morning_run frequency=weekday "
    "summary=user runs on weekdays evidence_transcript_ids=[12, 18]"
)


class SavePattern(BudgetCappedAbility):
    SYSTEM = True

    BUDGET_COUNTER_ATTR: ClassVar[str] = "_save_pattern_calls"
    BUDGET_CAP: ClassVar[int] = 20

    # Action-less single-purpose tool: the dispatcher pre-gate rejects a MISSING
    # or empty name/frequency/summary/evidence_transcript_ids as
    # code=missing-params before run() is reached (precedent: save_graph.py,
    # file_permissions.py). The pre-gate is truthiness-based, so an empty
    # evidence list is rejected there too, while whitespace-only name/summary
    # residue still reaches run().
    ACTION_REQUIRED: ClassVar[dict] = {
        "": ("name", "frequency", "summary", "evidence_transcript_ids")
    }

    def get_name(self) -> str:
        return "save_pattern"

    def get_summary(self) -> str:
        return (
            "Record a repeating behavioural pattern observed in the user's "
            "transcripts. Use snake_case names; reuse existing names exactly "
            "when reinforcing (case-sensitive). Requires at least 2 evidence "
            "transcript ids."
        )

    def get_examples(self) -> list[str]:
        return [
            "user goes for a run every weekday morning",
            "user reads before bed most nights",
            "user checks email first thing each morning",
            "user has coffee around 07:30 on workdays",
            "user meditates on weekends",
            "user takes a walk after lunch on weekdays",
        ]

    def get_search_tooltip(self) -> str:
        return "record behavioural patterns"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "snake_case identifier; mirror existing names when reinforcing.",
            },
            "frequency": {
                "type": "string",
                "enum": ["daily", "weekly", "weekday", "weekend", "ad-hoc"],
            },
            "time_anchor": {
                "type": "string",
                "description": "Optional anchor: '07:00' | 'evening' | 'weekends' | '' if not applicable.",
            },
            "summary": {
                "type": "string",
                "description": "One concise sentence describing the habitual behavior. Not a narrative or episode summary.",
            },
            "evidence_transcript_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["name", "frequency", "summary", "evidence_transcript_ids"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        capped = self.budget_exceeded()
        if capped is not None:
            return capped

        # The validation ORDER and rules are frozen (regex → frequency set →
        # min-2 evidence); only the failure shape changes from a phantom-success
        # dict to a loud ToolResult error, in parity with save_graph.
        name = (params.get("name") or "").strip()
        # The dispatcher pre-gate is truthiness-based, so a whitespace-only name
        # slips past it as a non-empty string and must be rejected here
        # (precedent: save_graph.py, file_permissions.py).
        if not name:
            return ToolResult.err(
                "Missing required parameter(s): name.",
                code="missing-params",
                valid=("name", "frequency", "summary", "evidence_transcript_ids"),
            )
        if not _NAME_PATTERN.fullmatch(name):
            return ToolResult.err(
                f"Invalid pattern name {name!r}; must be snake_case.",
                code="invalid-param",
                hint=_EXAMPLE_HINT,
            )

        frequency = params.get("frequency", "")
        if frequency not in _VALID_FREQUENCIES:
            return ToolResult.err(
                f"Unknown frequency {frequency!r}; not a recognised cadence.",
                code="invalid-param",
                valid=tuple(sorted(_VALID_FREQUENCIES)),
                hint=_EXAMPLE_HINT,
            )

        summary = (params.get("summary") or "").strip()
        if not summary:
            return ToolResult.err(
                "Missing required parameter(s): summary.",
                code="missing-params",
                valid=("name", "frequency", "summary", "evidence_transcript_ids"),
            )

        evidence = params.get("evidence_transcript_ids")
        n_evidence = len(evidence) if isinstance(evidence, list) else 0
        if n_evidence < 2:
            return ToolResult.err(
                f"At least 2 evidence transcript ids are required; got {n_evidence}.",
                code="invalid-param",
                hint="provide at least 2 evidence_transcript_ids from the transcript window",
            )

        validated = {
            "name": name,
            "frequency": frequency,
            "summary": summary,
            "evidence": evidence,
            "time_anchor": params.get("time_anchor", "") or "",
        }
        row_id, confidence_out, reinforced = _upsert_pattern(validated)

        self.bump_budget()
        proc = self.mp
        if proc is not None:
            touched = getattr(proc, "_touched_pattern_ids", None)
            if touched is not None:
                touched.add(row_id)

        body = {"saved": 1}
        if reinforced:
            body["reinforced"] = 1
        body["name"] = validated["name"]
        body["confidence"] = confidence_out
        body["row_id"] = row_id
        return ToolResult.ok(body)


def _upsert_pattern(validated: dict) -> tuple[int, float, bool]:
    """Insert or merge a behavioral_pattern row in data_graph.

    Returns ``(row_id, confidence, reinforced)`` — ``reinforced`` is True when an
    existing pattern was merged, False when a fresh row was inserted.
    """
    now_iso = utc_now().isoformat()
    db = get_shared_db_service()

    with db.connection() as conn:
        existing = conn.execute(
            "SELECT id, value, storage_strength, evidence_count FROM data_graph "
            "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            ("behavioral_pattern", validated["name"]),
        ).fetchone()

        if existing:
            row_id, conf = _update_existing_pattern(conn, existing, validated, now_iso)
            return row_id, conf, True
        row_id, conf = _insert_new_pattern(conn, validated, now_iso)
        return row_id, conf, False


def _update_existing_pattern(conn, existing, validated: dict, now_iso: str) -> tuple[int, float]:
    existing_id, existing_value = existing[0], existing[1]
    old_strength = float(existing[2]) if existing[2] is not None else 0.5
    old_evidence = int(existing[3]) if existing[3] is not None else 1
    prev = parse_json_column(existing_value)
    prev_conf = float(prev.get("confidence") or 0.0)
    new_conf = min(10.0, prev_conf + 7.0)
    merged_evidence = list(dict.fromkeys([*(prev.get("evidence_transcript_ids") or []), *validated["evidence"]]))
    new_value = {
        "name": validated["name"],
        "frequency": validated["frequency"],
        "time_anchor": validated["time_anchor"],
        "summary": validated["summary"],
        "confidence": new_conf,
        "last_seen_at": now_iso,
        "evidence_transcript_ids": merged_evidence,
    }
    new_evidence = old_evidence + 1
    boost = 0.05 / math.log2(new_evidence + 1)
    new_strength = min(1.0, old_strength + boost)
    conn.execute(
        "UPDATE data_graph "
        "SET value=?, last_confirmed_at=?, last_accessed_at=?, source=?, "
        "    evidence_count=?, storage_strength=?, retrieval_weight=1.0 "
        "WHERE id=?",
        (json.dumps(new_value), now_iso, now_iso, "pattern_match", new_evidence, new_strength, existing_id),
    )
    return existing_id, new_conf


def _insert_new_pattern(conn, validated: dict, now_iso: str) -> tuple[int, float]:
    new_value = {
        "name": validated["name"],
        "frequency": validated["frequency"],
        "time_anchor": validated["time_anchor"],
        "summary": validated["summary"],
        "confidence": 7.0,
        "last_seen_at": now_iso,
        "evidence_transcript_ids": list(validated["evidence"]),
    }
    cur = conn.execute(
        "INSERT INTO data_graph "
        "(kind, key, value, first_seen_at, last_confirmed_at, source, active) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        ("behavioral_pattern", validated["name"], json.dumps(new_value), now_iso, now_iso, "pattern_match"),
    )
    return cur.lastrowid, 7.0
