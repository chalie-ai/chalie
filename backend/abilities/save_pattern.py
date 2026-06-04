"""SavePattern — record a repeating behavioural pattern in the data graph.

Reachable when a processor lists ``"save_pattern"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget + decay-tracking state lives on the calling processor (read via
``self.MessageProcessor``).  PMP initialises ``_save_pattern_calls = 0`` and
``_touched_pattern_ids = set()`` in ``__init__``; this Ability uses ``getattr``
defaults so it remains usable from any processor that opts it in.
"""
import json
import logging
import math
import re

from abilities._ability import Ability
from services.database_service import get_shared_db_service
from services.time_utils import utc_now
from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})

_BUDGET_CAP = 20


class SavePattern(Ability):
    NAME = "save_pattern"
    SEARCH_TOOLTIP = "record behavioural patterns"
    SYSTEM = True
    SUMMARY = (
        "Record a repeating behavioural pattern observed in the user's "
        "transcripts. Use snake_case names; reuse existing names exactly "
        "when reinforcing (case-sensitive). Requires at least 2 evidence "
        "transcript ids."
    )
    EXAMPLES = [
        "user goes for a run every weekday morning",
        "user reads before bed most nights",
        "user checks email first thing each morning",
        "user has coffee around 07:30 on workdays",
        "user meditates on weekends",
        "user takes a walk after lunch on weekdays",
    ]
    INPUT_SCHEMA = {
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

    def run(self, params: dict) -> dict:
        proc = self.MessageProcessor
        count = getattr(proc, "_save_pattern_calls", 0) if proc is not None else 0
        if count >= _BUDGET_CAP:
            return {"budget_exceeded": True, "tool": "save_pattern"}

        validated = _validate_pattern_params(params)
        if "error" in validated:
            return validated

        row_id, confidence_out = _upsert_pattern(validated)

        if proc is not None:
            proc._save_pattern_calls = count + 1
            touched = getattr(proc, "_touched_pattern_ids", None)
            if touched is not None:
                touched.add(row_id)

        return {"ok": True, "name": validated["name"], "confidence": confidence_out, "row_id": row_id}


def _validate_pattern_params(params: dict) -> dict:
    """Validate a save_pattern request. Returns the normalized fields, or an error dict."""
    name = params.get("name", "")
    if not name or not _NAME_PATTERN.fullmatch(name):
        return {"error": "invalid_name", "name": name}
    frequency = params.get("frequency", "")
    if frequency not in _VALID_FREQUENCIES:
        return {"error": "invalid_frequency", "frequency": frequency}
    summary = (params.get("summary") or "").strip()
    if not summary:
        return {"error": "empty_summary"}
    evidence = params.get("evidence_transcript_ids") or []
    if not isinstance(evidence, list) or len(evidence) < 2:
        return {
            "error": "insufficient_evidence",
            "required_min": 2,
            "got": len(evidence) if isinstance(evidence, list) else 0,
        }
    return {
        "name": name,
        "frequency": frequency,
        "summary": summary,
        "evidence": evidence,
        "time_anchor": params.get("time_anchor", "") or "",
    }


def _upsert_pattern(validated: dict) -> tuple[int, float]:
    """Insert or merge a behavioral_pattern row in data_graph. Returns (row_id, confidence)."""
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
            return _update_existing_pattern(conn, existing, validated, now_iso)
        return _insert_new_pattern(conn, validated, now_iso)


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
