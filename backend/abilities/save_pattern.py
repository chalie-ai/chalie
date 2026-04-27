"""SavePattern — record a repeating behavioural pattern in the data graph.

Reachable when a processor lists ``"save_pattern"`` in its ``ALWAYS_AVAILABLE``
or ``DISCOVERABLE`` tool scope (currently just ``PatternMatchProcessor``).

Budget + decay-tracking state lives on the calling processor (read via
``current_processor()``).  PMP initialises ``_save_pattern_calls = 0`` and
``_touched_pattern_ids = set()`` in ``__init__``; this Ability uses ``getattr``
defaults so it remains usable from any processor that opts it in.
"""
import json
import logging
import re

from abilities._base import Ability
from services.database_service import get_shared_db_service
from services.message_processor import current_processor
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})

_BUDGET_CAP = 20


class SavePattern(Ability):
    NAME = "save_pattern"
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
            "summary": {"type": "string"},
            "evidence_transcript_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["name", "frequency", "summary", "evidence_transcript_ids"],
    }
    TIMEOUT = 10

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        proc = current_processor()
        count = getattr(proc, "_save_pattern_calls", 0) if proc is not None else 0
        if count >= _BUDGET_CAP:
            return {"budget_exceeded": True, "tool": "save_pattern"}

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
        time_anchor = params.get("time_anchor", "") or ""

        now_iso = utc_now().isoformat()
        db = get_shared_db_service()

        with db.connection() as conn:
            existing = conn.execute(
                "SELECT id, value FROM data_graph "
                "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                ("behavioral_pattern", name),
            ).fetchone()

            if existing:
                existing_id, existing_value = existing[0], existing[1]
                try:
                    prev = json.loads(existing_value or "{}") or {}
                except Exception:
                    prev = {}
                prev_conf = float(prev.get("confidence") or 0.0)
                new_conf = min(10.0, prev_conf + 7.0)
                prev_evidence = prev.get("evidence_transcript_ids") or []
                merged = list(dict.fromkeys([*prev_evidence, *evidence]))
                new_value = {
                    "name": name,
                    "frequency": frequency,
                    "time_anchor": time_anchor,
                    "summary": summary,
                    "confidence": new_conf,
                    "last_seen_at": now_iso,
                    "evidence_transcript_ids": merged,
                }
                conn.execute(
                    "UPDATE data_graph "
                    "SET value=?, last_confirmed_at=?, source=? "
                    "WHERE id=?",
                    (json.dumps(new_value), now_iso, "pattern_match", existing_id),
                )
                row_id = existing_id
                confidence_out = new_conf
            else:
                new_value = {
                    "name": name,
                    "frequency": frequency,
                    "time_anchor": time_anchor,
                    "summary": summary,
                    "confidence": 7.0,
                    "last_seen_at": now_iso,
                    "evidence_transcript_ids": list(evidence),
                }
                cur = conn.execute(
                    "INSERT INTO data_graph "
                    "(kind, key, value, first_seen_at, last_confirmed_at, source, active) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (
                        "behavioral_pattern",
                        name,
                        json.dumps(new_value),
                        now_iso,
                        now_iso,
                        "pattern_match",
                    ),
                )
                row_id = cur.lastrowid
                confidence_out = 7.0

        if proc is not None:
            proc._save_pattern_calls = count + 1
            touched = getattr(proc, "_touched_pattern_ids", None)
            if touched is not None:
                touched.add(row_id)

        return {"ok": True, "name": name, "confidence": confidence_out, "row_id": row_id}
