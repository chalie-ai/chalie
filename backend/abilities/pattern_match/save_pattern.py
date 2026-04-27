"""SavePattern — processor-internal helper for PatternMatchProcessor.

Not an Ability subclass and not registered in AbilityRegistry. Lives in a
subdirectory so the registry's shallow ``glob("*.py")`` walk skips it. Only
PatternMatchProcessor imports this module.
"""
import json
import re

from services.database_service import get_shared_db_service
from services.time_utils import utc_now

# Strict ASCII snake_case: leading lowercase letter, then lowercase letters,
# digits, underscores. `str.isalnum()` accepts Unicode digits/letters which we
# don't want — the LLM should only emit ASCII identifiers here.
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

_VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})


class SavePattern:
    NAME = "save_pattern"
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

    TOOL_SCHEMA: dict = {
        "name": NAME,
        "description": (
            "Record a repeating behavioural pattern observed in the user's "
            "transcripts. Use snake_case names; reuse existing names exactly "
            "when reinforcing (case-sensitive). Requires at least 2 evidence "
            "transcript ids."
        ),
        "input_schema": INPUT_SCHEMA,
    }

    def execute(self, args: dict, processor: object) -> dict:
        """Persist a behavioural pattern row.

        processor must be a PatternMatchProcessor instance — reads/writes
        _save_pattern_calls and _touched_pattern_ids.
        """
        if processor._save_pattern_calls >= 20:
            return {"budget_exceeded": True, "tool": "save_pattern"}

        name = args.get("name", "")
        if not name or not _NAME_PATTERN.fullmatch(name):
            return {"error": "invalid_name", "name": name}
        frequency = args.get("frequency", "")
        if frequency not in _VALID_FREQUENCIES:
            return {"error": "invalid_frequency", "frequency": frequency}
        summary = (args.get("summary") or "").strip()
        if not summary:
            return {"error": "empty_summary"}
        evidence = args.get("evidence_transcript_ids") or []
        # Spec requires >=2 evidence rows per save_pattern call. The system
        # prompt states the rule explicitly; this validator is the enforcement.
        if not isinstance(evidence, list) or len(evidence) < 2:
            return {
                "error": "insufficient_evidence",
                "required_min": 2,
                "got": len(evidence) if isinstance(evidence, list) else 0,
            }
        time_anchor = args.get("time_anchor", "") or ""

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

        processor._touched_pattern_ids.add(row_id)
        processor._save_pattern_calls += 1

        return {"ok": True, "name": name, "confidence": confidence_out, "row_id": row_id}
