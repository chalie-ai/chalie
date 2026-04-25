"""
PatternExtractor — SubconsciousWorker Step 3.

Extracts behavioural patterns from new super-episodes across six verticals
and four pattern classes.  Results are written to data_graph as rows with
kind='behavioral_pattern'.

Called by SubconsciousWorker.run_once() once per tick (idle-gated).
"""

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from services.background_llm_queue import create_background_llm_proxy
from services.data_graph_service import (
    DataGraphService,
    KIND_BEHAVIORAL_PATTERN,
    KIND_SYSTEM,
)
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# ── Vertical configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _VerticalConfig:
    pattern_class: str          # 'time_routine' | 'preference' | 'recurring_entity' | 'event_cadence'
    min_recurrence: int
    min_sigma_conf: float
    decay_days: int
    entity_kind: Optional[str] = None  # for recurring_entity: 'restaurant' | 'project' | …


_VERTICALS: dict[str, _VerticalConfig] = {
    'meal_times':      _VerticalConfig('time_routine',      min_recurrence=5, min_sigma_conf=3.0, decay_days=90),
    'cuisine_pref':    _VerticalConfig('preference',        min_recurrence=4, min_sigma_conf=2.5, decay_days=180),
    'restaurant_pref': _VerticalConfig('recurring_entity',  min_recurrence=3, min_sigma_conf=2.0, decay_days=180, entity_kind='restaurant'),
    'meetings':        _VerticalConfig('event_cadence',     min_recurrence=3, min_sigma_conf=2.0, decay_days=60),
    'deadlines':       _VerticalConfig('event_cadence',     min_recurrence=2, min_sigma_conf=1.5, decay_days=30),
    'projects':        _VerticalConfig('recurring_entity',  min_recurrence=3, min_sigma_conf=2.0, decay_days=120, entity_kind='project'),
}

# Maximum super-episodes bundled per vertical LLM call — keeps context bounded.
_MAX_SUPER_EPS_PER_CALL = 25

# ── Slot shapes (for prompt construction + JSON schema hints) ─────────────────

_SLOT_SCHEMAS = {
    'time_routine': {
        'event_label': 'str',
        'day_bucket': "'mon'|'tue'|'wed'|'thu'|'fri'|'sat'|'sun'|'weekday'|'weekend'",
        'hour_window': "'HH:MM-HH:MM'",
    },
    'preference': {
        'category': 'str',
        'choice': 'str',
        'weight': 'float (0.0-1.0)',
    },
    'recurring_entity': {
        'kind': 'str',
        'name': 'str',
        'context': 'str (free text)',
    },
    'event_cadence': {
        'event_type': 'str',
        'cadence_days': 'int',
        'next_expected': 'ISO datetime str',
    },
}

# ── Merge predicate helpers ───────────────────────────────────────────────────

def _hour_window_overlap(w1: str, w2: str) -> bool:
    """Return True if two 'HH:MM-HH:MM' windows have any overlap."""
    try:
        s1, e1 = (int(p.replace(':', '')) for p in w1.split('-'))
        s2, e2 = (int(p.replace(':', '')) for p in w2.split('-'))
        return s1 < e2 and s2 < e1
    except Exception:
        return False


def _normalise_name(name: str) -> str:
    """Lowercase and strip punctuation for fuzzy entity name matching."""
    return re.sub(r'[^\w\s]', '', name.lower()).strip()


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _entity_name_match(name_a: str, name_b: str) -> bool:
    """True if names are equivalent via normalised edit distance ≤2 or substring containment."""
    na, nb = _normalise_name(name_a), _normalise_name(name_b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return _edit_distance(na, nb) <= 2


# ── Pattern key derivation ────────────────────────────────────────────────────

def _pattern_key(vertical: str, pattern_class: str, slots: dict) -> str:
    """Derive a deterministic key for a pattern row.

    Key scheme:  ``{vertical}:{16-hex-sha256-of-merge-predicate}``

    Only merge-predicate slots participate in the hash so that cosmetic
    differences (weight, context, cadence_days) never produce a duplicate row.
    The hash is stable across Python invocations because it uses only str
    operations on sorted inputs.
    """
    if pattern_class == 'time_routine':
        canonical = f"{slots.get('event_label', '').lower()}|{slots.get('day_bucket', '').lower()}"
    elif pattern_class == 'preference':
        canonical = f"{slots.get('category', '').lower()}|{slots.get('choice', '').lower()}"
    elif pattern_class == 'recurring_entity':
        canonical = f"{slots.get('kind', '').lower()}|{_normalise_name(slots.get('name', ''))}"
    elif pattern_class == 'event_cadence':
        canonical = slots.get('event_type', '').lower()
    else:
        canonical = json.dumps(slots, sort_keys=True)

    digest = hashlib.sha256(f"{vertical}:{canonical}".encode()).hexdigest()[:16]
    return f"{vertical}:{digest}"


# ── Merge functions (pure, testable without DB) ───────────────────────────────

def merge_time_routine(existing: dict, observation: dict, now_iso: str) -> dict:
    """Merge an incoming time_routine observation into an existing pattern dict.

    Mutates and returns a copy of ``existing`` with updated fields.
    The caller is responsible for persisting the result.
    """
    merged = dict(existing)
    merged['recurrence'] = existing.get('recurrence', 1) + 1
    merged['sigma_confidence'] = existing.get('sigma_confidence', 0.0) + observation.get('confidence', 0.0)
    merged['last_seen'] = now_iso
    ids = list(existing.get('evidence_super_ep_ids', []))
    for ep_id in observation.get('evidence_super_ep_ids', []):
        if ep_id not in ids:
            ids.append(ep_id)
    merged['evidence_super_ep_ids'] = ids
    return merged


def merge_preference(existing: dict, observation: dict, now_iso: str) -> dict:
    """Merge a preference observation.  Weight = max(existing, new)."""
    merged = dict(existing)
    merged['recurrence'] = existing.get('recurrence', 1) + 1
    merged['sigma_confidence'] = existing.get('sigma_confidence', 0.0) + observation.get('confidence', 0.0)
    merged['last_seen'] = now_iso
    ids = list(existing.get('evidence_super_ep_ids', []))
    for ep_id in observation.get('evidence_super_ep_ids', []):
        if ep_id not in ids:
            ids.append(ep_id)
    merged['evidence_super_ep_ids'] = ids

    # Weight is normalised affinity — do not sum.
    existing_weight = (existing.get('slots') or {}).get('weight', 0.0)
    new_weight = (observation.get('slots') or {}).get('weight', 0.0)
    slots = dict(existing.get('slots') or {})
    slots['weight'] = max(existing_weight, new_weight)
    merged['slots'] = slots
    return merged


def merge_recurring_entity(existing: dict, observation: dict, now_iso: str) -> dict:
    """Merge a recurring_entity observation.

    Keeps the first-seen name form.  Updates context to the most recent
    observation.
    """
    merged = dict(existing)
    merged['recurrence'] = existing.get('recurrence', 1) + 1
    merged['sigma_confidence'] = existing.get('sigma_confidence', 0.0) + observation.get('confidence', 0.0)
    merged['last_seen'] = now_iso
    ids = list(existing.get('evidence_super_ep_ids', []))
    for ep_id in observation.get('evidence_super_ep_ids', []):
        if ep_id not in ids:
            ids.append(ep_id)
    merged['evidence_super_ep_ids'] = ids

    # Keep original name; update context to most recent observation.
    slots = dict(existing.get('slots') or {})
    new_context = (observation.get('slots') or {}).get('context')
    if new_context:
        slots['context'] = new_context
    merged['slots'] = slots
    return merged


def merge_event_cadence(existing: dict, observation: dict, now_iso: str) -> dict:
    """Merge an event_cadence observation.

    Recomputes cadence_days as the running mean of inter-event gaps and
    updates next_expected = last_seen + cadence_days.
    """
    merged = dict(existing)
    merged['recurrence'] = existing.get('recurrence', 1) + 1
    merged['sigma_confidence'] = existing.get('sigma_confidence', 0.0) + observation.get('confidence', 0.0)
    merged['last_seen'] = now_iso
    ids = list(existing.get('evidence_super_ep_ids', []))
    for ep_id in observation.get('evidence_super_ep_ids', []):
        if ep_id not in ids:
            ids.append(ep_id)
    merged['evidence_super_ep_ids'] = ids

    slots = dict(existing.get('slots') or {})
    obs_slots = observation.get('slots') or {}

    # Recompute cadence as running mean of all cadence observations.
    # We track a simple running mean using recurrence count.
    existing_cadence = existing.get('slots', {}).get('cadence_days', 0)
    new_cadence = obs_slots.get('cadence_days', existing_cadence)
    n = merged['recurrence']
    if n > 1 and existing_cadence > 0:
        slots['cadence_days'] = round(((existing_cadence * (n - 1)) + new_cadence) / n)
    else:
        slots['cadence_days'] = new_cadence

    # next_expected = last_seen + cadence_days
    try:
        last_dt = parse_utc(now_iso)
        slots['next_expected'] = (last_dt + timedelta(days=slots['cadence_days'])).isoformat()
    except Exception:
        slots['next_expected'] = obs_slots.get('next_expected', '')

    merged['slots'] = slots
    return merged


_MERGE_FUNCTIONS = {
    'time_routine': merge_time_routine,
    'preference': merge_preference,
    'recurring_entity': merge_recurring_entity,
    'event_cadence': merge_event_cadence,
}


# ── Match helpers (find existing pattern to merge into) ───────────────────────

def _patterns_match(pattern_class: str, existing_slots: dict, obs_slots: dict) -> bool:
    """Return True if an observation should be merged into an existing pattern row."""
    if pattern_class == 'time_routine':
        return (
            existing_slots.get('event_label', '').lower() == obs_slots.get('event_label', '').lower()
            and existing_slots.get('day_bucket', '').lower() == obs_slots.get('day_bucket', '').lower()
            and _hour_window_overlap(
                existing_slots.get('hour_window', ''),
                obs_slots.get('hour_window', ''),
            )
        )
    elif pattern_class == 'preference':
        return (
            existing_slots.get('category', '').lower() == obs_slots.get('category', '').lower()
            and existing_slots.get('choice', '').lower() == obs_slots.get('choice', '').lower()
        )
    elif pattern_class == 'recurring_entity':
        return (
            existing_slots.get('kind', '').lower() == obs_slots.get('kind', '').lower()
            and _entity_name_match(
                existing_slots.get('name', ''),
                obs_slots.get('name', ''),
            )
        )
    elif pattern_class == 'event_cadence':
        return existing_slots.get('event_type', '').lower() == obs_slots.get('event_type', '').lower()
    return False


# ── Promotion logic ───────────────────────────────────────────────────────────

def _should_promote(content: dict, vcfg: _VerticalConfig) -> bool:
    """Return True if a candidate pattern meets the promotion gate."""
    return (
        content.get('status') == 'candidate'
        and content.get('recurrence', 0) >= vcfg.min_recurrence
        and content.get('sigma_confidence', 0.0) >= vcfg.min_sigma_conf
        and len(set(content.get('evidence_super_ep_ids', []))) >= 2
    )


# ── Service ───────────────────────────────────────────────────────────────────

_instance: Optional['PatternExtractor'] = None
_instance_lock = threading.Lock()


def get_pattern_extractor() -> 'PatternExtractor':
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PatternExtractor()
    return _instance


class PatternExtractor:
    """Extract behavioural patterns from super-episodes and write to data_graph.

    Called by SubconsciousWorker.run_once() as Step 3.  Each call iterates the
    six verticals; one LLM call per vertical (when new super-episodes exist).
    """

    def __init__(self, db_service=None):
        from services.database_service import get_shared_db_service
        self._db = db_service or get_shared_db_service()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_once(self) -> dict:
        """Extract patterns across all six verticals.

        Returns a summary dict keyed by vertical name::

            {
                'meal_times': {'observations': 3, 'candidates_added': 2, 'promotions': 1},
                'cuisine_pref': {'observations': 0, 'candidates_added': 0, 'promotions': 0},
                'restaurant_pref': {'error': '...'},
                ...
            }

        Failure of one vertical never blocks the others.
        Per-tick LLM-call budget = len(_VERTICALS) (one per vertical when needed).
        """
        summary = {}
        for vertical_name, vcfg in _VERTICALS.items():
            try:
                summary[vertical_name] = self._extract_vertical(vertical_name, vcfg)
            except Exception:
                logger.exception("[PATTERN EXTRACTOR] %s failed", vertical_name)
                summary[vertical_name] = {'error': 'vertical extraction failed'}
        return summary

    # ── Per-vertical extraction ───────────────────────────────────────────────

    def _extract_vertical(self, vertical_name: str, vcfg: _VerticalConfig) -> dict:
        watermark = self._read_watermark(vertical_name)
        new_super_eps = self._fetch_new_super_episodes(watermark)

        if not new_super_eps:
            return {'observations': 0, 'candidates_added': 0, 'promotions': 0}

        # Cap to most recent N to keep context bounded.
        capped = new_super_eps[-_MAX_SUPER_EPS_PER_CALL:]

        existing_patterns = self._fetch_existing_patterns(vertical_name)
        observations = self._call_llm(vertical_name, vcfg, capped, existing_patterns)

        candidates_added = 0
        promotions = 0

        for obs in observations:
            try:
                added, promoted = self._apply_observation(vertical_name, vcfg, obs, existing_patterns)
                candidates_added += 1 if added else 0
                promotions += 1 if promoted else 0
            except Exception:
                logger.exception(
                    "[PATTERN EXTRACTOR] %s: failed to apply observation %s",
                    vertical_name, obs,
                )

        # Update watermark to max created_at of consumed batch.
        max_created = max(
            parse_utc(ep.get('created_at', '')).isoformat()
            for ep in capped
        )
        self._write_watermark(vertical_name, max_created)

        return {
            'observations': len(observations),
            'candidates_added': candidates_added,
            'promotions': promotions,
        }

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        vertical_name: str,
        vcfg: _VerticalConfig,
        super_eps: list[dict],
        existing_patterns: list[dict],
    ) -> list[dict]:
        """Issue one structured-output LLM call for this vertical.

        Returns a list of observation dicts on success; empty list on failure.
        """
        system_prompt = self._build_system_prompt(vertical_name, vcfg, existing_patterns)
        user_message = self._build_user_message(super_eps)

        llm = create_background_llm_proxy("pattern-extractor")
        try:
            response = llm.send_message(system_prompt, user_message)
        except Exception as exc:
            logger.warning("[PATTERN EXTRACTOR] %s: LLM call raised: %s", vertical_name, exc)
            return []

        if response is None:
            logger.warning("[PATTERN EXTRACTOR] %s: LLM returned None", vertical_name)
            return []

        return self._parse_llm_response(response.text, vertical_name)

    def _build_system_prompt(
        self,
        vertical_name: str,
        vcfg: _VerticalConfig,
        existing_patterns: list[dict],
    ) -> str:
        slot_schema = json.dumps(_SLOT_SCHEMAS[vcfg.pattern_class], indent=2)
        existing_json = json.dumps(
            [json.loads(p['value']) for p in existing_patterns if p.get('value')],
            indent=2,
        )

        entity_kind_note = ''
        if vcfg.entity_kind:
            entity_kind_note = f'\nThe "kind" slot for this vertical must be "{vcfg.entity_kind}".'

        return f"""You are a behavioural pattern extractor.

VERTICAL: {vertical_name}
PATTERN CLASS: {vcfg.pattern_class}
SLOT SCHEMA:
{slot_schema}
{entity_kind_note}

EXISTING PATTERNS FOR THIS VERTICAL (for deduplication reference):
{existing_json}

Extract observations from the super-episode gists provided in the user message.
Each observation must include:
- slots: {{"...slot fields..."}}
- evidence_super_ep_ids: ["<episode_id>", ...]
- confidence: 0.0 to 1.0

Return ONLY a JSON object with this exact shape:
{{"observations": [<observation>, ...]}}

Rules:
- Only extract observations that are clearly evident in the super-episode text.
- If nothing matches this vertical, return {{"observations": []}}.
- Do not invent data not present in the source gists.
- confidence reflects how strongly the episode text supports the observation.
"""

    def _build_user_message(self, super_eps: list[dict]) -> str:
        lines = []
        for ep in super_eps:
            lines.append(f"[{ep.get('id', '?')}] {ep.get('gist', '')}")
        return "Super-episode gists:\n\n" + "\n\n".join(lines)

    def _parse_llm_response(self, text: str, vertical_name: str) -> list[dict]:
        """Parse LLM JSON response; return empty list on any failure."""
        try:
            # Strip markdown code fences if present.
            clean = text.strip()
            if clean.startswith('```'):
                clean = re.sub(r'^```[^\n]*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean.rstrip())
            data = json.loads(clean)
            observations = data.get('observations', [])
            if not isinstance(observations, list):
                logger.warning("[PATTERN EXTRACTOR] %s: 'observations' is not a list", vertical_name)
                return []
            return observations
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("[PATTERN EXTRACTOR] %s: JSON parse failed: %s", vertical_name, exc)
            return []

    # ── Observation application ───────────────────────────────────────────────

    def _apply_observation(
        self,
        vertical_name: str,
        vcfg: _VerticalConfig,
        observation: dict,
        existing_patterns: list[dict],
    ) -> tuple[bool, bool]:
        """Merge or insert an observation.

        Returns (new_row_inserted: bool, promotion_fired: bool).
        """
        obs_slots = observation.get('slots') or {}
        pattern_class = vcfg.pattern_class
        now_iso = utc_now().isoformat()

        matched_pattern = self._find_matching_pattern(pattern_class, obs_slots, existing_patterns)

        if matched_pattern:
            # Merge into existing pattern.
            existing_content = json.loads(matched_pattern['value'])
            merge_fn = _MERGE_FUNCTIONS[pattern_class]
            updated = merge_fn(existing_content, observation, now_iso)

            promoted = False
            if _should_promote(updated, vcfg):
                updated['status'] = 'active'
                updated['last_confirmed_at'] = now_iso
                promoted = True

            self._persist_pattern(matched_pattern['key'], updated)
            # Refresh cache entry.
            for i, p in enumerate(existing_patterns):
                if p['key'] == matched_pattern['key']:
                    existing_patterns[i] = dict(matched_pattern)
                    existing_patterns[i]['value'] = json.dumps(updated)
                    break

            return False, promoted
        else:
            # Insert new candidate.
            content = self._build_new_content(vertical_name, vcfg, observation, now_iso)
            key = _pattern_key(vertical_name, pattern_class, obs_slots)

            self._persist_pattern(key, content)
            # Add to in-memory cache so subsequent observations in the same batch
            # can find and merge into this new row.
            existing_patterns.append({'key': key, 'value': json.dumps(content)})
            return True, False

    def _find_matching_pattern(
        self,
        pattern_class: str,
        obs_slots: dict,
        existing_patterns: list[dict],
    ) -> Optional[dict]:
        """Return the first existing pattern row whose slots match obs_slots."""
        for p in existing_patterns:
            try:
                content = json.loads(p['value'])
                if _patterns_match(pattern_class, content.get('slots', {}), obs_slots):
                    return p
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _build_new_content(
        self,
        vertical_name: str,
        vcfg: _VerticalConfig,
        observation: dict,
        now_iso: str,
    ) -> dict:
        return {
            'vertical': vertical_name,
            'class': vcfg.pattern_class,
            'slots': observation.get('slots') or {},
            'recurrence': 1,
            'sigma_confidence': observation.get('confidence', 0.0),
            'first_seen': now_iso,
            'last_seen': now_iso,
            'evidence_super_ep_ids': list(observation.get('evidence_super_ep_ids') or []),
            'status': 'candidate',
            'source': 'llm',
            'decay_days': vcfg.decay_days,
        }

    def _persist_pattern(self, key: str, content: dict) -> None:
        """Upsert a pattern row in data_graph.

        Patterns own their mutation semantics: one active row per (kind, key),
        in-place value updates on merge, plain INSERT on first-seen. This does
        NOT route through ``DataGraphService.store()`` because store()'s
        contradiction policies (additive insert, cosine_supersede,
        lut_canonicalize) are designed for traits/facts and would either
        accumulate stale duplicates or require embeddings the extractor does
        not own.
        """
        value = json.dumps(content)
        now_iso = utc_now().isoformat()
        with self._db.connection() as conn:
            existing = conn.execute(
                "SELECT id FROM data_graph "
                "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
                "LIMIT 1",
                (KIND_BEHAVIORAL_PATTERN, key),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE data_graph "
                    "SET value=?, last_confirmed_at=?, source=? "
                    "WHERE id=?",
                    (value, now_iso, 'pattern-extractor', existing['id']),
                )
            else:
                conn.execute(
                    "INSERT INTO data_graph "
                    "(kind, key, value, first_seen_at, last_confirmed_at, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (KIND_BEHAVIORAL_PATTERN, key, value, now_iso, now_iso,
                     'pattern-extractor'),
                )

    # ── Watermark helpers ─────────────────────────────────────────────────────

    def _watermark_key(self, vertical_name: str) -> str:
        return f'pattern_watermark:{vertical_name}'

    def _read_watermark(self, vertical_name: str) -> str:
        """Return ISO watermark string; epoch-zero if not set."""
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM data_graph "
                    "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
                    "ORDER BY last_confirmed_at DESC LIMIT 1",
                    (KIND_SYSTEM, self._watermark_key(vertical_name)),
                ).fetchone()
                if row and row[0]:
                    return row[0]
        except Exception as exc:
            logger.warning("[PATTERN EXTRACTOR] watermark read failed for %s: %s", vertical_name, exc)
        return '1970-01-01T00:00:00+00:00'

    def _write_watermark(self, vertical_name: str, iso_value: str) -> None:
        """Persist the watermark for a vertical."""
        try:
            dgs = DataGraphService(self._db)
            dgs.store(
                KIND_SYSTEM,
                self._watermark_key(vertical_name),
                iso_value,
                source='pattern-extractor',
            )
        except Exception as exc:
            logger.warning("[PATTERN EXTRACTOR] watermark write failed for %s: %s", vertical_name, exc)

    # ── DB queries ────────────────────────────────────────────────────────────

    def _fetch_new_super_episodes(self, watermark: str) -> list[dict]:
        """Fetch super-episodes (consolidated_from IS NOT NULL) created after watermark."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, gist, created_at FROM episodes "
                    "WHERE consolidated_from IS NOT NULL "
                    "  AND consolidated_from != '[]' "
                    "  AND deleted_at IS NULL "
                    "  AND created_at > ? "
                    "ORDER BY created_at ASC",
                    (watermark,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("[PATTERN EXTRACTOR] super-episode fetch failed: %s", exc)
            return []

    def _fetch_existing_patterns(self, vertical_name: str) -> list[dict]:
        """Fetch all active behavioral_pattern rows for a vertical from data_graph."""
        results = []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph "
                    "WHERE kind=? AND active=1 AND deleted_at IS NULL "
                    "  AND key LIKE ?",
                    (KIND_BEHAVIORAL_PATTERN, f'{vertical_name}:%'),
                ).fetchall()
                for r in rows:
                    try:
                        # Validate JSON is parseable.
                        json.loads(r['value'])
                        results.append({'key': r['key'], 'value': r['value']})
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception as exc:
            logger.warning("[PATTERN EXTRACTOR] existing patterns fetch failed for %s: %s", vertical_name, exc)
        return results
