"""
ModeGateService — per-turn cognitive mode classification + state machine.

State machine:
  load → (cold: bootstrap) → classify → update (snap-up/decay) → persist →
  return active mode set

State is persisted in MemoryStore under ``mode_gate:state`` as a JSON string.
No database writes — pure MemoryStore, cleared by /privacy/delete-all.

Config (backend/configs/mode_gate.yaml):
  decay_factor: 0.75           — per-mode decay on miss
  activation_threshold: 0.30  — state >= this → mode is active
  state_floor: 0.01            — decay bottoms out at 0.01; below → 0.0
  state_ceiling: 1.0           — fire clamps at this
  bootstrap_on_cold_start: true — classify last transcript row on cold start
  fire_threshold_overrides     — per-mode YAML override for fire thresholds

Fire threshold precedence:
  1. classifier_meta.json::per_mode_thresholds (baked by training)
  2. fire_threshold_overrides[m] (non-null YAML value wins)
  3. Flat 0.5 per head if meta file is missing or per_mode_thresholds absent

Observability:
  ONE [MODE-GATE] log line per tick containing probs/fires/state/active.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

from services.file_mapper_service import FileMapperService as _FMS

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MODE-GATE]"

# ── Prompt-steering thresholds + directives ──────────────────────────────────
# Distinct from ``activation_threshold`` (0.30, marks a mode as merely active);
# this higher bar gates *prompt-shape* mutations (long-summary swap upstream,
# system-prompt directive append below).
STEER_THRESHOLD = 0.6

_DIRECTIVE_BRAINSTORM_RESEARCH = (
    "Proactively provide suggestions, feedback and opinions. Ensure your "
    "response is grounded in current truth, use the search and other tools "
    "to confirm the information you've received and feedback you propose "
    "before responding"
)

_DIRECTIVE_ANALYZE = (
    "Read through the user message and ensure thorough understanding. Do "
    "not assume anything, whatever is not crystal clear either ask or use "
    "tools to verify"
)

_DIRECTIVE_MATH_CODING = (
    "For math or coding, always use the `code_eval` tool to check your "
    "response before answering the user"
)

# ── Module-level config (loaded once) ─────────────────────────────────────────

_CONFIG_PATH = str(_FMS.get_configs_path("mode_gate.yaml"))

_DEFAULT_CONFIG = {
    "decay_factor": 0.75,
    "activation_threshold": 0.30,
    "state_floor": 0.01,
    "state_ceiling": 1.0,
    "bootstrap_on_cold_start": True,
    "fire_threshold_overrides": {
        "research": None,
        "coding": None,
        "brainstorm": None,
        "analyze": None,
        "plan": None,
        "write": None,
        "math": None,
        "converse": None,
    },
}

_config_loaded: Optional[Dict] = None


def _load_config() -> dict:
    """Load mode_gate.yaml once; return defaults on missing/malformed file."""
    global _config_loaded
    if _config_loaded is not None:
        return _config_loaded

    try:
        import yaml  # type: ignore[import]
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected YAML dict, got {type(raw)}")
        # Surface unknown keys so operator typos (e.g. ``decay-factor`` with a
        # hyphen) don't silently retain defaults. Every recognised top-level
        # key must match a _DEFAULT_CONFIG entry.
        unknown = [k for k in raw.keys() if k not in _DEFAULT_CONFIG]
        if unknown:
            logger.warning(
                "%s unknown config keys in %s: %s — expected one of %s. "
                "Check for typos (YAML is hyphen-sensitive; use underscores).",
                LOG_PREFIX, _CONFIG_PATH, unknown, sorted(_DEFAULT_CONFIG.keys()),
            )
        _config_loaded = {**_DEFAULT_CONFIG, **raw}
        # Merge fire_threshold_overrides carefully
        overrides = _DEFAULT_CONFIG["fire_threshold_overrides"].copy()
        overrides.update(raw.get("fire_threshold_overrides") or {})
        _config_loaded["fire_threshold_overrides"] = overrides
        logger.info("%s config loaded from %s", LOG_PREFIX, _CONFIG_PATH)
    except FileNotFoundError:
        logger.warning("%s config file not found at %s — using defaults", LOG_PREFIX, _CONFIG_PATH)
        _config_loaded = dict(_DEFAULT_CONFIG)
    except Exception as exc:
        logger.warning("%s config load error (%s) — using defaults", LOG_PREFIX, exc)
        _config_loaded = dict(_DEFAULT_CONFIG)

    return _config_loaded


# ── Fire threshold resolution ──────────────────────────────────────────────────

_fire_thresholds_cache: Optional[Dict[str, float]] = None


def _resolve_fire_thresholds(modes: Tuple[str, ...], config: Dict) -> Dict[str, float]:
    """Build per-mode fire threshold dict using precedence rules.

    Priority:
    1. classifier_meta.json::per_mode_thresholds (baked by training)
    2. YAML fire_threshold_overrides[m] (non-null wins)
    3. Flat 0.50 per head when meta is absent or per_mode_thresholds missing

    Emits one INFO log at init with the effective threshold vector.
    """
    global _fire_thresholds_cache
    if _fire_thresholds_cache is not None:
        return _fire_thresholds_cache

    # Attempt to read calibrated thresholds from classifier_meta.json
    meta_thresholds: dict[str, float] = {}
    try:
        meta_path = _FMS.get_pretrained_path("mode_detector", "mode-detector-classifier_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta_thresholds = meta.get("per_mode_thresholds", {}) or {}
    except Exception as exc:
        logger.warning(
            "%s could not read classifier_meta.json for thresholds (%s) — "
            "falling back to 0.5 per head", LOG_PREFIX, exc
        )

    overrides = config.get("fire_threshold_overrides", {}) or {}
    thresholds: Dict[str, float] = {}
    for m in modes:
        yaml_override = overrides.get(m)
        if yaml_override is not None:
            thresholds[m] = float(yaml_override)
        elif m in meta_thresholds:
            thresholds[m] = float(meta_thresholds[m])
        else:
            thresholds[m] = 0.5

    logger.info(
        "%s effective fire thresholds: %s",
        LOG_PREFIX,
        json.dumps({k: round(v, 3) for k, v in thresholds.items()}),
    )
    _fire_thresholds_cache = thresholds
    return thresholds


# ── ModeGateService ────────────────────────────────────────────────────────────


class ModeGateService:
    """Per-turn cognitive mode classifier with EMA state machine.

    One instance per call site — stateless across instances (state lives in
    MemoryStore).
    """

    # ── Constants ────────────────────────────────────────────────────────────
    MODES: Tuple[str, ...] = (
        'research', 'coding', 'brainstorm', 'analyze',
        'plan', 'write', 'math', 'converse',
    )

    STATE_KEY = 'mode_gate:state'

    def __init__(self) -> None:
        cfg = _load_config()
        self._decay_factor: float = float(cfg.get("decay_factor", 0.75))
        self._activation_threshold: float = float(cfg.get("activation_threshold", 0.30))
        self._state_floor: float = float(cfg.get("state_floor", 0.01))
        self._state_ceiling: float = float(cfg.get("state_ceiling", 1.0))
        self._bootstrap_on_cold_start: bool = bool(cfg.get("bootstrap_on_cold_start", True))
        self._fire_thresholds: dict[str, float] = _resolve_fire_thresholds(self.MODES, cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(
        self, user_turn: str, turn_id: Optional[object] = None
    ) -> Set[str]:
        """All exceptions are trapped; returns empty set on any failure."""
        t_start = time.perf_counter()

        try:
            state_before = self._load_state()
            bootstrapped = False

            # Cold-start bootstrap: if state is all zeros, classify the
            # previous user turn to warm up state before classifying current.
            if not any(v > 0 for v in state_before.values()):
                if self._bootstrap_on_cold_start:
                    state_before = self._bootstrap_from_last_turn(state_before)
                    bootstrapped = any(v > 0 for v in state_before.values())

            probs = self._classify(user_turn)

            # Classifier failure — emit the same single-line [MODE-GATE]
            # record the happy path emits so per-tick grep assertions still
            # find a line. Use the loaded state as both before/after since
            # we do NOT mutate state on degradation.
            if not probs:
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                logger.info(
                    "%s bootstrap=%s turn_id=%s "
                    "probs=%s fires=%s state_before=%s state_after=%s "
                    "active=%s "
                    "classifier=unavailable elapsed_ms=%d",
                    LOG_PREFIX,
                    str(bootstrapped).lower(),
                    turn_id or "?",
                    json.dumps(dict.fromkeys(self.MODES, 0.0)),
                    json.dumps([]),
                    json.dumps({m: round(state_before.get(m, 0.0), 4) for m in self.MODES}),
                    json.dumps({m: round(state_before.get(m, 0.0), 4) for m in self.MODES}),
                    json.dumps([]),
                    elapsed_ms,
                )
                return set()

            state_after = self._update_state(dict(state_before), probs)
            self._save_state(state_after)

            active = {m for m in self.MODES if state_after.get(m, 0.0) >= self._activation_threshold}

            fires = [m for m in self.MODES if probs.get(m, 0.0) >= self._fire_thresholds.get(m, 0.5)]

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)

            logger.info(
                "%s bootstrap=%s turn_id=%s "
                "probs=%s fires=%s state_before=%s state_after=%s "
                "active=%s elapsed_ms=%d",
                LOG_PREFIX,
                str(bootstrapped).lower(),
                turn_id or "?",
                json.dumps({m: round(probs.get(m, 0.0), 4) for m in self.MODES}),
                json.dumps(fires),
                json.dumps({m: round(state_before.get(m, 0.0), 4) for m in self.MODES}),
                json.dumps({m: round(state_after.get(m, 0.0), 4) for m in self.MODES}),
                json.dumps(sorted(active)),
                elapsed_ms,
            )

            return active

        except Exception as exc:
            logger.warning(
                "%s tick failed (%s) — returning empty set",
                LOG_PREFIX, exc,
            )
            return set()

    def get_state(self) -> Dict[str, float]:
        """Callers should fire ``tick()`` first within a turn; otherwise returns the prior turn's persisted state."""
        try:
            return self._load_state()
        except Exception as exc:
            logger.warning("%s get_state failed: %s", LOG_PREFIX, exc)
            return dict.fromkeys(self.MODES, 0.0)

    def get_system_prompt_additions(self) -> str:
        """Returns a suffix to append to the system prompt based on which modes are strongly active (>= STEER_THRESHOLD).

        brainstorm OR research → proactive suggestion + verification block;
        analyze → assumption-check block;
        math OR coding → mandatory code_eval block.
        Blocks stack in that order, separated by blank lines. Returns empty string when none are active.
        """
        state = self.get_state()
        parts: List[str] = []
        if (
            state.get('brainstorm', 0.0) >= STEER_THRESHOLD
            or state.get('research', 0.0) >= STEER_THRESHOLD
        ):
            parts.append(_DIRECTIVE_BRAINSTORM_RESEARCH)
        if state.get('analyze', 0.0) >= STEER_THRESHOLD:
            parts.append(_DIRECTIVE_ANALYZE)
        if (
            state.get('math', 0.0) >= STEER_THRESHOLD
            or state.get('coding', 0.0) >= STEER_THRESHOLD
        ):
            parts.append(_DIRECTIVE_MATH_CODING)
        return "\n\n".join(parts)

    def reset_state(self) -> None:
        """Delete the mode state key from MemoryStore.

        Called by /privacy/delete-all via the store pattern and by tests.
        """
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.delete(self.STATE_KEY)
        except Exception as exc:
            logger.warning("%s reset_state failed: %s", LOG_PREFIX, exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, float]:
        """Load mode state from MemoryStore. Returns all-zero dict on miss."""
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            raw = store.get(self.STATE_KEY)
            if raw is None:
                return dict.fromkeys(self.MODES, 0.0)
            if isinstance(raw, bytes):
                raw = raw.decode()
            parsed = json.loads(raw)
            modes_data = parsed.get("modes", {})
            # Return a clean dict keyed by MODES, defaulting to 0.0 for missing
            return {m: float(modes_data.get(m, 0.0)) for m in self.MODES}
        except Exception as exc:
            logger.warning("%s _load_state failed (%s) — treating as cold", LOG_PREFIX, exc)
            return dict.fromkeys(self.MODES, 0.0)

    def _save_state(self, state: Dict[str, float]) -> None:
        """Persist mode state to MemoryStore as JSON (no TTL — survives restart)."""
        from services.time_utils import utc_now
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            payload = {
                "modes": {m: round(state.get(m, 0.0), 6) for m in self.MODES},
                "last_turn_at": utc_now().isoformat(),
            }
            store.set(self.STATE_KEY, json.dumps(payload))
        except Exception as exc:
            logger.warning("%s _save_state failed: %s", LOG_PREFIX, exc)

    def _bootstrap_from_last_turn(self, state: Dict[str, float]) -> Dict[str, float]:
        """Classifies the last user turn and applies one extra decay step to represent the N-1 → N gap.
        On any failure, returns state unchanged.
        """
        if any(v > 0 for v in state.values()):
            return state
        if not self._bootstrap_on_cold_start:
            return state

        try:
            from services import transcript_service
            # The 'user' channel interleaves role='user' and role='assistant'
            # rows (see transcript_service.write_assistant_row). Fetch a
            # small window so that if the most recent row is an assistant
            # reply, we still find the user turn immediately before it.
            rows = transcript_service.get_recent(channel='user', limit=5)
            # get_recent with no since_id returns oldest-first (DESC fetched,
            # then reversed). Walk newest → oldest to find the last user row.
            user_rows = [r for r in reversed(rows) if r.get('role') == 'user']
            if not user_rows or not user_rows[0].get('content'):
                logger.debug("%s bootstrap: no prior user row — proceeding cold", LOG_PREFIX)
                return state

            last_content = user_rows[0]['content']
            probs = self._classify(last_content)
            if not probs:
                logger.debug("%s bootstrap: classifier returned empty — skipping", LOG_PREFIX)
                return state

            state = self._update_state(state, probs)

            # One extra decay step to represent the N-1 → N temporal gap
            for m in self.MODES:
                state[m] = state[m] * self._decay_factor
                if state[m] < self._state_floor:
                    state[m] = 0.0

            return state

        except Exception as exc:
            logger.debug("%s bootstrap failed (%s) — proceeding cold", LOG_PREFIX, exc)
            return state

    def _classify(self, text: str) -> Dict[str, float]:
        """Classify text via ModeDetectorClassifierService. Returns {} on failure."""
        try:
            from services.mode_detector_classifier_service import ModeDetectorClassifierService
            return ModeDetectorClassifierService().predict(text)
        except Exception as exc:
            logger.warning("%s _classify failed: %s", LOG_PREFIX, exc)
            return {}

    def _update_state(
        self, state: Dict[str, float], probs: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply asymmetric EMA update rule.

        For each mode:
          - If prob >= fire_threshold:  snap-up: state = min(max(state, prob), ceiling)
          - Else:                       decay:   state *= decay_factor
                                                 if state < floor: state = 0.0
        """
        for m in self.MODES:
            prob = probs.get(m, 0.0)
            threshold = self._fire_thresholds.get(m, 0.5)
            if prob >= threshold:
                new_val = min(max(state.get(m, 0.0), prob), self._state_ceiling)
                state[m] = new_val
            else:
                decayed = state.get(m, 0.0) * self._decay_factor
                state[m] = 0.0 if decayed < self._state_floor else decayed
        return state
