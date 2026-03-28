"""
SituationModelService — Binding layer for situational intelligence.

Aggregates raw signals from 10 existing services into one state object with
composite scores and behavioral directives. All signal sources are lazy-
imported and fail-open: if a source is unavailable, that signal is skipped
and defaults are used. This service never crashes.

Design principles:
- Deterministic. No LLM. <1ms compute per score.
- State cached in MemoryStore at `situation:current` with 5min TTL.
- Per-component `updated_at` timestamps track freshness individually.
- `meta.version` increments on every full update.
- Machine consumers read structured scores; LLM gets plain-language directives.
"""

import json
import logging
import threading
from typing import Optional

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SITUATION]"

# MemoryStore keys
_CURRENT_KEY = "situation:current"
_HISTORY_KEY = "situation:history"
_CURRENT_TTL = 300   # 5 minutes
_HISTORY_MAX = 5     # rolling window for stability

# Spark phase trust mapping (first_contact=0.3, building=0.5, warm=0.7, graduated=0.9)
_SPARK_PHASE_TRUST = {
    "first_contact": 0.3,
    "building": 0.5,
    "warm": 0.7,
    "graduated": 0.9,
}

_SPARK_EARLY_PHASES = {"first_contact", "building"}

# Attention → interruptibility base value
_ATTENTION_INTERRUPTIBILITY = {
    "casual": 0.8,
    "focused": 0.4,
    "deep_focus": 0.1,
    "distracted": 0.9,
    "away": 0.5,
}

# Energy → numeric score
_ENERGY_SCORE = {
    "high": 0.9,
    "moderate": 0.6,
    "low": 0.3,
}


# ── Singleton management ──────────────────────────────────────────────────────

_instance: Optional["SituationModelService"] = None
_instance_lock = threading.Lock()


def get_situation_model_service() -> "SituationModelService":
    """Return the shared SituationModelService singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SituationModelService()
    return _instance


# ── Service ───────────────────────────────────────────────────────────────────

class SituationModelService:
    """
    Binding layer that aggregates ambient, focus, topic, identity, client, engagement,
    relationship phase, world state, and conversation thread signals into a single
    structured state object with composite scores and behavioral directives.
    """

    def __init__(self):
        self._store = None
        self._lock = threading.Lock()

    # ── MemoryStore access ────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from services.memory_client import MemoryClientService
            self._store = MemoryClientService.create_connection()
        return self._store

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current(self) -> dict:
        """
        Return full state object. Reads from MemoryStore cache and recomputes
        if the cache is missing or expired.
        """
        try:
            raw = self._get_store().get(_CURRENT_KEY)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Cache read failed: {e}")

        # Cache miss — compute fresh
        return self.update_on_heartbeat()

    def get_score(self, score_name: str) -> Optional[dict]:
        """
        Return a single composite score dict with `value` and `drivers`.
        Returns None if the score name is unknown.
        """
        state = self.get_current()
        return state.get("scores", {}).get(score_name)

    def update_on_message(self, thread_id: str = None) -> dict:
        """
        Full refresh: collect all signals, recompute all scores, persist.
        Called by the digest worker on every user message.

        Args:
            thread_id: Active thread identifier (used for focus + thread signals).

        Returns:
            Updated state dict.
        """
        with self._lock:
            return self._compute_and_store(thread_id=thread_id, full=True)

    def update_on_heartbeat(self) -> dict:
        """
        Lightweight refresh: re-collect ambient and client signals only.
        Called on ~5min client heartbeat.

        Returns:
            Updated state dict.
        """
        with self._lock:
            return self._compute_and_store(thread_id=None, full=False)

    def get_directive(self) -> Optional[str]:
        """
        Return a behavioral directive string for prompt injection (plain language).
        Returns None when the situation is unremarkable — the `{{situation}}` block
        should be omitted entirely when this returns None.
        """
        state = self.get_current()
        return self._derive_directive(state)

    # ── Signal collectors (all fail-open) ────────────────────────────────────

    def _collect_ambient(self) -> dict:
        """Read ambient inferences from MemoryStore key `ambient:prev_inferences`."""
        defaults = {
            "place": None,
            "energy": None,
            "attention": None,
            "mobility": None,
            "tempo": None,
            "device_context": None,
            "updated_at": utc_now().isoformat(),
        }
        try:
            raw = self._get_store().get("ambient:prev_inferences")
            if not raw:
                return defaults
            data = json.loads(raw)
            return {
                "place": data.get("place"),
                "energy": data.get("energy"),
                "attention": data.get("attention"),
                "mobility": data.get("mobility"),
                "tempo": data.get("tempo"),
                "device_context": data.get("device_context"),
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Ambient signal failed: {e}")
            return defaults

    def _collect_focus(self, thread_id: Optional[str]) -> dict:
        """Read focus session for the given thread from FocusSessionService."""
        defaults = {
            "active": False,
            "topic": None,
            "distraction": 0.0,
            "updated_at": utc_now().isoformat(),
        }
        if not thread_id:
            return defaults
        try:
            from services.focus_session_service import FocusSessionService
            focus = FocusSessionService().get_focus(thread_id)
            if not focus:
                return defaults
            return {
                "active": True,
                "topic": focus.get("topic") or focus.get("description"),
                "distraction": 0.0,
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Focus signal failed: {e}")
            return defaults

    def _collect_topic(self, thread_id: Optional[str]) -> dict:
        """Read current topic from the threads table via DatabaseService."""
        defaults = {
            "name": None,
            "confidence": 0.5,
            "is_new": False,
            "updated_at": utc_now().isoformat(),
        }
        if not thread_id:
            return defaults
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT current_topic FROM threads WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    return {
                        "name": row[0],
                        "confidence": 0.75,
                        "is_new": False,
                        "updated_at": utc_now().isoformat(),
                    }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Topic signal failed: {e}")
        return defaults

    def _collect_identity(self) -> dict:
        """Read identity vectors from IdentityService."""
        defaults = {
            "assertiveness": 0.5,
            "warmth": 0.5,
            "updated_at": utc_now().isoformat(),
        }
        try:
            from services.database_service import get_shared_db_service
            from services.identity_service import IdentityService
            db = get_shared_db_service()
            svc = IdentityService(db)
            vectors = svc.get_vectors()
            if not vectors:
                return defaults
            return {
                "assertiveness": vectors.get("assertiveness", {}).get("current_activation", 0.5),
                "warmth": vectors.get("warmth", {}).get("current_activation", 0.5),
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Identity signal failed: {e}")
            return defaults

    def _collect_client(self) -> dict:
        """Read device, battery, and timezone from ClientContextService."""
        defaults = {
            "device": None,
            "battery": None,
            "tz": None,
            "hour": None,
            "updated_at": utc_now().isoformat(),
        }
        try:
            from services.client_context_service import ClientContextService
            ctx = ClientContextService().get()
            if not ctx:
                return defaults
            device = ctx.get("device", {})
            battery = ctx.get("battery", {})
            tz = ctx.get("timezone")
            hour = None
            local_time = ctx.get("local_time", "")
            if local_time and "T" in local_time:
                try:
                    hour = int(local_time.split("T")[1].split(":")[0])
                except (IndexError, ValueError):
                    pass
            return {
                "device": device.get("class") if isinstance(device, dict) else None,
                "battery": battery.get("level") if isinstance(battery, dict) else None,
                "tz": tz,
                "hour": hour,
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Client signal failed: {e}")
            return defaults

    def _collect_engagement(self) -> dict:
        """Read rolling engagement score from MemoryStore key `proactive:engagement_score`."""
        defaults = {
            "score": 0.7,
            "updated_at": utc_now().isoformat(),
        }
        try:
            raw = self._get_store().get("proactive:engagement_score")
            score = float(raw) if raw else 0.7
            score = max(0.0, min(1.0, score))
            return {
                "score": score,
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Engagement signal failed: {e}")
            return defaults

    def _collect_spark(self) -> dict:
        """
        Read relationship phase from SparkStateService if available.
        SparkStateService is optional — stubs to 'warm' when absent.
        """
        defaults = {
            "phase": "warm",
            "updated_at": utc_now().isoformat(),
        }
        try:
            from services.spark_state_service import SparkStateService
            phase = SparkStateService().get_phase()
            return {
                "phase": phase if isinstance(phase, str) else "warm",
                "updated_at": utc_now().isoformat(),
            }
        except Exception:
            # SparkStateService not yet available — use default
            return defaults

    def _collect_world_state(self, thread_id: Optional[str]) -> dict:
        """Read salient world state items from WorldStateService."""
        defaults = {
            "items": [],
            "updated_at": utc_now().isoformat(),
        }
        try:
            # Use cached model items to avoid heavy DB queries on every message
            store = self._get_store()
            raw = store.get("world_model:items")
            items = []
            if raw:
                payload = json.loads(raw)
                # Flatten all item types into summary dicts
                for item in payload.get("scheduled_items", []):
                    items.append({"type": "scheduled", "label": item.get("message", "")})
                for item in payload.get("persistent_tasks", []):
                    items.append({"type": "task", "label": item.get("description", "")})
                for item in payload.get("lists", []):
                    items.append({"type": "list", "label": item.get("name", "")})
            return {
                "items": items[:5],
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} World state signal failed: {e}")
            return defaults

    def _collect_thread(self, thread_id: Optional[str]) -> dict:
        """Read exchange count and thread age from ThreadConversationService + threads table."""
        defaults = {
            "exchange_count": 0,
            "age_minutes": 0.0,
            "updated_at": utc_now().isoformat(),
        }
        if not thread_id:
            return defaults
        try:
            from services.thread_conversation_service import ThreadConversationService
            from services.database_service import get_shared_db_service
            from services.time_utils import parse_utc

            svc = ThreadConversationService()
            exchange_count = svc.get_exchange_count(thread_id)

            age_minutes = 0.0
            try:
                db = get_shared_db_service()
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT created_at FROM threads WHERE thread_id = ?",
                        (thread_id,),
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        created_at = parse_utc(row[0])
                        now = utc_now()
                        age_minutes = (now - created_at).total_seconds() / 60.0
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Thread age query failed: {e}")

            return {
                "exchange_count": exchange_count,
                "age_minutes": round(age_minutes, 1),
                "updated_at": utc_now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Thread signal failed: {e}")
            return defaults

    def _collect_phase(self, thread_id: Optional[str]) -> dict:
        """
        Read conversation phase from ConversationPhaseService.
        Stubs to unknown/0.5/sustaining when the service isn't available yet.
        """
        defaults = {
            "current": "unknown",
            "momentum": 0.5,
            "direction": "sustaining",
            "updated_at": utc_now().isoformat(),
        }
        if not thread_id:
            return defaults
        try:
            from services.conversation_phase_service import ConversationPhaseService
            phase_data = ConversationPhaseService().get_phase(thread_id)
            if phase_data:
                return {
                    "current": phase_data.get("current", "unknown"),
                    "momentum": float(phase_data.get("momentum", 0.5)),
                    "direction": phase_data.get("direction", "sustaining"),
                    "updated_at": phase_data.get("updated_at", utc_now().isoformat()),
                }
        except Exception:
            # ConversationPhaseService not yet available
            pass
        return defaults

    # ── Composite score computations ──────────────────────────────────────────

    def _score_interruptibility(self, raw: dict) -> dict:
        """
        interruptibility = weighted combination of:
        - attention (ambient): casual=0.8, focused=0.4, deep_focus=0.1, else=0.6
        - focus active: True=0.0, False=1.0
        - energy: low=0.3, moderate=0.6, high=0.9
        - exchange momentum: 1.0 - min(exchange_count / (session_minutes + 1), 1.0)
          (more exchanges per minute = lower interruptibility)
        Weights: attention=0.35, focus=0.30, energy=0.20, momentum=0.15
        """
        ambient = raw.get("ambient", {})
        focus = raw.get("focus", {})
        thread = raw.get("thread", {})

        attention_str = ambient.get("attention") or "casual"
        attention_val = _ATTENTION_INTERRUPTIBILITY.get(attention_str, 0.6)

        focus_active = focus.get("active", False)
        focus_val = 0.0 if focus_active else 1.0

        energy_str = ambient.get("energy") or "moderate"
        energy_val = _ENERGY_SCORE.get(energy_str, 0.6)

        exchange_count = thread.get("exchange_count", 0)
        age_minutes = max(thread.get("age_minutes", 1.0), 1.0)
        rate = exchange_count / age_minutes  # exchanges per minute
        # High rate → low interruptibility. Normalise: rate of 1/min → ~0.5 interruptibility.
        momentum_val = max(0.0, 1.0 - min(rate / 2.0, 1.0))

        value = (
            0.35 * attention_val
            + 0.30 * focus_val
            + 0.20 * energy_val
            + 0.15 * momentum_val
        )

        return {
            "value": round(max(0.0, min(1.0, value)), 3),
            "drivers": {
                "attention": round(attention_val, 3),
                "focus": round(focus_val, 3),
                "energy": round(energy_val, 3),
                "momentum": round(momentum_val, 3),
            },
        }

    def _score_receptiveness(self, raw: dict) -> dict:
        """
        receptiveness = weighted combination of:
        - energy: low=0.3, moderate=0.6, high=0.9
        - engagement score (direct 0-1)
        - spark phase: first_contact=0.3, building=0.5, warm=0.7, graduated=0.9
        Weights: energy=0.30, engagement=0.40, spark_phase=0.30
        """
        ambient = raw.get("ambient", {})
        engagement = raw.get("engagement", {})
        spark = raw.get("spark", {})

        energy_str = ambient.get("energy") or "moderate"
        energy_val = _ENERGY_SCORE.get(energy_str, 0.6)

        engagement_val = float(engagement.get("score", 0.7))
        engagement_val = max(0.0, min(1.0, engagement_val))

        phase = spark.get("phase", "warm")
        spark_val = _SPARK_PHASE_TRUST.get(phase, 0.7)

        value = (
            0.30 * energy_val
            + 0.40 * engagement_val
            + 0.30 * spark_val
        )

        return {
            "value": round(max(0.0, min(1.0, value)), 3),
            "drivers": {
                "energy": round(energy_val, 3),
                "engagement": round(engagement_val, 3),
                "spark_phase": round(spark_val, 3),
            },
        }

    def _score_cognitive_load(self, raw: dict) -> dict:
        """
        cognitive_load = weighted combination of:
        - topic_depth: exchange_count / 20 (capped at 1.0)
        - exchange_count: exchange_count / 30 (capped at 1.0)
        - session_duration: age_minutes / 60 (capped at 1.0)
        - energy: inverted (low=0.7, moderate=0.4, high=0.1)
        Weights: topic_depth=0.25, exchange_count=0.25, session_duration=0.25, energy=0.25
        """
        thread = raw.get("thread", {})
        ambient = raw.get("ambient", {})

        exchange_count = thread.get("exchange_count", 0)
        age_minutes = thread.get("age_minutes", 0.0)

        topic_depth = min(exchange_count / 20.0, 1.0)
        exchange_norm = min(exchange_count / 30.0, 1.0)
        session_norm = min(age_minutes / 60.0, 1.0)

        # Energy inverted: high energy = lower cognitive load
        energy_str = ambient.get("energy") or "moderate"
        energy_forward = _ENERGY_SCORE.get(energy_str, 0.6)
        energy_inv = 1.0 - energy_forward  # low energy → high load

        value = (
            0.25 * topic_depth
            + 0.25 * exchange_norm
            + 0.25 * session_norm
            + 0.25 * energy_inv
        )

        return {
            "value": round(max(0.0, min(1.0, value)), 3),
            "drivers": {
                "topic_depth": round(topic_depth, 3),
                "exchange_count": round(exchange_norm, 3),
                "session_duration": round(session_norm, 3),
                "energy": round(energy_inv, 3),
            },
        }

    def _score_stability(self) -> dict:
        """
        stability = 1.0 - variance of composite scores across last 5 snapshots.
        Reads rolling history from MemoryStore key `situation:history`.
        Falls back to 0.8 when history is insufficient.
        """
        try:
            raw_history = self._get_store().lrange(_HISTORY_KEY, 0, _HISTORY_MAX - 1)
            if len(raw_history) < 2:
                return {
                    "value": 0.8,
                    "drivers": {"score_variance_5m": 0.0},
                }

            composites = []
            for entry_raw in raw_history:
                try:
                    entry = json.loads(entry_raw)
                    # Use mean of all score values as a single composite number
                    scores = entry.get("scores", {})
                    vals = [
                        scores.get(k, {}).get("value", 0.5)
                        for k in ("interruptibility", "receptiveness", "cognitive_load")
                    ]
                    composites.append(sum(vals) / len(vals))
                except Exception:
                    continue

            if len(composites) < 2:
                return {
                    "value": 0.8,
                    "drivers": {"score_variance_5m": 0.0},
                }

            mean = sum(composites) / len(composites)
            variance = sum((x - mean) ** 2 for x in composites) / len(composites)
            stability = max(0.0, min(1.0, 1.0 - variance))

            return {
                "value": round(stability, 3),
                "drivers": {"score_variance_5m": round(variance, 4)},
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Stability score failed: {e}")
            return {
                "value": 0.8,
                "drivers": {"score_variance_5m": 0.0},
            }

    def _score_proactivity_budget(
        self,
        interruptibility: dict,
        spark: dict,
        stability: dict,
    ) -> dict:
        """
        proactivity_budget = weighted combination of:
        - interruptibility.value
        - spark phase trust (same mapping as receptiveness)
        - stability.value
        Weights: interruptibility=0.40, trust=0.35, stability=0.25
        """
        interruptibility_val = interruptibility.get("value", 0.5)
        phase = spark.get("phase", "warm")
        trust_val = _SPARK_PHASE_TRUST.get(phase, 0.7)
        stability_val = stability.get("value", 0.8)

        value = (
            0.40 * interruptibility_val
            + 0.35 * trust_val
            + 0.25 * stability_val
        )

        return {
            "value": round(max(0.0, min(1.0, value)), 3),
            "drivers": {
                "interruptibility": round(interruptibility_val, 3),
                "trust": round(trust_val, 3),
                "stability": round(stability_val, 3),
            },
        }

    # ── Staleness helpers ─────────────────────────────────────────────────────

    def _stalest_freshest(self, raw: dict) -> tuple:
        """
        Return (stalest_input, freshest_input) dicts describing the least and
        most recently updated signal sources.
        """
        now = utc_now()
        ages = {}
        for source, data in raw.items():
            if not isinstance(data, dict):
                continue
            updated_at_str = data.get("updated_at")
            if not updated_at_str:
                continue
            try:
                from services.time_utils import parse_utc
                updated_at = parse_utc(updated_at_str)
                age_seconds = (now - updated_at).total_seconds()
                ages[source] = age_seconds
            except Exception:
                continue

        if not ages:
            return (
                {"source": "unknown", "age_seconds": 0},
                {"source": "unknown", "age_seconds": 0},
            )

        stalest_src = max(ages, key=ages.get)
        freshest_src = min(ages, key=ages.get)
        return (
            {"source": stalest_src, "age_seconds": round(ages[stalest_src], 1)},
            {"source": freshest_src, "age_seconds": round(ages[freshest_src], 1)},
        )

    # ── Directive generation ──────────────────────────────────────────────────

    def _derive_directive(self, state: dict) -> Optional[str]:
        """
        Deterministic directive rules evaluated by priority (highest first).
        Emits at most 2 directives. Returns None when nothing's interesting.
        """
        scores = state.get("scores", {})
        phase_data = state.get("phase", {})
        raw = state.get("raw", {})

        phase = phase_data.get("current", "unknown")
        focus = raw.get("focus", {})
        ambient = raw.get("ambient", {})
        spark = raw.get("spark", {})

        engagement_val = raw.get("engagement", {}).get("score", 0.7)
        stability_val = scores.get("stability", {}).get("value", 0.8)
        cognitive_load_val = scores.get("cognitive_load", {}).get("value", 0.0)
        receptiveness_val = scores.get("receptiveness", {}).get("value", 0.5)

        energy = ambient.get("energy")
        client = raw.get("client", {})
        hour = client.get("hour")

        spark_phase = spark.get("phase", "warm")

        # Rule set: (priority, condition, directive_text)
        rules = []

        # Priority 10: conversation closing
        if phase == "closing":
            rules.append((10, "User is wrapping up — keep it brief, don't introduce new topics."))

        # Priority 9: deep focus active
        focus_topic = focus.get("topic")
        if focus.get("active"):
            label = f"'{focus_topic}'" if focus_topic else "current task"
            rules.append((9, f"User is in deep focus on {label} — only surface urgent items."))

        # Priority 9: disengaging
        if engagement_val < 0.3:
            rules.append((9, "User is disengaging — don't push, let the conversation close naturally."))

        # Priority 8: volatile context
        if stability_val < 0.4:
            rules.append((8, "Context is volatile — be conservative, shorter responses."))

        # Priority 8: evening + low energy
        if energy == "low" and hour is not None and hour >= 20:
            rules.append((8, "Evening, low energy — shorter responses, warmer tone."))

        # Priority 7: cognitive load high
        if cognitive_load_val > 0.75:
            rules.append((7, "User's bandwidth is limited — be concise, use bullets."))

        # Priority 6: engaged and deepening
        if receptiveness_val > 0.8 and phase == "deepening":
            rules.append((6, "User is engaged and open to depth — follow the thread, offer tangents."))

        # Priority 5: opening + early spark phase
        if phase == "opening" and spark_phase in _SPARK_EARLY_PHASES:
            rules.append((5, "New conversation — warm greeting, light touch."))

        if not rules:
            return None

        # Sort by priority descending, emit top 2
        rules.sort(key=lambda x: x[0], reverse=True)
        directives = [text for _, text in rules[:2]]
        return " ".join(directives)

    # ── Core compute pipeline ─────────────────────────────────────────────────

    def _compute_and_store(self, thread_id: Optional[str], full: bool) -> dict:
        """
        Collect signals, compute scores, persist state.

        Args:
            thread_id: Active thread ID (may be None on heartbeat updates).
            full: If True, collect all signals. If False, only ambient + client.
        """
        now_iso = utc_now().isoformat()

        # Collect raw signals
        ambient = self._collect_ambient()
        client = self._collect_client()

        if full:
            focus = self._collect_focus(thread_id)
            topic = self._collect_topic(thread_id)
            identity = self._collect_identity()
            engagement = self._collect_engagement()
            spark = self._collect_spark()
            world_state = self._collect_world_state(thread_id)
            thread = self._collect_thread(thread_id)
            phase = self._collect_phase(thread_id)
        else:
            # Lightweight heartbeat — preserve existing values from cache
            cached = self._load_cached_raw()
            focus = cached.get("focus", self._collect_focus(None))
            topic = cached.get("topic", {"name": None, "confidence": 0.5, "is_new": False, "updated_at": now_iso})
            identity = cached.get("identity", self._collect_identity())
            engagement = cached.get("engagement", self._collect_engagement())
            spark = cached.get("spark", self._collect_spark())
            world_state = cached.get("world_state", {"items": [], "updated_at": now_iso})
            thread = cached.get("thread", {"exchange_count": 0, "age_minutes": 0.0, "updated_at": now_iso})
            phase = cached.get("phase", {"current": "unknown", "momentum": 0.5, "direction": "sustaining", "updated_at": now_iso})

        raw = {
            "ambient": ambient,
            "focus": focus,
            "topic": topic,
            "identity": identity,
            "client": client,
            "engagement": engagement,
            "spark": spark,
            "world_state": world_state,
            "thread": thread,
        }

        # Compute composite scores
        interruptibility = self._score_interruptibility(raw)
        receptiveness = self._score_receptiveness(raw)
        cognitive_load = self._score_cognitive_load(raw)
        stability = self._score_stability()
        proactivity_budget = self._score_proactivity_budget(interruptibility, spark, stability)

        scores = {
            "interruptibility": interruptibility,
            "receptiveness": receptiveness,
            "cognitive_load": cognitive_load,
            "proactivity_budget": proactivity_budget,
            "stability": stability,
        }

        # Determine version
        version = self._next_version()

        # Staleness metadata
        stalest, freshest = self._stalest_freshest(raw)

        state = {
            "raw": raw,
            "scores": scores,
            "phase": phase,
            "meta": {
                "version": version,
                "computed_at": now_iso,
                "stalest_input": stalest,
                "freshest_input": freshest,
            },
        }

        # Persist
        try:
            store = self._get_store()
            store.set(_CURRENT_KEY, json.dumps(state), ex=_CURRENT_TTL)

            # Push snapshot to history for stability computation
            snapshot = {
                "scores": scores,
                "ts": now_iso,
            }
            store.lpush(_HISTORY_KEY, json.dumps(snapshot))
            store.ltrim(_HISTORY_KEY, 0, _HISTORY_MAX - 1)
            store.expire(_HISTORY_KEY, 3600)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Persist failed: {e}")

        directive = self._derive_directive(state)
        logger.info(
            f"{LOG_PREFIX} v{version} | "
            f"interruptibility={interruptibility['value']:.2f} "
            f"receptiveness={receptiveness['value']:.2f} "
            f"cognitive_load={cognitive_load['value']:.2f} "
            f"proactivity={proactivity_budget['value']:.2f} "
            f"stability={stability['value']:.2f} | "
            f"phase={phase.get('current', 'unknown')} | "
            f"directive={repr(directive)}"
        )

        return state

    def _load_cached_raw(self) -> dict:
        """Load the `raw` section from the cached state, or return empty dict."""
        try:
            raw = self._get_store().get(_CURRENT_KEY)
            if raw:
                return json.loads(raw).get("raw", {})
        except Exception:
            pass
        return {}

    def _next_version(self) -> int:
        """Atomically increment and return the version counter from MemoryStore."""
        try:
            return int(self._get_store().incr("situation:version") or 1)
        except Exception:
            return 1
