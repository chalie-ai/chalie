"""
Self-Model Service — Foundational interoception for Chalie.

Continuously aggregates three signal categories into a cached MemoryStore snapshot:
  1. Epistemic  — memory warmth, recall reliability, topic depth
  2. Operational — thread health, provider status, queue depth, memory pressure
  3. Capability  — tool inventory, capability categories, provider features

Design:
  - Deterministic, zero-LLM, <50ms refresh
  - Follows AmbientInferenceService pattern (cached, always-fresh)
  - Noteworthy list is EMPTY when healthy — only populated on degradation
  - Each noteworthy item carries a severity weight (0.0-1.0) for downstream consumers
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from services.memory_client import MemoryClientService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SELF MODEL]"

CACHE_KEY = "self_model:snapshot"
CACHE_TTL = 45  # seconds
REFRESH_INTERVAL = 30  # background thread cycle

# Critical cognitive jobs — if any lack an assigned provider, that's noteworthy
CRITICAL_JOBS = frozenset({
    'frontal-cortex', 'cognitive-triage', 'cognitive-drift',
})

# Tool-agnostic capability categories derived from manifest documentation keywords
CATEGORY_KEYWORDS = {
    "search": ["search", "query", "find", "lookup", "retrieve"],
    "media": ["image", "video", "audio", "photo", "media"],
    "communication": ["email", "message", "notify", "send", "slack"],
    "data": ["database", "spreadsheet", "csv", "data", "analytics"],
    "productivity": ["calendar", "task", "todo", "remind", "schedule"],
    "development": ["code", "git", "deploy", "build", "test"],
    "news": ["news", "article", "headline", "feed"],
}

# Severity weights for noteworthy triggers
SEVERITY_MISSING_PROVIDER = 0.8
SEVERITY_DEAD_THREADS = 0.6
SEVERITY_STALE_HEARTBEAT = 0.5
SEVERITY_QUEUE_CONGESTION = 0.4
SEVERITY_HIGH_RECALL_FAILURE = 0.3
SEVERITY_LOW_ACTIVATION = 0.2
SEVERITY_CAPABILITY_GAP = 0.2


def _utc_now() -> datetime:
    """Timezone-aware UTC now. Inlined to avoid dependency on time_utils."""
    return datetime.now(timezone.utc)


class SelfModelService:
    """Aggregates epistemic, operational, and capability signals into a cached snapshot."""

    def __init__(self, db_service=None):
        self._db = db_service
        self._store = MemoryClientService.create_connection()

    def _get_db(self):
        if self._db is None:
            from services.database_service import get_shared_db_service
            self._db = get_shared_db_service()
        return self._db

    # ── Public API ──────────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return cached snapshot (sub-ms hit) or refresh if expired."""
        raw = self._store.get(CACHE_KEY)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return self._refresh()

    def has_noteworthy_state(self) -> bool:
        """Fast gate — True only when something is degraded."""
        snapshot = self.get_snapshot()
        return len(snapshot.get("noteworthy", [])) > 0

    def get_memory_richness(self) -> float:
        """0.0 (empty system) to 1.0 (rich memory), from cached snapshot.

        Composite score from episode count, concept count, trait count,
        and epistemic warmth. Workers use this to self-regulate: skip
        expensive cycles when memory is too thin to produce useful results.
        """
        snapshot = self.get_snapshot()

        # Extract counts from operational.memory_pressure
        pressure = snapshot.get("operational", {}).get("memory_pressure", {})
        episode_count = pressure.get("episode_count", 0)
        concept_count = pressure.get("concept_count", 0)
        trait_count = pressure.get("trait_count", 0)

        # Epistemic warmth (current conversation context)
        context_warmth = snapshot.get("epistemic", {}).get("context_warmth", 0.0)

        # Weighted composite — saturate at reasonable ceilings
        score = (
            0.35 * min(1.0, episode_count / 50)
            + 0.25 * min(1.0, concept_count / 30)
            + 0.20 * min(1.0, trait_count / 10)
            + 0.20 * context_warmth
        )
        return round(score, 3)

    def format_for_prompt(self) -> str:
        """
        Format self-awareness as a prompt section with behavioral guidance.

        Returns empty string when healthy (zero token cost).
        When degraded, includes both the signals AND behavioral directives
        so the LLM knows how to adjust its conversational tone.
        """
        snapshot = self.get_snapshot()
        noteworthy = snapshot.get("noteworthy", [])
        if not noteworthy:
            return ""

        lines = ["## Self-Awareness", "You are currently experiencing:"]
        directives = set()

        for item in noteworthy:
            signal = item["signal"]
            severity = item["severity"]
            lines.append(f"- {signal}")

            # Map signals to behavioral directives
            if "recall" in signal or "memory" in signal.lower():
                directives.add(
                    "Be transparent about memory uncertainty. Prefer clarifying questions "
                    "over confident assertions when drawing on memory. Hedge appropriately "
                    '("if I recall correctly..." / "I may be missing some context...").'
                )
            if "provider" in signal or "queue" in signal or "heartbeat" in signal:
                directives.add(
                    "Background processing may be slower than usual. "
                    "If a task requires multiple steps, set expectations about timing."
                )
            if "capability_gap" in signal:
                directives.add(
                    "If the user asks for something you lack a tool for, explain the "
                    "limitation and suggest alternatives rather than simply refusing."
                )
            if "thread" in signal.lower():
                directives.add(
                    "Some background cognitive functions may be impaired. "
                    "Acknowledge if you notice gaps in your awareness."
                )

        if directives:
            lines.append("")
            lines.append("Adapt your behavior:")
            for d in sorted(directives):
                lines.append(f"- {d}")

        return "\n".join(lines)

    # ── Refresh pipeline ────────────────────────────────────────

    def _refresh(self) -> dict:
        """Gather all signal categories and cache the snapshot."""
        snapshot = {
            "epistemic": self._gather_epistemic(),
            "operational": self._gather_operational(),
            "capability": self._gather_capability(),
            "noteworthy": [],
            "refreshed_at": _utc_now().isoformat(),
        }
        snapshot["noteworthy"] = self._assess_noteworthy(snapshot)

        try:
            self._store.setex(CACHE_KEY, CACHE_TTL, json.dumps(snapshot))
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Cache write failed: {e}")

        return snapshot

    # ── Epistemic layer ─────────────────────────────────────────

    def _gather_epistemic(self) -> dict:
        """Memory warmth, recall reliability, topic depth signals."""
        topic = self._get_active_topic()

        wm_depth = self._get_working_memory_depth(topic)

        # Context warmth: driven by working memory depth and FOK
        wm_score = min(1.0, wm_depth / 4.0)
        fok_signal = self._get_fok_signal(topic)
        fok_score = min(1.0, fok_signal / 5.0)
        context_warmth = round(
            (wm_score * 0.6) + (fok_score * 0.4), 3
        )

        return {
            "context_warmth": context_warmth,
            "working_memory_depth": wm_depth,
            "partial_match_signal": fok_signal,
            "recall_failure_rate": self._get_recall_failure_rate(topic),
            "topic_age": self._get_topic_age(),
            "recent_modes": self._get_recent_modes(),
            "focus_active": self._get_focus_active(topic),
            "skill_reliability": self._get_skill_reliability(),
        }

    def _get_active_topic(self) -> str:
        """Get the current active topic from MemoryStore."""
        try:
            topic = self._store.get("recent_topic")
            return topic if topic else "general"
        except Exception:
            return "general"

    def _get_working_memory_depth(self, topic: str) -> int:
        try:
            return self._store.llen(f"working_memory:{topic}")
        except Exception:
            return 0

    def _get_fok_signal(self, topic: str) -> int:
        """Feeling-of-Knowing: partial match count from last recall."""
        try:
            value = self._store.get(f"fok:{topic}")
            return int(value) if value else 0
        except Exception:
            return 0

    def _get_recall_failure_rate(self, topic: str) -> float:
        """Per-topic recall failure rate from procedural memory."""
        try:
            from services.procedural_memory_service import ProceduralMemoryService
            db = self._get_db()
            service = ProceduralMemoryService(db)
            action_stats = service.get_action_stats("recall")
            context_stats = (action_stats or {}).get("context_stats") or {}
            if topic in context_stats:
                topic_stats = context_stats[topic]
                total = topic_stats.get("total", 0)
                failures = topic_stats.get("failures", 0)
                if total > 0:
                    return round(failures / total, 3)
            return 0.0
        except Exception:
            return 0.0

    def _get_topic_age(self) -> str:
        """How long the current topic has been active."""
        try:
            ttl = self._store.ttl("recent_topic")
            if ttl and ttl > 0:
                age_seconds = 1800 - ttl  # recent_topic has 30min TTL
                if age_seconds < 60:
                    return f"{age_seconds}s"
                elif age_seconds < 3600:
                    return f"{age_seconds // 60}min"
                else:
                    return f"{age_seconds // 3600}h {(age_seconds % 3600) // 60}min"
            return "unknown"
        except Exception:
            return "unknown"

    def _get_recent_modes(self) -> list:
        """Last 5 mode selections from routing decisions."""
        try:
            from services.routing_decision_service import RoutingDecisionService
            db = self._get_db()
            service = RoutingDecisionService(db)
            decisions = service.get_recent_decisions(hours=1, limit=5)
            return [d["selected_mode"] for d in decisions]
        except Exception:
            return []

    def _get_focus_active(self, topic: str) -> bool:
        try:
            from services.focus_session_service import FocusSessionService
            return FocusSessionService().get_focus(topic) is not None
        except Exception:
            return False

    def _get_skill_reliability(self) -> dict:
        """Condensed skill reliability: only skills with >= 5 attempts."""
        try:
            from services.procedural_memory_service import ProceduralMemoryService
            db = self._get_db()
            service = ProceduralMemoryService(db)
            all_weights = service.get_all_policy_weights()

            result = {}
            for action_name in all_weights:
                stats = service.get_action_stats(action_name)
                if not stats:
                    continue
                attempts = stats.get("total_attempts", 0)
                if attempts < 5:
                    continue
                successes = stats.get("total_successes", 0)
                reliability = (successes + 1) / (attempts + 2)  # Laplace smoothing
                result[action_name] = {
                    "reliability": round(reliability, 3),
                    "attempts": attempts,
                }
            return result
        except Exception:
            return {}

    # ── Operational layer ───────────────────────────────────────

    def _gather_operational(self) -> dict:
        """Thread health, provider status, queue depth, memory pressure."""
        return {
            "thread_health": self._get_thread_health(),
            "provider_status": self._get_provider_status(),
            "queue_depth": self._get_queue_depth(),
            "memory_pressure": self._get_memory_pressure(),
            "bg_llm_heartbeat_stale": self._is_bg_llm_stale(),
        }

    def _get_thread_health(self) -> dict:
        """Read thread health published by WorkerManager to MemoryStore."""
        try:
            raw = self._store.get("self_model:thread_health")
            if raw:
                data = json.loads(raw)
                return {
                    "alive": len(data.get("alive", [])),
                    "total": data.get("total", 0),
                    "dead_threads": data.get("dead", []),
                }
        except Exception:
            pass
        return {"alive": 0, "total": 0, "dead_threads": []}

    def _get_provider_status(self) -> dict:
        """Check LLM provider assignments for critical cognitive jobs."""
        try:
            from services.provider_db_service import ProviderDbService
            db = self._get_db()
            provider_service = ProviderDbService(db)

            # Count active providers
            providers = provider_service.get_all_providers()
            active_count = sum(1 for p in providers if p.get("is_active"))

            # Check which critical jobs have assigned providers
            assignments = provider_service.get_all_job_assignments()
            assigned_jobs = {a["job_name"] for a in assignments}
            unassigned = [j for j in CRITICAL_JOBS if j not in assigned_jobs]

            return {
                "active_count": active_count,
                "unassigned_jobs": sorted(unassigned),
            }
        except Exception:
            return {"active_count": 0, "unassigned_jobs": []}

    def _get_queue_depth(self) -> dict:
        """Read LLM queue depths from MemoryStore."""
        try:
            from services.background_llm_queue import QUEUE_KEY
            bg_llm = self._store.llen(QUEUE_KEY)
        except Exception:
            bg_llm = 0

        try:
            prompt_queue = self._store.llen("prompt-queue")
        except Exception:
            prompt_queue = 0

        return {"bg_llm": bg_llm, "prompt_queue": prompt_queue}

    def _get_memory_pressure(self) -> dict:
        """Episode/concept/trait counts and average activation from SQLite."""
        try:
            db = self._get_db()
            with db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM episodes")
                episode_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM semantic_concepts")
                concept_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM user_traits")
                trait_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT AVG(activation_score) FROM episodes "
                    "WHERE activation_score > 0"
                )
                row = cursor.fetchone()
                avg_activation = round(row[0], 3) if row[0] else 1.0

                cursor.close()

            return {
                "episode_count": episode_count,
                "concept_count": concept_count,
                "trait_count": trait_count,
                "avg_activation": avg_activation,
            }
        except Exception:
            return {"episode_count": 0, "concept_count": 0, "trait_count": 0, "avg_activation": 1.0}

    def _is_bg_llm_stale(self) -> bool:
        """Check if background LLM worker heartbeat is stale (>30s)."""
        try:
            from services.background_llm_queue import (
                HEARTBEAT_KEY,
                HEARTBEAT_STALE_THRESHOLD,
            )
            last_hb = self._store.get(HEARTBEAT_KEY)
            if last_hb:
                elapsed = time.time() - float(last_hb)
                return elapsed > HEARTBEAT_STALE_THRESHOLD
            # No heartbeat yet — might be early startup
            return False
        except Exception:
            return False

    # ── Capability layer ────────────────────────────────────────

    def _gather_capability(self) -> dict:
        """Tool inventory, capability categories, provider features."""
        tool_names = []
        capability_categories = {}

        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()
            tool_names = registry.get_tool_names()

            # Categorize tools by scanning manifest documentation keywords
            for name in tool_names:
                manifest = registry.get_tool_full_description(name)
                if not manifest:
                    continue
                doc = (manifest.get("documentation", "") or "").lower()
                desc = (manifest.get("description", "") or "").lower()
                text = f"{doc} {desc}"

                for category, keywords in CATEGORY_KEYWORDS.items():
                    if any(kw in text for kw in keywords):
                        if category not in capability_categories:
                            capability_categories[category] = []
                        if name not in capability_categories[category]:
                            capability_categories[category].append(name)
        except Exception:
            pass

        # Innate skills from authoritative registry
        innate_skills = []
        try:
            from services.innate_skills.registry import ALL_SKILL_NAMES
            innate_skills = sorted(ALL_SKILL_NAMES)
        except Exception:
            pass

        # Provider features
        provider_features = self._get_provider_features()

        # Frequent capability gaps (Phase 3)
        frequent_gaps = self.get_frequent_gaps(min_occurrences=2, limit=3)

        return {
            "tool_count": len(tool_names),
            "tool_names": sorted(tool_names),
            "innate_skills": innate_skills,
            "capability_categories": capability_categories,
            "provider_features": provider_features,
            "frequent_gaps": frequent_gaps,
        }

    def _get_provider_features(self) -> dict:
        """Detect provider feature availability."""
        features = {
            "vision": False,
            "local_inference": False,
            "cloud_inference": False,
        }
        try:
            from services.provider_db_service import ProviderDbService
            db = self._get_db()
            providers = ProviderDbService(db).get_all_providers()

            for p in providers:
                if not p.get("is_active"):
                    continue
                platform = (p.get("platform") or "").lower()
                if platform in ("anthropic", "openai", "gemini"):
                    features["cloud_inference"] = True
                    features["vision"] = True  # all cloud providers support vision
                elif platform == "ollama":
                    features["local_inference"] = True
        except Exception:
            pass
        return features

    # ── Noteworthy assessment ───────────────────────────────────

    def _assess_noteworthy(self, snapshot: dict) -> List[dict]:
        """
        Determine what is worth surfacing. Returns empty list when healthy.

        Each item is {"signal": str, "severity": float} where severity
        is used by the mode router for weighted self_constraint scoring.
        """
        notes = []
        op = snapshot.get("operational", {})
        ep = snapshot.get("epistemic", {})

        # Dead worker threads (severity: 0.6)
        dead = op.get("thread_health", {}).get("dead_threads", [])
        if dead:
            names = ", ".join(dead[:3])
            suffix = f" (+{len(dead) - 3} more)" if len(dead) > 3 else ""
            notes.append({
                "signal": f"Worker threads down: {names}{suffix}",
                "severity": SEVERITY_DEAD_THREADS,
            })

        # Missing providers for critical jobs (severity: 0.8)
        unassigned = op.get("provider_status", {}).get("unassigned_jobs", [])
        if unassigned:
            notes.append({
                "signal": f"No LLM provider assigned for: {', '.join(unassigned)}",
                "severity": SEVERITY_MISSING_PROVIDER,
            })

        # Stale background LLM heartbeat (severity: 0.5)
        if op.get("bg_llm_heartbeat_stale"):
            notes.append({
                "signal": "Background LLM worker is stale (no heartbeat >30s)",
                "severity": SEVERITY_STALE_HEARTBEAT,
            })

        # Queue congestion (severity: 0.4)
        bg_depth = op.get("queue_depth", {}).get("bg_llm", 0)
        if bg_depth > 15:
            notes.append({
                "signal": f"LLM queue congested ({bg_depth}/25)",
                "severity": SEVERITY_QUEUE_CONGESTION,
            })

        # High recall failure rate (severity: 0.3)
        rfr = ep.get("recall_failure_rate", 0)
        if rfr > 0.4:
            notes.append({
                "signal": f"Memory recall unreliable for current topic (failure rate: {rfr:.0%})",
                "severity": SEVERITY_HIGH_RECALL_FAILURE,
            })

        # Low average memory activation (severity: 0.2)
        avg_act = op.get("memory_pressure", {}).get("avg_activation", 1.0)
        if avg_act < 0.3:
            notes.append({
                "signal": f"Overall memory activation is low ({avg_act:.2f}) — thin context",
                "severity": SEVERITY_LOW_ACTIVATION,
            })

        # Recurring capability gaps (severity: 0.2)
        cap = snapshot.get("capability", {})
        gaps = cap.get("frequent_gaps", [])
        if gaps:
            top = gaps[0]
            notes.append({
                "signal": f"Recurring capability_gap: {top['request_summary'][:60]} ({top['occurrences']}x)",
                "severity": SEVERITY_CAPABILITY_GAP,
            })

        return notes

    # ── Capability gap learning ─────────────────────────────────

    def log_capability_gap(
        self,
        request_summary: str,
        detection_source: str,
        detected_category: Optional[str] = None,
        confidence: float = 0.5,
    ) -> None:
        """
        Log a capability gap. Deduplicates by exact match on request_summary.

        Args:
            request_summary: What the user asked for (truncated to 200 chars)
            detection_source: 'act_loop', 'triage', 'user_correction'
            detected_category: Optional category from CATEGORY_KEYWORDS
            confidence: 0.0-1.0 confidence that this is a real gap
        """
        try:
            db = self._get_db()
            summary = request_summary[:200].strip()
            if not summary:
                return

            with db.connection() as conn:
                cursor = conn.cursor()

                # Check for existing unresolved gap
                cursor.execute(
                    "SELECT id, occurrences FROM capability_gaps "
                    "WHERE resolved_at IS NULL AND request_summary = ? LIMIT 1",
                    (summary,)
                )
                existing = cursor.fetchone()

                if existing:
                    cursor.execute(
                        "UPDATE capability_gaps SET occurrences = occurrences + 1, "
                        "last_seen_at = datetime('now') WHERE id = ?",
                        (existing[0],)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO capability_gaps "
                        "(request_summary, detected_category, detection_source, confidence) "
                        "VALUES (?, ?, ?, ?)",
                        (summary, detected_category, detection_source, confidence)
                    )
                conn.commit()
                cursor.close()

            logger.debug(f"{LOG_PREFIX} Logged capability gap: {summary[:50]}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to log capability gap: {e}")

    def get_frequent_gaps(self, min_occurrences: int = 3, limit: int = 5) -> list:
        """Get most frequently requested capabilities Chalie lacks."""
        try:
            db = self._get_db()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, request_summary, detected_category, occurrences, first_seen_at "
                    "FROM capability_gaps "
                    "WHERE resolved_at IS NULL AND occurrences >= ? "
                    "ORDER BY occurrences DESC LIMIT ?",
                    (min_occurrences, limit)
                )
                rows = cursor.fetchall()
                cursor.close()
                return [
                    {
                        "id": r[0],
                        "request_summary": r[1],
                        "category": r[2],
                        "occurrences": r[3],
                        "first_seen": r[4],
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def link_gap_to_curiosity(self, gap_id: int, thread_id: str) -> None:
        """Link a capability gap to a seeded curiosity thread."""
        try:
            db = self._get_db()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE capability_gaps SET seeded_curiosity_thread_id = ? WHERE id = ?",
                    (thread_id, gap_id)
                )
                conn.commit()
                cursor.close()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to link gap to curiosity: {e}")

    def resolve_gap(self, gap_id: int, resolved_by: str) -> None:
        """Mark a capability gap as resolved (e.g., when a new tool fills it)."""
        try:
            db = self._get_db()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE capability_gaps SET resolved_at = datetime('now'), "
                    "resolved_by = ? WHERE id = ?",
                    (resolved_by, gap_id)
                )
                conn.commit()
                cursor.close()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to resolve gap: {e}")


# ── Background worker ───────────────────────────────────────────

def self_model_worker(shared_state=None):
    """Background thread: refresh self-model snapshot every 30s."""
    service = SelfModelService()
    logger.info(f"{LOG_PREFIX} Worker started (refresh every {REFRESH_INTERVAL}s)")

    while True:
        try:
            service._refresh()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Refresh failed: {e}")

        time.sleep(REFRESH_INTERVAL)
