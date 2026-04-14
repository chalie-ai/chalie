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
from typing import List

from services.memory_client import MemoryClientService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SELF MODEL]"

CACHE_KEY = "self_model:snapshot"
CACHE_TTL = 45  # seconds
REFRESH_INTERVAL = 30  # background thread cycle

# Critical cognitive jobs — if any lack an assigned provider, that's noteworthy
CRITICAL_JOBS = frozenset({
    'frontal-cortex', 'cognitive-triage',
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
SEVERITY_LOW_ACTIVATION = 0.2


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
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"{LOG_PREFIX} Cache parse failed, refreshing: {e}", exc_info=True)
        return self._refresh()

    def has_noteworthy_state(self) -> bool:
        """Fast gate — True only when something is degraded."""
        snapshot = self.get_snapshot()
        return len(snapshot.get("noteworthy", [])) > 0

    def get_memory_richness(self) -> float:
        """0.0 (no activity) to 1.0 (rich memory), from cached snapshot.

        Logarithmic composite — models cognitive development where early
        experiences carry disproportionate weight. A toddler's first trip
        to the park is intensely rich; by adulthood it barely registers.

        Early on, even 1-2 episodes push richness well above zero because
        they represent 100% of accumulated experience. As memories grow,
        each additional one contributes less (diminishing returns). Hard
        cap at 1.0 ensures stabilisation once the system has a solid
        foundation.

        Workers use this to self-regulate: skip expensive cycles when
        memory is too thin to produce useful results. The logarithmic
        curve ensures fresh installations are never starved — the gate
        only blocks when there is genuinely zero activity.
        """
        import math

        snapshot = self.get_snapshot()

        # Extract counts from operational.memory_pressure
        pressure = snapshot.get("operational", {}).get("memory_pressure", {})
        episode_count = pressure.get("episode_count", 0)
        concept_count = pressure.get("concept_count", 0)
        trait_count = pressure.get("trait_count", 0)

        # Epistemic warmth (current conversation context)
        context_warmth = snapshot.get("epistemic", {}).get("context_warmth", 0.0)

        # Logarithmic scaling — early items contribute disproportionately,
        # diminishing returns as counts grow, hard cap at 1.0.
        # log(1+1)/log(1+50) ≈ 0.18, log(1+5)/log(1+50) ≈ 0.46,
        # log(1+50)/log(1+50) = 1.0
        def _log_saturate(count: int, ceiling: int) -> float:
            if count <= 0:
                return 0.0
            return min(1.0, math.log(1 + count) / math.log(1 + ceiling))

        score = (
            0.35 * _log_saturate(episode_count, 50)
            + 0.25 * _log_saturate(concept_count, 30)
            + 0.20 * _log_saturate(trait_count, 10)
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
        snapshot["noteworthy"].extend(self._check_pipeline_health())

        try:
            self._store.setex(CACHE_KEY, CACHE_TTL, json.dumps(snapshot))
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Cache write failed: {e}")

        return snapshot

    # ── Epistemic layer ─────────────────────────────────────────

    def _gather_epistemic(self) -> dict:
        """Memory warmth, recall reliability, and topic depth signals.

        Uses the ``"general"`` topic key for working-memory and FOK look-ups.
        The ``recent_topic`` MemoryStore key is no longer written by any
        production code path (removed with the topic-classifier), so the old
        dynamic look-up always resolved to ``"general"`` anyway — this makes
        that default explicit and eliminates the dead ``topic_age`` field.

        Returns:
            dict: Mapping of epistemic signal names to their current values.
        """
        channel = "general"

        wm_depth = self._get_working_memory_depth(channel)

        # Context warmth: driven by working memory depth and FOK
        wm_score = min(1.0, wm_depth / 4.0)
        fok_signal = self._get_fok_signal(channel)
        fok_score = min(1.0, fok_signal / 5.0)
        context_warmth = round(
            (wm_score * 0.6) + (fok_score * 0.4), 3
        )

        return {
            "context_warmth": context_warmth,
            "working_memory_depth": wm_depth,
            "partial_match_signal": fok_signal,
            "focus_active": False,
        }

    def _get_working_memory_depth(self, channel: str) -> int:
        try:
            return self._store.llen(f"working_memory:{channel}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get working memory depth: {e}", exc_info=True)
            return 0

    def _get_fok_signal(self, channel: str) -> int:
        """Feeling-of-Knowing: partial match count from last recall."""
        try:
            value = self._store.get(f"fok:{channel}")
            return int(value) if value else 0
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get FOK signal: {e}", exc_info=True)
            return 0

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
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to get thread health: {e}", exc_info=True)
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
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to get provider status: {e}", exc_info=True)
            return {"active_count": 0, "unassigned_jobs": []}

    def _get_queue_depth(self) -> dict:
        """Read LLM queue depths from MemoryStore."""
        try:
            from services.background_llm_queue import QUEUE_KEY
            bg_llm = self._store.llen(QUEUE_KEY)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get bg_llm queue depth: {e}", exc_info=True)
            bg_llm = 0

        return {"bg_llm": bg_llm}

    def _get_memory_pressure(self) -> dict:
        """Episode/concept/trait counts and average activation from SQLite."""
        try:
            db = self._get_db()
            with db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM episodes")
                episode_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM data_graph "
                    "WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"
                )
                concept_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM data_graph "
                    "WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"
                )
                trait_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT AVG(retrieval_weight) FROM episodes "
                    "WHERE retrieval_weight > 0"
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
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to get memory pressure: {e}", exc_info=True)
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
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to check bg LLM heartbeat staleness: {e}", exc_info=True)
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
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load tool inventory: {e}", exc_info=True)

        # Innate skills from authoritative registry
        innate_skills = []
        try:
            from services.innate_skills.registry import ALL_SKILL_NAMES
            innate_skills = sorted(ALL_SKILL_NAMES)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load innate skills: {e}", exc_info=True)

        # Provider features
        provider_features = self._get_provider_features()

        return {
            "tool_count": len(tool_names),
            "tool_names": sorted(tool_names),
            "innate_skills": innate_skills,
            "capability_categories": capability_categories,
            "provider_features": provider_features,
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
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get provider features: {e}", exc_info=True)
        return features

    # ── Noteworthy assessment ───────────────────────────────────

    def _check_pipeline_health(self) -> list:
        """Check cognitive pipeline health -- returns noteworthy items for degraded pipelines."""
        checks = []
        try:
            db = self._get_db()
            with db.connection() as conn:
                # 1. Compaction stale: topic with uncompacted content exceeding
                # the fallback token threshold (chars/4 ≈ tokens, 36K token fallback).
                # Uses the same fallback threshold as compaction_service so this
                # warning fires iff compaction would actually trigger.
                _COMPACTION_WARN_CHARS = 36_000 * 4  # ~144K chars ≈ 36K tokens
                row = conn.execute("""
                    SELECT tt.channel,
                           COUNT(tt.id) as entries,
                           SUM(LENGTH(tt.content)) as total_chars
                    FROM transcript tt
                    LEFT JOIN compactions tc ON tc.channel = tt.channel
                    WHERE tc.channel IS NULL
                    GROUP BY tt.channel
                    HAVING total_chars >= ?
                    ORDER BY total_chars DESC LIMIT 1
                """, (_COMPACTION_WARN_CHARS,)).fetchone()
                if row:
                    estimated_tokens = (row[2] or 0) // 4
                    checks.append({
                        'signal': f"Compaction stale: topic has {row[1]} entries (~{estimated_tokens:,} tokens), no compaction",
                        'severity': 0.7,
                    })

                # 2. Goal evidence empty: 10+ goals, 0 evidence
                try:
                    goal_count = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
                    evidence_count = conn.execute("SELECT COUNT(*) FROM goal_evidence").fetchone()[0]
                    if goal_count >= 10 and evidence_count == 0:
                        checks.append({
                            'signal': f"Goal evidence inactive: {goal_count} goals, 0 evidence",
                            'severity': 0.6,
                        })
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Goal evidence check failed: {e}", exc_info=True)

                # 3. Goal duplication: any goal title appearing 3+ times
                try:
                    dup = conn.execute("""
                        SELECT description, COUNT(*) as c FROM goals
                        GROUP BY description HAVING c >= 3
                        ORDER BY c DESC LIMIT 1
                    """).fetchone()
                    if dup:
                        checks.append({
                            'signal': f"Duplicate goals: '{dup[0][:50]}' x{dup[1]}",
                            'severity': 0.5,
                        })
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Goal duplication check failed: {e}", exc_info=True)

                # 4. Proactive rejection rate: >95% in last 24h
                try:
                    candidates = conn.execute("""
                        SELECT COUNT(*) FROM interaction_log
                        WHERE event_type = 'proactive_candidate'
                          AND created_at > datetime('now', '-24 hours')
                    """).fetchone()[0]
                    rejected = conn.execute("""
                        SELECT COUNT(*) FROM interaction_log
                        WHERE event_type = 'action_gate_rejected'
                          AND created_at > datetime('now', '-24 hours')
                    """).fetchone()[0]
                    if candidates >= 10:
                        rate = (rejected / candidates) * 100 if candidates > 0 else 0
                        if rate > 95:
                            checks.append({
                                'signal': f"Proactive {rate:.0f}% rejection rate ({rejected}/{candidates} in 24h)",
                                'severity': 0.4,
                            })
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Proactive rejection rate check failed: {e}", exc_info=True)

                # Orphaned episodes check removed — topics table dropped in migration 035

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Pipeline health check failed: {e}")
        return checks

    def _assess_noteworthy(self, snapshot: dict) -> List[dict]:
        """
        Determine what is worth surfacing. Returns empty list when healthy.

        Each item is {"signal": str, "severity": float}.
        """
        notes = []
        op = snapshot.get("operational", {})

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

        # Low average memory activation (severity: 0.2)
        avg_act = op.get("memory_pressure", {}).get("avg_activation", 1.0)
        if avg_act < 0.3:
            notes.append({
                "signal": f"Overall memory activation is low ({avg_act:.2f}) — thin context",
                "severity": SEVERITY_LOW_ACTIVATION,
            })

        return notes


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
