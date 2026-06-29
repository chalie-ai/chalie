"""System namespace — /health, /ready, /metrics (probes) and /api/system/status, /api/system/observability/* (admin).

Read-only observability + diagnostics (not CRUD): 17 of 18 routes are GETs
returning a status/diagnostic DTO or an opaque service-owned shape. The one
mutating-ish route (``POST /health`` heartbeat ingest) is an action endpoint
kept at 200 (it returns status/result, not a created resource). Datetimes serialize as ISO-8601 UTC via the foundation serializer — the
local ``_now_iso()`` helper is deleted. Where a protected test pins an exact shape
(records 400, degraded-200, non-ISO ``compacted_at``, raw telemetry/metrics),
current behavior is preserved.
"""

import json
import logging
import os
import sqlite3
from typing import TYPE_CHECKING, cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now
from utils.logger import LOG_FILE_PATH as _LOG_FILE_PATH  # Written exclusively by utils/logger.py in the same process

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.context_usage import ContextUsage
from .dto.health import ClientTelemetry, Health
from .dto.network import NetworkState, NetworkUpdate, NetworkUpdateResult
from .dto.observability_compaction import CompactionRecord, CompactionView
from .dto.observability_errors import ErrorsList, LogError
from .dto.observability_records import RecordsPage
from .dto.observability_tools import ToolUsage, ToolUsageList
from .dto.readiness import Readiness
from .dto.research_run import ResearchRunDetail, ResearchRunsList
from .dto.setting import Setting, SettingWrite
from .dto.system_status import SystemStatus
from .dto.write_queue import WriteQueueStats

if TYPE_CHECKING:
    from pathlib import Path
    from werkzeug.datastructures import FileStorage
    from services.client_context_service import ClientContextService

logger = logging.getLogger(__name__)

health_ns = Namespace("health", description="Liveness, readiness, and metrics probes", path="/")
system_ns = Namespace("system", description="System operations", path="/api/system")

# Signal source for telemetry signals absorbed by WorldState from health pings.
_SIGNAL_SOURCE_HEALTH = "/health"

# Records browser bounds + valid sources (invalid source is a preserved 400, not 422).
_RECORDS_LIMIT = 250
_VALID_SOURCES = {"episodes", "user", "system"}
_RESEARCH_LIST_LIMIT = 100

_LOG_TAIL_BYTES = 256 * 1024   # 256 KB tail read — never loads the full file
_ERROR_LEVELS = frozenset({"ERROR", "CRITICAL"})
_ERROR_CAP = 200

# Uploaded TLS material is owner-read/write only — never group/world readable.
_SECURE_FILE_MODE = 0o600

# Each namespace registers only the models its own routes reference — the probes
# expose just their status shapes, the admin surface owns the diagnostics models.
# No cross-pollination of the OpenAPI definitions.
_HEALTH_DTOS = (Health, ClientTelemetry, Readiness, Error)
_SYSTEM_DTOS = (
    SystemStatus,
    RecordsPage,
    ToolUsage,
    ToolUsageList,
    CompactionRecord,
    CompactionView,
    ResearchRunsList,
    ResearchRunDetail,
    WriteQueueStats,
    LogError,
    ErrorsList,
    ContextUsage,
    Setting,
    SettingWrite,
    NetworkState,
    NetworkUpdate,
    NetworkUpdateResult,
    Error,
)
register_dto(health_ns, *_HEALTH_DTOS)
register_dto(system_ns, *_SYSTEM_DTOS)

_H = health_ns.models
_S = system_ns.models



def _health() -> Health:
    """Build the standard health response — status + running app version."""
    from consumer import APP_VERSION
    return Health(status="ok", version=APP_VERSION)


def _mirror_telemetry_to_world_state(svc: "ClientContextService", data: dict[str, object]) -> None:
    """Mirror persisted client telemetry into WorldState as Signals."""
    from services.world_state import world_state, Signal
    world_state.set("telemetry", svc.get() or data)
    world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind="heartbeat", payload=data))
    device_class = data.get("device_class") or cast(dict[str, object], data.get("device") or {}).get("class")
    if device_class:
        world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind="device", payload={"device_class": device_class}))
    local_time = data.get("local_time")
    if local_time:
        world_state.absorb(Signal(source=_SIGNAL_SOURCE_HEALTH, kind="local_time", payload={"local_time": local_time}))


def _persist_heartbeat(data: dict[str, object]) -> None:
    """Persist client context + mirror to WorldState. Each step is independently logged on failure."""
    from services.client_context_service import ClientContextService
    svc = ClientContextService()
    svc.save(data)
    try:
        _mirror_telemetry_to_world_state(svc, data)
    except Exception as ws_err:
        logger.warning(f"[HEALTH] Failed to mirror telemetry to WorldState: {ws_err}")


# ---------------------------------------------------------------------------
# Liveness + readiness probes (public — no auth)
# ---------------------------------------------------------------------------

@health_ns.route("/health")
class HealthResource(Resource):
    @health_ns.response(200, "OK", model=_H["Health"])
    def get(self) -> ResponseReturnValue:
        """Health check endpoint (no auth required)."""
        return _health().model_dump(mode="json"), 200

    @health_ns.expect(_H["ClientTelemetry"])
    @health_ns.response(200, "OK", model=_H["Health"])
    @responds(Health, code=200)
    @expects(ClientTelemetry)
    def post(self, dto: ClientTelemetry) -> Health:
        """Health check endpoint (no auth required). POST saves client context (best-effort)."""
        data = dto.model_dump(mode="json")
        try:
            if data:
                _persist_heartbeat(data)
        except Exception as e:
            logger.warning(f"[HEALTH] Failed to save client context: {e}")
        return _health()


@health_ns.route("/ready")
class ReadinessResource(Resource):
    @health_ns.response(200, "Ready", model=_H["Readiness"])
    @health_ns.response(503, "Not ready", model=_H["Readiness"])
    @responds(Readiness, code=200)
    def get(self) -> ResponseReturnValue:
        """Readiness probe — 200 only when SQLite, MemoryStore, embeddings, and ONNX are ready.

        Delegates to the read-only :func:`services.preflight_service.run_preflight`.
        The onnxruntime CPU self-heal and its actionable ERROR hint run once at boot
        (``RuntimeDepsService.ensure_onnxruntime``); this probe only reports state, so
        a broken runtime surfaces as ``embeddings: {status: error, message: <hint>}``
        rather than the eternal ``loading`` it reported before.
        """
        from services.preflight_service import run_preflight
        components = run_preflight()
        ready = all(c.get("status") == "ok" for c in components.values())
        dto = Readiness(ready=ready, **components)
        # Variable status (200/503): return the pre-serialized body + code so the
        # foundation serializer emits the exact component shapes the tests pin.
        return dto.model_dump(mode="json"), (200 if ready else 503)


# ---------------------------------------------------------------------------
# Metrics + system status
# ---------------------------------------------------------------------------

@health_ns.route("/metrics")
class MetricsResource(Resource):
    @require_session
    @health_ns.response(200, "Metrics dashboard (opaque service-owned body)")
    @health_ns.response(500, "Failed to retrieve metrics", model=_H["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """Metrics dashboard endpoint (raw passthrough — service-owned shape)."""
        try:
            from services.metrics_service import MetricsService
            return MetricsService().get_dashboard_data()
        except Exception as e:
            logger.error(f"[REST API] Metrics error: {e}")
            return error("Failed to retrieve metrics", 500)


@system_ns.route("/status")
class SystemStatusResource(Resource):
    @require_session
    @system_ns.response(200, "System health (ok or degraded — both 200)", model=_S["SystemStatus"])
    @system_ns.response(500, "Status check failed", model=_S["Error"])
    @responds(SystemStatus, code=200)
    def get(self) -> SystemStatus | ResponseReturnValue:
        """Comprehensive system health and diagnostics."""
        try:
            from services.memory_client import MemoryClientService
            from services.database_service import get_shared_db_service

            store = MemoryClientService.create_connection()
            status = "ok"
            memory: dict[str, object] = {}
            storage: dict[str, object] = {}
            memory_store_error: str | None = None
            database_error: str | None = None

            # MemoryStore health
            try:
                store.ping()
                memory["working_memory_keys"] = len(store.keys("working_memory:*"))
                memory["gist_keys"] = len(store.keys("gist_index:*"))
                memory["fact_keys"] = len(store.keys("fact_index:*"))
            except Exception as e:
                status = "degraded"
                memory_store_error = str(e)

            # SQLite counts
            try:
                db = get_shared_db_service()
                with db.connection() as conn:
                    cursor = conn.cursor()
                    for table_label, query in [
                        ("episodes", "SELECT COUNT(*) FROM episodes"),
                        ("concepts", "SELECT COUNT(*) FROM data_graph WHERE kind = 'user_specific' AND deleted_at IS NULL AND active=1"),
                    ]:
                        try:
                            cursor.execute(query)
                            row = cursor.fetchone()
                            storage[table_label] = row[0] if row else 0
                        except sqlite3.OperationalError as e:
                            logger.warning(f"[SYSTEM] Count query failed for '{table_label}': {e}")
                            storage[table_label] = -1
                            status = "degraded"
                        except Exception as e:
                            logger.warning(f"[SYSTEM] Count query failed for '{table_label}': {e}")
                            storage[table_label] = -1
            except Exception as e:
                status = "degraded"
                database_error = str(e)

            return SystemStatus(
                status=status,
                memory=memory,
                storage=storage,
                memory_store_error=memory_store_error,
                database_error=database_error,
            )
        except Exception as e:
            logger.error(f"[REST API] System status error: {e}")
            return error("System status check failed", 500)


# ─────────────────────────────────────────────
# Observability — cognitive legibility endpoints
# ─────────────────────────────────────────────

@system_ns.route("/observability/records")
class ObservabilityRecordsResource(Resource):
    @require_session
    @system_ns.param("source", "episodes | user | system", _in="query", required=True)
    @system_ns.param("offset", "Rows to skip (>=0)", _in="query")
    @system_ns.param("q", "Substring filter", _in="query")
    @system_ns.response(200, "Records page", model=_S["RecordsPage"])
    @system_ns.response(400, "invalid source / invalid offset", model=_S["Error"])
    @system_ns.response(500, "Failed to retrieve records", model=_S["Error"])
    @responds(RecordsPage, code=200)
    def get(self) -> RecordsPage | ResponseReturnValue:
        """Paginated record browser for episodes, user, and system memory sources."""
        try:
            source = request.args.get("source", "")
            if source not in _VALID_SOURCES:
                return error("invalid source", 400)

            raw_offset = request.args.get("offset", "0")
            try:
                offset = int(raw_offset)
            except (ValueError, TypeError):
                return error("invalid offset", 400)
            if offset < 0:
                return error("invalid offset", 400)

            q = (request.args.get("q", "") or "")[:200]

            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            if source == "episodes":
                rows = db.fetch_all(
                    "SELECT created_at AS created, "
                    "COALESCE(last_relevant_at, created_at) AS last_accessed, "
                    "gist AS value, location_name "
                    "FROM episodes "
                    "WHERE deleted_at IS NULL "
                    "AND (? = '' OR gist LIKE ?) "
                    "ORDER BY COALESCE(last_relevant_at, created_at) DESC, created_at DESC "
                    "LIMIT ? OFFSET ?",
                    (q, f"%{q}%", _RECORDS_LIMIT, offset),
                )
            else:
                kind = "user_specific" if source == "user" else "system"
                rows = db.fetch_all(
                    "SELECT first_seen_at AS created, last_accessed_at AS last_accessed, "
                    "key, value "
                    "FROM data_graph "
                    "WHERE kind = ? AND active = 1 AND deleted_at IS NULL "
                    "AND (? = '' OR key LIKE ? OR value LIKE ?) "
                    "ORDER BY last_accessed_at IS NULL, last_accessed_at DESC, first_seen_at DESC "
                    "LIMIT ? OFFSET ?",
                    (kind, q, f"%{q}%", f"%{q}%", _RECORDS_LIMIT, offset),
                )

            rows = rows or []
            serialised: list[dict[str, object]] = []
            for r in rows:
                row: dict[str, object] = {
                    "created": r["created"],
                    "last_accessed": r["last_accessed"],
                    "value": r["value"],
                }
                if source == "episodes":
                    row["location"] = r.get("location_name") or ""
                else:
                    row["key"] = r["key"]
                serialised.append(row)

            return RecordsPage(
                generated_at=utc_now(),
                source=source,
                rows=serialised,
                offset=offset,
                limit=_RECORDS_LIMIT,
                returned=len(rows),
                has_more=len(rows) == _RECORDS_LIMIT,
            )
        except Exception as e:
            logger.error(f"[REST API] observability/records error: {e}")
            return error("Failed to retrieve records", 500)


@system_ns.route("/observability/tools")
class ObservabilityToolsResource(Resource):
    @require_session
    @system_ns.response(200, "Tool usage list", model=_S["ToolUsageList"])
    @system_ns.response(500, "Failed to retrieve tool data", model=_S["Error"])
    @responds(ToolUsageList, code=200)
    def get(self) -> ToolUsageList | ResponseReturnValue:
        """Tool usage counts from tool_calls — count + last_used per tool."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            rows = db.fetch_all(
                "SELECT tool_name, COUNT(*) AS count, MAX(created_at) AS last_used_at "
                "FROM tool_calls "
                "WHERE tool_name NOT IN ('compaction', 'tool_compaction', 'trail_compaction', "
                "'chat_history_compactor', 'thinking') "
                "GROUP BY tool_name "
                "ORDER BY last_used_at DESC"
            )
            return ToolUsageList(
                generated_at=utc_now(),
                tools=[
                    ToolUsage(
                        tool_name=cast(str, r["tool_name"]),
                        count=cast(int, r["count"]),
                        last_used_at=cast(str, r["last_used_at"]),
                    )
                    for r in (rows or [])
                ],
            )
        except Exception as e:
            logger.error(f"[REST API] observability/tools error: {e}")
            return error("Failed to retrieve tool data", 500)


@system_ns.route("/observability/token-usage")
class ObservabilityTokenUsageResource(Resource):
    @require_session
    @system_ns.param("window", "hour | day | week | month | lifetime (default day)", _in="query")
    @system_ns.param("usage_class", "chat | subagent | subconscious (optional filter)", _in="query")
    @system_ns.response(200, "Token usage (opaque service-owned body)")
    @system_ns.response(422, "Invalid window", model=_S["Error"])
    @system_ns.response(500, "Failed to retrieve token usage data", model=_S["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """Token usage aggregated by time window, model, provider, and usage class (raw passthrough)."""
        from services.llm_call_log_service import get_token_usage, VALID_WINDOWS
        window = request.args.get("window", "day")
        if window not in VALID_WINDOWS:
            return error(f"Invalid window '{window}'. Use: {', '.join(sorted(VALID_WINDOWS))}", 422)
        usage_class = request.args.get("usage_class") or None
        try:
            return get_token_usage(window=window, usage_class=usage_class)
        except Exception as e:
            logger.error(f"[REST API] observability/token-usage error: {e}")
            return error("Failed to retrieve token usage data", 500)


@system_ns.route("/observability/world-state")
class ObservabilityWorldStateResource(Resource):
    @require_session
    @system_ns.response(200, "World state (opaque service-owned body)")
    @system_ns.response(500, "Failed to retrieve world state", model=_S["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """World state as seen by the ACT loop — rendered block + raw inputs (raw passthrough)."""
        try:
            from services.world_state import world_state, _fetch_schedule_rows
            from services.heartbeat_service import heartbeat_service

            return {
                "rendered": world_state.render(),
                "inputs": {
                    "telemetry": heartbeat_service.read(),
                    "signals": world_state.get("signals"),
                    "schedule": _fetch_schedule_rows(),
                },
            }
        except Exception as e:
            logger.error(f"[REST API] observability/world-state error: {e}")
            return error("Failed to retrieve world state", 500)


@system_ns.route("/observability/compaction")
class ObservabilityCompactionResource(Resource):
    @require_session
    @system_ns.response(200, "Compaction view", model=_S["CompactionView"])
    @system_ns.response(500, "Failed to retrieve compaction summary", model=_S["Error"])
    @responds(CompactionView, code=200)
    def get(self) -> CompactionView | ResponseReturnValue:
        """Continuity-compaction synthesis for the chat ('user') channel (read-only)."""
        try:
            from services import compaction_persistence, locale_service
            # MAIN-spine checkpoint (for_turn_id NULL): the watermark is a turn_id.
            # The DTO field keeps its name so the existing Brain view needs no change.
            record = compaction_persistence.get_compaction("user", None)
            if not record:
                return CompactionView(compaction=None)
            return CompactionView(
                compaction=CompactionRecord(
                    summary=cast(str, record["compacted_text"]),
                    compacted_up_to_id=cast(int, record["compacted_up_to"]),
                    compacted_at=locale_service.format_date(cast(str, record["created_at"]), for_ui=True) or "",
                ),
            )
        except Exception:
            logger.exception("[REST API] observability/compaction error")
            return error("Failed to retrieve compaction summary", 500)


@system_ns.route("/observability/research")
class ObservabilityResearchResource(Resource):
    @require_session
    @system_ns.response(200, "Research runs list", model=_S["ResearchRunsList"])
    @system_ns.response(500, "Failed to retrieve research runs", model=_S["Error"])
    @responds(ResearchRunsList, code=200)
    def get(self) -> ResearchRunsList | ResponseReturnValue:
        """Newest-first list of proactive-research (Auto Research) runs (read-only)."""
        try:
            from services import discovery_runs
            return ResearchRunsList(
                generated_at=utc_now(),
                runs=discovery_runs.list_runs(_RESEARCH_LIST_LIMIT),
            )
        except Exception:
            logger.exception("[REST API] observability/research error")
            return error("Failed to retrieve research runs", 500)


@system_ns.route("/observability/research/<int:run_id>")
@system_ns.param("run_id", "Research run id", _in="path")
class ObservabilityResearchDetailResource(Resource):
    @require_session
    @system_ns.response(200, "Research run detail", model=_S["ResearchRunDetail"])
    @system_ns.response(404, "Not found", model=_S["Error"])
    @system_ns.response(500, "Failed to retrieve research run", model=_S["Error"])
    @responds(ResearchRunDetail, code=200)
    def get(self, run_id: int) -> ResearchRunDetail | ResponseReturnValue:
        """One Auto Research run: the grounding it ran against plus its full output."""
        try:
            from services import discovery_runs
            run = discovery_runs.get_run_detail(run_id)
            if run is None:
                return error("not found", 404)
            return ResearchRunDetail(generated_at=utc_now(), run=run)
        except Exception:
            logger.exception("[REST API] observability/research detail error")
            return error("Failed to retrieve research run", 500)


@system_ns.route("/observability/write-queue")
class ObservabilityWriteQueueResource(Resource):
    @require_session
    @system_ns.response(200, "Write queue stats", model=_S["WriteQueueStats"])
    @system_ns.response(500, "Failed to retrieve write queue stats", model=_S["Error"])
    @responds(WriteQueueStats, code=200)
    def get(self) -> WriteQueueStats | ResponseReturnValue:
        """Write queue runtime statistics (backlog, completed writes, error count)."""
        try:
            from services.write_queue_service import get_write_queue
            stats = get_write_queue().get_stats()
            return WriteQueueStats(
                queue_size=stats["queue_size"],
                processed=stats["processed"],
                errors=stats["errors"],
            )
        except Exception as e:
            logger.error(f"[REST API] observability/write-queue error: {e}")
            return error("Failed to retrieve write queue stats", 500)


@system_ns.route("/observability/telemetry")
class ObservabilityTelemetryResource(Resource):
    @require_session
    @system_ns.response(200, "Telemetry summary (opaque, dynamic event-type keys)")
    @system_ns.response(500, "Failed to retrieve telemetry summary", model=_S["Error"])
    @responds(code=200)
    def get(self) -> ResponseReturnValue:
        """Telemetry event summary across all tracked event types (raw passthrough)."""
        try:
            from services.telemetry_service import get_telemetry_collector
            summary = get_telemetry_collector().get_summary()
            return {"generated_at": utc_now().isoformat(), **summary}
        except Exception as e:
            logger.error(f"[REST API] observability/telemetry error: {e}")
            return error("Failed to retrieve telemetry summary", 500)


def _tail_error_lines() -> list[dict[str, object]]:
    """Read the last ~256 KB of the JSON log file (_LOG_FILE_PATH) and return ERROR/CRITICAL entries newest-first.

    Written by utils.logger.Logger.start() — see backend/utils/logger.py FileHandler.
    Skips malformed JSON lines silently. Capped at _ERROR_CAP entries.
    """
    try:
        size = os.path.getsize(_LOG_FILE_PATH)
    except OSError:
        return []

    errors: list[dict[str, object]] = []
    try:
        fh_cm = open(_LOG_FILE_PATH, "r", errors="replace")
    except OSError:
        return []

    with fh_cm as fh:
        if size > _LOG_TAIL_BYTES:
            # Absolute seek — text-mode files only allow 0-byte end-relative seeks.
            fh.seek(size - _LOG_TAIL_BYTES)
            fh.readline()   # discard partial first line
        for raw in fh:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if entry.get("level") in _ERROR_LEVELS:
                errors.append({"timestamp": entry.get("timestamp", ""), "message": entry.get("message", "")})

    errors.reverse()
    return errors[:_ERROR_CAP]


@system_ns.route("/observability/errors")
class ObservabilityErrorsResource(Resource):
    @require_session
    @system_ns.response(200, "Recent error log lines", model=_S["ErrorsList"])
    @system_ns.response(500, "Failed to retrieve error log", model=_S["Error"])
    @responds(ErrorsList, code=200)
    def get(self) -> ErrorsList | ResponseReturnValue:
        """Recent ERROR and CRITICAL log lines from /tmp/chalie.log, newest first."""
        try:
            return ErrorsList(
                generated_at=utc_now(),
                errors=[
                    LogError(
                        timestamp=cast(str, e["timestamp"]),
                        message=cast(str, e["message"]),
                    )
                    for e in _tail_error_lines()
                ],
            )
        except Exception as e:
            logger.error(f"[REST API] observability/errors error: {e}")
            return error("Failed to retrieve error log", 500)


# ──────────────────────────────────────────────
# Settings endpoints
# ──────────────────────────────────────────────

@system_ns.route("/context-usage")
class ContextUsageResource(Resource):
    @require_session
    @system_ns.response(200, "Context usage", model=_S["ContextUsage"])
    @system_ns.response(500, "Failed to retrieve context usage", model=_S["Error"])
    @responds(ContextUsage, code=200)
    def get(self) -> ContextUsage | ResponseReturnValue:
        """Last user-turn request size + context window for the composer indicator."""
        try:
            from services.llm_call_log_service import get_last_chat_request_tokens
            from services.provider_cache_service import ProviderCacheService
            last = get_last_chat_request_tokens()
            selected = ProviderCacheService.get_selected_provider() or {}
            return ContextUsage(
                last_request_tokens=last,
                context_window=cast("int | None", selected.get("max_tokens")),
            )
        except Exception as e:
            logger.error(f"[REST API] context-usage error: {e}")
            return error("Failed to retrieve context usage", 500)


@system_ns.route("/settings/<key>")
@system_ns.param("key", "Setting key", _in="path")
class SettingsResource(Resource):
    @require_session
    @system_ns.response(200, "Setting value", model=_S["Setting"])
    @system_ns.response(500, "Failed to get setting", model=_S["Error"])
    @responds(Setting, code=200)
    def get(self, key: str) -> Setting | ResponseReturnValue:
        """Read an opaque setting by key."""
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        try:
            svc = SettingsService(get_shared_db_service())
            return Setting(key=key, value=svc.get(key))
        except Exception as e:
            logger.error(f"[REST API] get setting error: {e}")
            return error("Failed to get setting", 500)

    @require_session
    @system_ns.expect(_S["SettingWrite"])
    @system_ns.response(200, "Setting written (value or null on delete)", model=_S["Setting"])
    @system_ns.response(500, "Failed to save setting", model=_S["Error"])
    @responds(Setting, code=200)
    @expects(SettingWrite)
    def put(self, key: str, dto: SettingWrite) -> Setting | ResponseReturnValue:
        """Write (or, on empty/falsey value, delete) an opaque setting."""
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        try:
            svc = SettingsService(get_shared_db_service())
            value = dto.value
            if not value:
                svc.delete(key)
            else:
                svc.set(key, str(value))
            return Setting(key=key, value=value or None)
        except Exception as e:
            logger.error(f"[REST API] set setting error: {e}")
            return error("Failed to save setting", 500)


# ──────────────────────────────────────────────
# Network / TLS endpoints
# ──────────────────────────────────────────────

def _save_secure(upload: "FileStorage", path: "Path") -> None:
    """Persist an uploaded file owner-only (0600) from creation — never a world-readable window."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECURE_FILE_MODE)
    with os.fdopen(fd, "wb") as sink:
        upload.save(sink)


def _store_tls_material(cert: "FileStorage", key: "FileStorage") -> bool:
    """Validate an uploaded cert+key pair and, only if they form a TLS context, install them.

    Writes to temp siblings (0600) first and atomically replaces the live files
    only on success, so a bad upload never reaches the serving path, crash-loops
    boot, or destroys a working certificate already on file.
    """
    import ssl
    secure_dir = FileMapperService.get_secure_dir()
    secure_dir.mkdir(parents=True, exist_ok=True)
    cert_path = FileMapperService.get_ssl_cert_path()
    key_path = FileMapperService.get_ssl_key_path()
    tmp_cert = cert_path.with_name(cert_path.name + ".tmp")
    tmp_key = key_path.with_name(key_path.name + ".tmp")
    _save_secure(cert, tmp_cert)
    _save_secure(key, tmp_key)
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(str(tmp_cert), str(tmp_key))
    except (ssl.SSLError, OSError) as exc:
        logger.warning("[SSL] uploaded certificate/key rejected: %s", exc)
        tmp_cert.unlink(missing_ok=True)
        tmp_key.unlink(missing_ok=True)
        return False
    os.replace(tmp_cert, cert_path)
    os.replace(tmp_key, key_path)
    return True


@system_ns.route("/network")
class NetworkResource(Resource):
    @require_session
    @system_ns.response(200, "Current deployment domain + TLS state", model=_S["NetworkState"])
    @system_ns.response(500, "Failed to read network settings", model=_S["Error"])
    @responds(NetworkState, code=200)
    def get(self) -> NetworkState | ResponseReturnValue:
        """Current deployment domain + TLS state for the System page."""
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        try:
            svc = SettingsService(get_shared_db_service())
            return NetworkState(
                deployment_domain=svc.get(SettingsService.DEPLOYMENT_DOMAIN) or "",
                ssl_enabled=svc.get_bool(SettingsService.SSL_ENABLED),
                ssl_cert_present=FileMapperService.get_ssl_cert_path().is_file(),
            )
        except Exception as e:
            logger.error(f"[REST API] network state error: {e}")
            return error("Failed to read network settings", 500)

    @require_session
    @system_ns.expect(_S["NetworkUpdate"])
    @system_ns.response(200, "Saved — restarting to apply", model=_S["NetworkUpdateResult"])
    @system_ns.response(422, "Invalid certificate/key, or SSL enabled without one", model=_S["Error"])
    @system_ns.response(500, "Failed to save network settings", model=_S["Error"])
    @responds(NetworkUpdateResult, code=200)
    @expects(NetworkUpdate, source="form")
    def put(self, dto: NetworkUpdate) -> NetworkUpdateResult | ResponseReturnValue:
        """Persist deployment domain + TLS config, then restart to apply.

        A supplied cert/key pair is written 0600 to the secure dir and validated
        by loading it; ``ssl_enabled`` is honored only when a usable cert is on
        file, so a bad upload can never crash-loop the server.
        """
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        from services.restart_service import request_restart
        try:
            if (dto.ssl_cert is None) != (dto.ssl_key is None):
                return error("Certificate and key must be supplied together", 422)
            if dto.ssl_cert is not None and dto.ssl_key is not None \
                    and not _store_tls_material(dto.ssl_cert, dto.ssl_key):
                return error("Certificate and key are not a valid pair", 422)

            cert_ready = (FileMapperService.get_ssl_cert_path().is_file()
                          and FileMapperService.get_ssl_key_path().is_file())
            if dto.ssl_enabled and not cert_ready:
                return error("Cannot enable SSL without a certificate and key", 422)

            svc = SettingsService(get_shared_db_service())
            svc.set(SettingsService.DEPLOYMENT_DOMAIN, dto.deployment_domain)
            svc.set_bool(SettingsService.SSL_ENABLED, dto.ssl_enabled)

            request_restart()
            return NetworkUpdateResult(ssl_enabled=dto.ssl_enabled, restarting=True)
        except Exception as e:
            logger.error(f"[REST API] network update error: {e}")
            return error("Failed to save network settings", 500)