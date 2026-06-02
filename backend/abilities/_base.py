"""Ability base class — the single-dispatch chokepoint for all tool calls.

Every tool call from the ACT loop enters through ``Ability.dispatch()``.
This is the ONLY path from MessageProcessor._loop() to ability execution.

Spec: ACT Loop Orchestrator Refactor §5, §7b.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from abc import ABC, abstractmethod
from typing import ClassVar
from uuid import uuid4


# These module-level names are populated at the bottom of this file (after the
# Ability class is defined) so that unittest.mock.patch("abilities._base.X")
# can intercept them from tests.  The imports are deferred to the END of the
# module to avoid circular-import issues (_registry imports Ability from here).
AbilityRegistry: object = None  # type: ignore[assignment]
PolicyService: object = None  # type: ignore[assignment]
WebSocketBroker: object = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ── Module-level active-delegate registry ─────────────────────────────────────
# Maps delegate_id → cancel threading.Event.
# Populated by dispatch() for ASYNC_CAPABLE tools on async-capable channels.

_active_delegates: dict[str, threading.Event] = {}


def _supports_async_delivery(channel: str) -> bool:
    """Return True when the channel supports async result delivery.

    Currently only the 'user' channel supports async delivery via
    dispatch_message(hidden_input=True).  Spec §5 / §5b.
    """
    return channel == "user"


def _load_tool_telemetry() -> "dict | None":
    """Pull a flattened telemetry dict for tool dispatch.

    Returns None when no client context is stored yet (fresh boot, no
    heartbeat) so abilities can fall back gracefully.

    Formerly in act_dispatcher_service; moved here in T3.
    """
    try:
        from services.locale_service import (  # noqa: PLC0415
            get_timezone_name, get_locale, get_language,
            get_currency, get_location, format_date,
        )
        from services.time_utils import utc_now  # noqa: PLC0415

        location = get_location()
        loc_name = location.get("name") or ""
        city, country = "", ""
        if "," in loc_name:
            city, country = [p.strip() for p in loc_name.split(",", 1)]

        return {
            "lat": location.get("lat"),
            "lon": location.get("lon"),
            "location_name": loc_name,
            "city": city,
            "country": country,
            "time": format_date(utc_now(), "%H:%M:%S", for_ui=True) or "",
            "timezone": get_timezone_name(),
            "locale": get_locale(),
            "language": get_language(),
            "currency": get_currency(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Ability] telemetry load failed: %s", exc)
        return None


def _run_with_timeout(ability: "Ability", channel: str, params: dict, timeout: float) -> dict:
    """Execute ability.run() in a daemon thread bounded by *timeout* seconds.

    Copies the calling thread's contextvars so abilities can reach
    current_processor() inside the thread.

    Returns a result dict with 'status' and 'result' keys.

    Spec §5 / I13.
    """
    result_box: dict = {"result": None, "exc": None}
    ctx = contextvars.copy_context()

    def _target() -> None:
        try:
            telemetry = _load_tool_telemetry()
        except Exception:  # noqa: BLE001
            telemetry = None
        try:
            result_box["result"] = ability.run(channel, params, telemetry)
        except Exception as exc:  # noqa: BLE001
            result_box["exc"] = exc

    thread = threading.Thread(target=ctx.run, args=(_target,), daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return {"status": "timeout", "result": f"Action exceeded {timeout}s timeout"}

    if result_box["exc"] is not None:
        exc = result_box["exc"]
        # VaultLockedError gets a friendlier message.
        try:
            from services.vault_service import VaultLockedError  # noqa: PLC0415
            if isinstance(exc, VaultLockedError):
                return {
                    "status": "error",
                    "result": (
                        "This function is currently unavailable. "
                        "The vault is locked. "
                        "Notify the user that you could not complete this action "
                        "because they were logged out"
                    ),
                }
        except ImportError:
            pass
        return {"status": "error", "result": f"Error: {exc}"}

    raw = result_box["result"]
    return _normalise_run_result(raw)


def _normalise_run_result(raw: object) -> dict:
    """Coerce the raw return value of ability.run() into a {'status','result'} dict."""
    if isinstance(raw, dict):
        if "status" in raw:
            # Already in canonical form.
            result_val = raw.get("result", "")
            return {"status": raw["status"], "result": str(result_val) if result_val is not None else ""}
        # Legacy dict without status — treat as success.
        return {"status": "success", "result": raw}
    if isinstance(raw, str):
        return {"status": "success", "result": raw}
    return {"status": "success", "result": str(raw) if raw is not None else ""}


def _dispatch_mcp(mp: object, tool_name: str, params: dict) -> dict:
    """Route an _mcp_<server>_<tool> call through McpClientService.

    Policy enforcement is applied before the MCP call using the dynamically-
    seeded policy_rules rows from McpClientService._seed_policy_rows.

    Spec §5 / I9.
    """
    try:
        from services.mcp_client_service import McpClientService  # noqa: PLC0415
        raw = McpClientService().dispatch_mcp_tool(tool_name, params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Ability.dispatch] MCP tool %r failed: %s", tool_name, exc)
        return {"status": "error", "result": f"MCP tool error: {exc}"}

    # McpClientService.dispatch_mcp_tool returns {'text': ...}
    if isinstance(raw, dict) and "text" in raw:
        return {"status": "success", "result": raw["text"]}
    return {"status": "success", "result": str(raw)}


def _process_discovered_tools(mp: object, result: dict) -> None:
    """Add find_tools-discovered tool schemas to mp.discovered_tools.

    Mirrors the side-effect logic previously in MessageProcessor.handle_tool().
    Spec §5 / I12.
    """
    discovered = result.get("_discovered_tools") or []
    if not discovered:
        return

    try:
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        from services.mcp_client_service import McpClientService  # noqa: PLC0415

        existing_names = {t.get("name") for t in getattr(mp, "discovered_tools", [])}

        for name in discovered:
            if name in existing_names:
                continue
            if name.startswith("_mcp_"):
                schema = McpClientService().get_tool_schema(name)
                if schema is None:
                    continue
            else:
                try:
                    a = AbilityRegistry.get(name)
                    schema = {
                        "name": a.NAME,
                        "description": a.SUMMARY,
                        "input_schema": a.INPUT_SCHEMA,
                    }
                except (KeyError, AttributeError):
                    continue

            if hasattr(mp, "discovered_tools"):
                mp.discovered_tools.append(schema)
            existing_names.add(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Ability._process_discovered_tools] failed: %s", exc)


def _emit(config: object, event: dict) -> None:
    """Broadcast *event* when config.broadcast_to is non-None; no-op otherwise.

    Spec §5 / N1 / N2 / AC-28.
    """
    if config is None or getattr(config, "broadcast_to", None) is None:
        return
    try:
        WebSocketBroker().broadcast(event)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[Ability._emit] broadcast failed: %s", exc)


def _run_async_delegate(
    ability: "Ability",
    channel: str,
    params: dict,
    delegate_id: str,
    cancel_event: threading.Event,
) -> None:
    """Daemon-thread body for ASYNC_CAPABLE tool execution.

    Runs the tool synchronously, then delivers the result via
    dispatch_message(hidden_input=True) so the parent ACT loop gets a fresh
    turn with the result.

    Spec §5 / §5b / J5.
    """
    try:
        try:
            result = _run_with_timeout(ability, channel, params, ability.TIMEOUT)
            result_text = str(result.get("result", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Ability._run_async_delegate] execution failed for %s: %s",
                delegate_id,
                exc,
            )
            result_text = None

        if result_text is not None and not cancel_event.is_set():
            try:
                from api.chat import dispatch_message  # noqa: PLC0415
                dispatch_message(result_text, channel=channel, hidden_input=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Ability._run_async_delegate] delivery failed for %s: %s",
                    delegate_id,
                    exc,
                )
    finally:
        _active_delegates.pop(delegate_id, None)


class Ability(ABC):
    """Base class for every dispatchable tool.

    ``dispatch()`` is the single chokepoint for all tool calls from the ACT
    loop.  It handles: sanitize → policy → timeout thread → execute → record.

    Tool *scope* (always-available vs discoverable) lives on ProcessorConfig
    (always_available / discoverable / blocked fields).  Abilities only describe
    what the tool is (NAME / SUMMARY / EXAMPLES / INPUT_SCHEMA / TIMEOUT /
    ASYNC_CAPABLE).

    Spec: §5 / AC-4.
    """

    NAME: ClassVar[str]
    SUMMARY: ClassVar[str]
    EXAMPLES: ClassVar[list[str]]
    INPUT_SCHEMA: ClassVar[dict]
    TIMEOUT: ClassVar[int] = 10
    ASYNC_CAPABLE: ClassVar[bool] = False
    SEARCH_TOOLTIP: ClassVar[str] = ""
    POLICY_CATEGORY: ClassVar[str] = ""
    POLICY_LABELS: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (marked with abstractmethod) are exempt.
        if ABC in cls.__bases__:
            return
        # Skip intermediate abstract classes that still have abstract methods.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("NAME", "INPUT_SCHEMA"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        for attr in ("SUMMARY", "EXAMPLES"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if not isinstance(cls.EXAMPLES, list) or not all(
            isinstance(e, str) for e in cls.EXAMPLES
        ):
            raise TypeError(f"{cls.__name__}.EXAMPLES must be list[str]")
        if not (6 <= len(cls.EXAMPLES) <= 8):
            raise TypeError(
                f"{cls.__name__}.EXAMPLES must have 6–8 entries, got {len(cls.EXAMPLES)}"
            )
        if (
            getattr(cls, "__module__", "").startswith("abilities.")
            and not getattr(cls, "SEARCH_TOOLTIP", "")
        ):
            raise TypeError(f"{cls.__name__} must define a non-empty SEARCH_TOOLTIP")

    # ── The single dispatch chokepoint ────────────────────────────────────────

    @staticmethod
    def dispatch(
        mp: object,
        tool_name: str,
        params: dict,
        call_id: "str | None" = None,
    ) -> str:
        """Every tool call from the ACT loop goes through here.

        Handles: sanitize → policy → timeout thread → execute → record (§4c).
        For ASYNC_CAPABLE tools on async-capable channels: spawn daemon thread,
        return ack immediately, deliver result later via dispatch_message.
        Returns the rendered result string.

        Spec §5 / AC-4 / I1–I14.
        """
        # ── Early cancel check ────────────────────────────────────────────────
        # I14: cancel mid-tool-loop stops further dispatches.
        cancel_ev = getattr(mp, "cancel_event", None)
        if cancel_ev is not None and cancel_ev.is_set():
            return ""

        from services.message_processor import _sanitize_llm_args  # noqa: PLC0415

        # 1. Sanitize params, pop act_summary (I2)
        params = _sanitize_llm_args(dict(params))
        act_summary = params.pop("act_summary", None)
        call_id = call_id or uuid4().hex[:12]

        config = getattr(mp, "config", None)

        # 2. Broadcast start event (no-op unless config.broadcast_to is set) (I4 / N4)
        _emit(config, {
            "type": "act_tool_start",
            "name": tool_name,
            "id": call_id,
            "summary": act_summary,
        })

        # 3. Resolve handler (I9 / I6)
        if tool_name.startswith("_mcp_"):
            # MCP path: policy is enforced inside _dispatch_mcp for _mcp_ tools
            # via seeded policy rows (same as the old ActDispatcherService path).
            result = _dispatch_mcp(mp, tool_name, params)
        else:
            ability = AbilityRegistry.get(tool_name)
            if ability is None:
                result = {"status": "error", "result": f"Unknown tool: {tool_name}"}
            else:
                # 4. Policy gate (I7 / I8)
                blocked = PolicyService.enforce(tool_name, params, config.channel if config else "")
                if blocked is not None:
                    result = blocked
                else:
                    # 5. Async or sync? (J2 / J6)
                    channel = config.channel if config else ""
                    if ability.ASYNC_CAPABLE and _supports_async_delivery(channel):
                        delegate_id = f"{tool_name}_{uuid4().hex[:8]}"
                        cancel_event = threading.Event()
                        _active_delegates[delegate_id] = cancel_event
                        thread = threading.Thread(
                            target=_run_async_delegate,
                            args=(ability, channel, params, delegate_id, cancel_event),
                            daemon=True,
                        )
                        thread.start()
                        result = {
                            "status": "success",
                            "result": (
                                f"{tool_name} dispatched (id: {delegate_id}). "
                                "You will be notified when it completes."
                            ),
                        }
                    else:
                        # Sync path: execute in timeout thread (I13)
                        result = _run_with_timeout(ability, channel, params, ability.TIMEOUT)

        # 6. find_tools side-effect — discovered tools on success only (I12)
        if tool_name == "find_tools" and result.get("status") != "error":
            _process_discovered_tools(mp, result)

        # 7. Record to tool_calls — exactly one row per call (I10)
        result_text = str(result.get("result", ""))
        Ability.record(
            tool_name=tool_name,
            params=params,
            result=result_text,
            transcript_id=getattr(mp, "uid", None),
            ephemeral=True,
        )

        ok = result.get("status") != "error"

        # 8. Broadcast end event (I4 / I5 / N4)
        _emit(config, {
            "type": "act_tool_end",
            "name": tool_name,
            "id": call_id,
            "ok": ok,
        })

        # 9. Return result text (I11)
        return result_text

    # ── Trail statics — the ONLY write/read/render path for the trail ────────
    # Spec §4c / F2 / F3 / F4 / AC-24.

    @staticmethod
    def record(
        tool_name: str,
        params: dict,
        result: str,
        transcript_id: "int | None",
        ephemeral: bool = True,
    ) -> None:
        """Rule 1 — the ONLY write path.  One INSERT, raw params/result, no render.

        Skips silently when transcript_id is None (delegates with skip_transcript
        have no anchor row; the FK column is NOT NULL so we never attempt the
        insert).

        Spec §4c / F2 / AC-24.
        """
        if transcript_id is None:
            logger.debug(
                "[Ability.record] skipping record (no transcript_id): tool=%s",
                tool_name,
            )
            return
        import json as _json  # noqa: PLC0415
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.time_utils import utc_now  # noqa: PLC0415

        try:
            db = get_shared_db_service()
            params_json = _json.dumps(params)
            now = utc_now().isoformat()
            with db.connection() as conn:
                conn.execute(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, ephemeral, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (transcript_id, tool_name, params_json, result,
                     1 if ephemeral else 0, now),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Ability.record] trail write failed (non-fatal): tool=%s transcript=%s: %s",
                tool_name, transcript_id, exc,
            )

    @staticmethod
    def fetch_by_transcript_id(transcript_id: int) -> "list[dict]":
        """Rule 3 (fetch) — every tool_calls row for an ACT loop, oldest→newest.

        Ordered by autoincrement id (not created_at — one-second granularity
        makes created_at ambiguous when several rows land in the same second).

        Spec §4c / F3.
        """
        from services.database_service import get_shared_db_service  # noqa: PLC0415

        try:
            db = get_shared_db_service()
            return db.fetch_all(
                "SELECT id, tool_name, params, result, created_at, ephemeral "
                "FROM tool_calls WHERE transcript_id = ? ORDER BY id",
                (transcript_id,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Ability.fetch_by_transcript_id] query failed for transcript_id=%s: %s",
                transcript_id, exc,
            )
            return []

    @staticmethod
    def render(row: dict) -> str:
        """Rule 2 — the ONLY render path.  One row → its representation.

        Invariant shape: '[tool_name] params → result'
        Same function for the LLM prompt, the UI card, and the audit view.

        Spec §4c / F4.
        """
        tool_name = row.get("tool_name", "unknown")
        params_raw = row.get("params") or "{}"
        result = row.get("result") or ""
        return f"[{tool_name}] {params_raw} → {result}"

    # ── Active-delegate registry ──────────────────────────────────────────────

    @staticmethod
    def cancel_delegate(delegate_id: str) -> bool:
        """Cancel a running async delegate.  Returns True if found.

        Spec §5 / J8.
        """
        ev = _active_delegates.get(delegate_id)
        if ev:
            ev.set()
            return True
        return False

    @staticmethod
    def get_active_delegates() -> list[str]:
        """Return IDs of currently running async delegates.

        Spec §5 / J7.
        """
        return list(_active_delegates.keys())

    # ── Instance method — the actual tool logic ───────────────────────────────

    @abstractmethod
    def run(self, channel: str, params: dict, telemetry: "dict | None") -> "dict | str":
        """Execute the ability.  Called by dispatch() inside a timeout thread.

        Override this method on every Ability subclass — it is the single tool
        entrypoint.

        Must return either:
        - A dict with 'status' and 'result' keys (canonical form).
        - A dict without 'status' (legacy — treated as success).
        - A plain string (treated as success result text).

        Args:
            channel: The conversation channel for this invocation.
            params: Input parameters from the LLM, framework keys stripped.
            telemetry: Optional client telemetry (location, locale, etc.) or None.

        Returns:
            dict when the result is structured data, or str for plain text.

        Spec §5.
        """
        ...

    # ── Schema hooks (unchanged) ──────────────────────────────────────────────

    def get_description(self) -> str:
        """Return the tool description for LLM tool presentation.

        Override to enrich the description at runtime (e.g. find_tools
        appends a discoverable-tools index).  The default returns SUMMARY.
        """
        return self.SUMMARY

    def get_input_schema(self) -> dict:
        """Return the INPUT_SCHEMA for LLM tool presentation.

        Override to enrich the schema at runtime.  The default returns
        the class-level INPUT_SCHEMA unchanged.
        """
        return self.INPUT_SCHEMA

    @classmethod
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Resolve a rich-media payload's runtime state at parse time.

        The default implementation returns the payload unchanged. Override on
        subclasses whose card needs data that does NOT live in the LLM-visible
        tool result.

        Called by ``rich_media_parser.parse()`` exactly once per rich segment.
        """
        return payload


# ── Module-level aliases for patchability ─────────────────────────────────────
#
# Populated AFTER the Ability class is fully defined to avoid the circular
# import that would occur if imported at the top of the file (_registry.py
# imports Ability from here).
#
# unittest.mock.patch("abilities._base.AbilityRegistry.get", ...) works because
# patch targets the name in this module's namespace and then patches the
# attribute on the object it finds there.

def _populate_module_aliases() -> None:
    global AbilityRegistry, PolicyService, WebSocketBroker
    try:
        from abilities._registry import AbilityRegistry as _AR  # noqa: PLC0415
        AbilityRegistry = _AR
    except ImportError:
        pass
    try:
        from services.policy_service import PolicyService as _PS  # noqa: PLC0415
        PolicyService = _PS
    except ImportError:
        pass
    try:
        from services.websocket_broker import WebSocketBroker as _WS  # noqa: PLC0415
        WebSocketBroker = _WS
    except ImportError:
        pass


_populate_module_aliases()
