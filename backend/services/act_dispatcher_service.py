"""
ACT Dispatcher Service

Dispatches internal cognitive actions with timeout enforcement.
Uses the innate skills system for all cognitive operations.

Returns structured results with confidence and notes for downstream
critic evaluation.
"""

import contextvars
import json
import time
import uuid
from typing import Dict, Any
from threading import Thread
import logging

from services.act_action_categories import DETERMINISTIC_ACTIONS as _DETERMINISTIC_ACTIONS, READ_ACTIONS as _READ_ACTIONS


def _load_tool_telemetry() -> dict | None:
    """Pull a flattened telemetry dict for the active client context.

    Returns ``None`` when no client context is stored yet (fresh boot, no
    heartbeat) so abilities can fall back gracefully. Any failure is logged
    and treated as no-telemetry rather than raising — abilities must remain
    callable even when the telemetry table is empty or malformed.
    """
    try:
        from services.client_context_service import ClientContextService
        from services.tool_output_utils import build_tool_telemetry
        ctx = ClientContextService().get()
        if not ctx:
            return None
        return build_tool_telemetry(ctx)
    except Exception as e:
        logging.warning(f"[ACT DISPATCH] telemetry load failed: {e}")
        return None


def _estimate_confidence(action_type: str, raw_result: Any) -> float:
    """Estimate confidence based on action type and result richness.

    Deterministic actions always return 0.92.  Read actions are scored by the
    length of the result string.  All other action types default to 0.50.

    Args:
        action_type: The action category string (e.g. ``"recall"``, ``"memorize"``).
        raw_result: Raw result value returned by the action handler.

    Returns:
        Confidence score in the range [0.0, 1.0].
    """
    if action_type in _DETERMINISTIC_ACTIONS:
        return 0.92
    if action_type in _READ_ACTIONS:
        result_str = str(raw_result) if raw_result else ''
        if len(result_str) > 100:
            return 0.75
        if len(result_str) > 20:
            return 0.60
        return 0.40
    return 0.50


def _extract_notes(action_type: str, action: Dict[str, Any], raw_result: Any) -> str:
    """Extract contextual notes from an action result for critic review.

    Currently produces notes for ``schedule`` (parsed date, recurrence) and
    ``recall`` (query string) action types.

    Args:
        action_type: The action category string.
        action: Full action specification dict.
        raw_result: Raw result value returned by the action handler.

    Returns:
        Semicolon-delimited notes string, or empty string if none apply.
    """
    notes_parts = []
    if action_type == 'schedule':
        if isinstance(raw_result, dict):
            if 'parsed_date' in raw_result:
                notes_parts.append(f"parsed date: {raw_result['parsed_date']}")
            if 'recurrence' in raw_result:
                notes_parts.append(f"recurrence: {raw_result['recurrence']}")
        elif isinstance(raw_result, str) and 'scheduled' in raw_result.lower():
            notes_parts.append(f"schedule result: {raw_result[:200]}")
    if action_type == 'recall':
        query = action.get('query', '')
        if query:
            notes_parts.append(f"query: {query}")
    return '; '.join(notes_parts) if notes_parts else ''


def _summarize_params(action: dict, max_keys: int = 4) -> dict:
    """Extract a small preview of action params for the permission card."""
    skip = {'type', 'action', '_rich_media_ordinal'}
    preview = {}
    for k, v in action.items():
        if k in skip:
            continue
        if len(preview) >= max_keys:
            break
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + '...'
        preview[k] = v
    return preview


def _build_action_description(action_id: str, action: dict) -> str:
    """Build a human-readable one-liner describing the action for the permission card."""
    _VERBS = {
        'email.read': 'Reading an email',
        'email.search': 'Searching emails',
        'email.draft': 'Drafting an email',
        'email.send': 'Sending an email',
        'email.reply': 'Replying to an email',
        'email.forward': 'Forwarding an email',
        'email.manage': 'Managing an email',
        'calendar.update_event': 'Updating a calendar event',
        'calendar.create_event': 'Creating a calendar event',
        'schedule.create': 'Creating a scheduled task',
        'schedule.cancel': 'Cancelling a scheduled task',
        'browser.interact': 'Interacting with a webpage',
        'browser.render': 'Reading a webpage',
        'document.delete': 'Deleting a document',
        'document.create': 'Creating a document',
        'list.delete': 'Deleting a list',
        'memory.forget': 'Forgetting a memory',
        'code_eval': 'Running code',
    }

    # Pick the first available context param in priority order per action.
    _CONTEXT_KEYS = {
        'email.manage': ['operation'],
        'email.draft': ['to', 'subject'],
        'email.send': ['to', 'subject'],
        'email.reply': ['uid'],
        'email.forward': ['uid', 'to'],
        'email.search': ['sender', 'subject', 'keyword'],
        'schedule.create': ['description'],
        'schedule.cancel': ['description'],
        'browser.interact': ['url'],
        'browser.render': ['url'],
        'document.delete': ['name'],
        'document.create': ['name'],
        'list.delete': ['list_name'],
        'calendar.update_event': ['summary'],
        'calendar.create_event': ['summary'],
    }

    base = _VERBS.get(action_id)
    if not base:
        # Fallback: "Email read" → "Email Read"
        base = action_id.replace('.', ' ').replace('_', ' ').title()

    # Special case: email.manage uses operation as the verb
    if action_id == 'email.manage':
        op = action.get('operation', '')
        if op:
            base = op.replace('_', ' ').title() + ' an email'

    # Append first meaningful context value
    context_keys = _CONTEXT_KEYS.get(action_id, [])
    for key in context_keys:
        val = action.get(key)
        if val and isinstance(val, str) and key != 'operation':
            if len(val) > 60:
                val = val[:57] + '...'
            return f"{base} — {val}"
    return base


class ActDispatcherService:
    """Dispatches internal cognitive actions with timeout enforcement."""

    _POLICY_DENY_MSG = "POLICY BLOCK: This action ({action_id}) is permanently blocked by the user's policy settings (state=deny). Do NOT retry — the user must change this in Brain → Policies before it can run."
    _POLICY_UNAVAILABLE_MSG = "POLICY BLOCK: This action ({action_id}) requires explicit user approval but cannot be requested during background processing. It has been logged for the user to review."
    _POLICY_USER_DENIED_MSG = "POLICY BLOCK: The user was shown a permission prompt for this action ({action_id}) and explicitly denied it. Do NOT retry this action in this conversation."
    _POLICY_TIMEOUT_MSG = "POLICY BLOCK: This action ({action_id}) requires user approval. A permission prompt was shown but the user did not respond. Do NOT retry this action unless the user asks."

    def __init__(self, timeout: float = 10.0):
        """Initialize dispatcher with ability handlers.

        Args:
            timeout: Maximum execution time per action (seconds)
        """
        self.timeout = timeout
        self.handlers = {}

        # Per-turn ordinal counter: maps tool name → call count this turn.
        # Reset by reset_turn_ordinals() at the start of each ACT turn.
        # Exposed to tools via the '_rich_media_ordinal' key injected into params.
        self._turn_ordinals: Dict[str, int] = {}

        # Register every ability from the registry as a handler.
        # Each handler calls ability.execute(channel, params, telemetry).
        # Telemetry is fetched fresh per dispatch so abilities see current
        # coordinates, location_name, locale, etc. — without it, location-aware
        # tools like weather fall through to inferior fallbacks (e.g. wttr.in
        # returns mojibake for Maltese place names).
        from abilities._registry import AbilityRegistry
        for ability in AbilityRegistry.all():
            _ability = ability  # capture for closure
            self.handlers[_ability.NAME] = (
                lambda channel, action, _a=_ability: _a.execute(
                    channel,
                    {k: v for k, v in action.items() if k not in ('type', 'exchange_id')},
                    _load_tool_telemetry(),
                )
            )

    def reset_turn_ordinals(self) -> None:
        """Reset per-tool ordinal counters at the start of a new ACT turn.

        Must be called once per MessageProcessor turn before any dispatch_action
        calls so that ordinals are scoped to the turn, not the dispatcher lifetime.
        """
        self._turn_ordinals = {}

    def dispatch_action(self, channel: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single action with timeout enforcement.

        Args:
            channel: Current conversation channel
            action: Action specification dict

        Returns:
            Action result dict with status, result text, and execution time
        """
        action_type = action.get('type', 'unknown')
        start_time = time.time()

        logging.info(f"[ACT DISPATCH] Executing {action_type}")

        # Rich-media ordinal is injected ONLY for the user-facing channel.
        # Subagents (and any other internal channel) must never see the trailer
        # because their natural-language synthesis is consumed by the parent
        # rather than rendered to the user — a span emitted at that hop has no
        # tool_calls row paired to it and would either be paraphrased away by
        # the parent or leak through as raw markup. Stripping the ordinal here
        # is the single physical chokepoint that prevents the trailer from
        # ever reaching a non-user dispatch path.
        action = dict(action)
        if channel == 'user':
            self._turn_ordinals[action_type] = self._turn_ordinals.get(action_type, 0) + 1
            action['_rich_media_ordinal'] = self._turn_ordinals[action_type]

        # Get handler
        handler = self.handlers.get(action_type)
        if not handler:
            # Try wrapper intent routing before falling through to error
            wrapper_result = self._try_wrapper_intent(action_type, action)
            if wrapper_result:
                return wrapper_result

            logging.error(f"[ACT DISPATCH] No handler for '{action_type}'. Registered: {list(self.handlers.keys())}")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unknown action type: {action_type}",
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': '',
            }

        # ── Policy enforcement ──────────────────────────────────────────
        try:
            policy_result = self._enforce_policy(action_type, action, channel)
            if policy_result is not None:
                return policy_result
        except Exception as _pol_err:
            logging.warning(f"[ACT DISPATCH] Policy check failed, allowing: {_pol_err}")

        # Determine effective timeout: ability TIMEOUT ClassVar overrides the default.
        # Falls back to self.timeout when the action type is not a registered ability.
        effective_timeout = self.timeout
        from abilities._registry import AbilityRegistry
        try:
            ability_timeout = AbilityRegistry.get(action_type).TIMEOUT
        except KeyError:
            ability_timeout = None
        if ability_timeout and ability_timeout > effective_timeout:
            effective_timeout = float(ability_timeout)

        return self._execute_with_timeout(action_type, channel, action, handler, effective_timeout, start_time)


    def _execute_with_timeout(
        self,
        action_type: str,
        channel: str,
        action: Dict[str, Any],
        handler: Any,
        effective_timeout: float,
        start_time: float,
    ) -> Dict[str, Any]:
        """Run handler in a daemon thread with timeout; return a result dict.

        Copies calling thread's contextvars so abilities can resolve
        current_processor() (and any ContextVar-backed state bound by
        MessageProcessor.send()). Without this copy the spawned Thread starts
        with a fresh context and current_processor() returns None.
        """
        try:
            from services.vault_service import VaultLockedError

            result_container = {'result': None, 'error': None, 'exc': None}
            ctx = contextvars.copy_context()

            def target():
                """Thread target: invoke the action handler and capture the result."""
                try:
                    result_container['result'] = handler(channel, action)
                except Exception as e:
                    result_container['error'] = str(e)
                    result_container['exc'] = e

            thread = Thread(target=ctx.run, args=(target,))
            thread.daemon = True
            thread.start()
            thread.join(timeout=effective_timeout)

            execution_time = time.time() - start_time

            if thread.is_alive():
                return {
                    'action_type': action_type,
                    'status': 'timeout',
                    'result': f"Action exceeded {effective_timeout}s timeout",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            if result_container['exc'] is not None and isinstance(result_container['exc'], VaultLockedError):
                return {
                    'action_type': action_type,
                    'status': 'error',
                    'result': "This function is currently unavailable. The vault is locked. Notify the user that you could not complete this action because they where logged out",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            if result_container['error']:
                return {
                    'action_type': action_type,
                    'status': 'error',
                    'result': f"Error: {result_container['error']}",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            return self._build_success_result(action_type, action, result_container['result'], execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            logging.exception(f"[ACT DISPATCH] Unexpected error in {action_type}:")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unexpected error: {str(e)}",
                'execution_time': execution_time,
                'confidence': 0.0,
                'notes': '',
            }

    def _build_success_result(
        self,
        action_type: str,
        action: Dict[str, Any],
        raw_result: Any,
        execution_time: float,
    ) -> Dict[str, Any]:
        """Build the success result dict from a raw handler return value."""
        reply_actions = None
        _discovered_tools = None
        if isinstance(raw_result, dict) and 'text' in raw_result:
            reply_actions = raw_result.get('reply_actions')
            _discovered_tools = raw_result.get('_discovered_tools')
            raw_result = raw_result['text']

        confidence = _estimate_confidence(action_type, raw_result)
        notes = _extract_notes(action_type, action, raw_result)

        dispatch_result = {
            'action_type': action_type,
            'status': 'success',
            'result': raw_result,
            'execution_time': execution_time,
            'confidence': confidence,
            'notes': notes,
        }
        if reply_actions:
            dispatch_result['reply_actions'] = reply_actions
        if _discovered_tools:
            dispatch_result['_discovered_tools'] = _discovered_tools
        return dispatch_result

    def _try_wrapper_intent(self, action_type: str, action: Dict[str, Any]) -> Dict[str, Any] | None:
        """Check if a connected wrapper declares the action type as a capability.

        If a capable wrapper is found, emit a CognitiveIntent via IntentService
        and return a structured result dict.  Returns None if no wrapper handles
        this action type (caller should fall through to the error path).

        Args:
            action_type: The action category string (e.g. ``"open_pr"``).
            action: Full action specification dict.

        Returns:
            Result dict with ``status: "intent_emitted"`` on success, or None.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.wrapper_auth_service import WrapperAuthService
            from services.intent_service import IntentService, CognitiveIntent

            db = get_shared_db_service()
            wrapper_svc = WrapperAuthService(db)
            wrappers = wrapper_svc.list_wrappers()

            # Find the first wrapper that declares this action type
            target_wrapper = None
            for w in wrappers:
                caps = w.get("capabilities", {})
                declared_intents = caps.get("intents", [])
                if action_type in declared_intents:
                    target_wrapper = w
                    break

            if not target_wrapper:
                return None

            import uuid
            intent_id = str(uuid.uuid4())
            intent = CognitiveIntent(
                intent_id=intent_id,
                intent_type="execute",
                target_wrapper=target_wrapper["wrapper_id"],
                payload={
                    "action_type": action_type,
                    "params": action.get("params", {}),
                    "description": action.get("description", ""),
                },
                confidence=0.7,
            )

            intent_svc = IntentService()
            intent_svc.emit(intent)

            logging.info(
                f"[ACT DISPATCH] Routed {action_type} to wrapper "
                f"{target_wrapper['wrapper_id']} via intent {intent_id}"
            )

            return {
                "action_type": action_type,
                "status": "intent_emitted",
                "result": f"Intent emitted to wrapper {target_wrapper['wrapper_id']}",
                "target_wrapper": target_wrapper["wrapper_id"],
                "intent_id": intent_id,
                "execution_time": 0.0,
                "confidence": 0.7,
                "notes": f"intent_id={intent_id}; target={target_wrapper['wrapper_id']}",
            }
        except Exception as e:
            logging.debug(f"[ACT DISPATCH] Wrapper intent routing failed: {e}")
            return None

    def _enforce_policy(self, action_type, action, channel):
        """Check policy and return a result dict if blocked, or None to proceed."""
        from services.policy_service import PolicyService, get_defaults, USAGE_CLASS_TO_CONTEXT
        from services.message_processor import current_processor

        action_id = PolicyService.resolve_action_id(action_type, action)

        # Unknown actions (not in default matrix) skip enforcement
        if action_id not in get_defaults():
            return None

        # Resolve context from the current processor's USAGE_CLASS
        proc = current_processor()
        usage_class = getattr(proc, 'USAGE_CLASS', 'chat') if proc else 'chat'
        context = USAGE_CLASS_TO_CONTEXT.get(usage_class, 'chat')

        from services.database_service import get_shared_db_service
        svc = PolicyService(get_shared_db_service())
        state = svc.check(action_id, context)

        if state == 'allow':
            return None

        if state == 'deny':
            svc.log_blocked(action_id, context, 'policy_deny', _summarize_params(action))
            return {
                'action_type': action_type,
                'status': 'policy_denied',
                'result': self._POLICY_DENY_MSG.format(action_id=action_id),
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': f'policy:{action_id}/{context}=deny',
            }

        # state == 'ask'
        if context == 'subconscious':
            svc.log_blocked(action_id, context, 'user_unavailable', _summarize_params(action))
            return {
                'action_type': action_type,
                'status': 'policy_denied',
                'result': self._POLICY_UNAVAILABLE_MSG.format(action_id=action_id),
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': f'policy:{action_id}/{context}=ask(auto-reject)',
            }

        # Chat / subagent: request user permission via REST → Redis
        verdict = self._request_permission(action_id, action, context)
        if verdict != 'approved':
            reason = 'user_denied' if verdict == 'denied' else 'timeout'
            msg = (self._POLICY_USER_DENIED_MSG if verdict == 'denied'
                   else self._POLICY_TIMEOUT_MSG).format(action_id=action_id)
            svc.log_blocked(action_id, context, reason, _summarize_params(action))
            return {
                'action_type': action_type,
                'status': 'policy_denied',
                'result': msg,
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': f'policy:{action_id}/{context}=ask({verdict})',
            }

        return None  # Approved — proceed with execution

    def _request_permission(self, action_id, action, context):
        """Publish a permission_request via Redis and poll for response.

        Polls indefinitely (2s interval) until the user responds via the
        REST endpoint or the WebSocket disconnects.  Returns one of:
          'approved'  — user clicked Allow
          'denied'    — user clicked Deny
          'timeout'   — poll exhausted (should not happen with indefinite wait,
                        but kept as a safety net at 1 hour)
        """
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()

            request_id = str(uuid.uuid4())
            event = json.dumps({
                'type': 'permission_request',
                'request_id': request_id,
                'action_id': action_id,
                'context': context,
                'description': _build_action_description(action_id, action),
            })
            store.publish('output:events', event)

            response_key = f'policy:response:{request_id}'
            deadline = time.monotonic() + 3600  # 1-hour safety net
            while time.monotonic() < deadline:
                raw = store.get(response_key)
                if raw:
                    store.delete(response_key)
                    resp = json.loads(raw)
                    return 'approved' if resp.get('approved', False) else 'denied'
                time.sleep(2)

            return 'timeout'
        except Exception as e:
            logging.warning(f"[ACT DISPATCH] Permission request failed: {e}")
            return 'approved'  # Fail open on Redis errors

