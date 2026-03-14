"""
ACT Loop Service - Iteration manager for the cognitive ACT loop.

Manages ACT loop with hard iteration cap, timeout, and semantic repetition
detection. Fatigue-based termination has been removed — the loop runs until
it has something to say, hits the iteration cap (30), times out, or detects
semantic repetition.
"""

import re
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import logging

# Strip [TOOL:name]...[/TOOL] wrappers and cost metadata so the LLM
# sees clean, synthesisable text — not programmatic markers.
_TOOL_WRAPPER_RE = re.compile(
    r'\[TOOL:\w+\]\s*',
    re.IGNORECASE,
)
_TOOL_END_RE = re.compile(
    r'\s*\(cost:\s*\d+ms,\s*~\d+\s*tokens\)\s*\[/TOOL\]',
    re.IGNORECASE,
)


def _strip_tool_markers(text: str) -> str:
    """Remove [TOOL:name]...[/TOOL] wrapper and cost metadata from tool result text."""
    text = _TOOL_WRAPPER_RE.sub('', text)
    text = _TOOL_END_RE.sub('', text)
    return text.strip()


# Actions that never block the ACT loop — dispatched to background threads.
# Their results aren't needed for subsequent iteration reasoning.
FIRE_AND_FORGET: frozenset = frozenset({'memorize', 'focus'})

# Iteration threshold after which the soft nudge hint is injected once.
_SOFT_NUDGE_AFTER = 10


class ActLoopService:
    """Manages ACT loop with hard iteration cap, timeout, and repetition detection."""

    def __init__(
        self,
        config: dict,
        cumulative_timeout: float = 60.0,
        per_action_timeout: float = 10.0,
        max_iterations: int = 30,
        cortex_iteration_service=None,
        critic=None,
        dispatcher=None,
        loop_id: str = '',
        scratchpad_enabled: bool = True,
    ):
        """
        Initialize ACT loop service.

        Args:
            config: Configuration dict
            cumulative_timeout: Maximum total execution time (seconds, safety limit)
            per_action_timeout: Maximum time per individual action (seconds)
            max_iterations: Hard cap on iteration count (default 30)
            cortex_iteration_service: Service for exploration bonus calculation
            critic: Optional CriticService instance for post-action verification
            dispatcher: Optional ActDispatcherService instance (reused across iterations)
            loop_id: Unique ID for this loop's scratchpad namespace
            scratchpad_enabled: Whether to gate large results to scratchpad
        """
        self.config = config
        self.cumulative_timeout = cumulative_timeout
        self.per_action_timeout = per_action_timeout
        self.max_iterations = max_iterations
        self.cortex_iteration_service = cortex_iteration_service

        # Critic and dispatcher (injected, not monkey-patched)
        self._critic = critic
        self._dispatcher = dispatcher

        # Loop state
        self.loop_id = loop_id or None  # Set by caller before loop starts
        self.scratchpad_enabled = scratchpad_enabled
        self._scratchpad_counter = 0
        self.iteration_number = 0  # 0-based iteration counter
        self.start_time = time.time()
        self.cumulative_cost = 0.0
        self.current_confidence = 0.0  # Tracks confidence across iterations

        # History tracking
        self.act_history = []  # Action results for context injection
        self.iteration_logs = []  # Iteration data for batch SQLite write
        self.context_extras = {}  # Extra params merged into every action dispatch

        # Per-loop flags (declared here — not monkey-patched externally)
        self._escalation_hint_injected = False
        self.soft_nudge_injected = False  # True after soft nudge emitted at iteration 10

    def _write_to_scratchpad(self, entry: dict) -> None:
        if not self.scratchpad_enabled or not self.loop_id:
            return
        import json
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        key = f"scratchpad:{self.loop_id}:entries"
        store.rpush(key, json.dumps(entry))

    def can_continue(self, mode: str = 'ACT', max_history_tokens: int = 24000, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Determine if ACT loop should continue.

        Exit conditions (in priority order):
        1. Non-ACT terminal mode (RESPOND, CLARIFY, etc.)
        2. Cumulative timeout exceeded (safety limit)
        3. Hard iteration cap reached (max_iterations, default 30)
        4. History token budget exceeded (prevents unbounded prompt growth)

        Semantic repetition detection is handled externally by ACTOrchestrator.

        Args:
            mode: Selected mode from decision gate (default 'ACT')
            max_history_tokens: Maximum estimated tokens for act_history context.
                Estimated via word_count * 1.3 heuristic.

        Returns:
            Tuple of (can_continue: bool, termination_reason: str | None)
        """
        # Terminal modes always exit (RESPOND, CLARIFY, etc)
        if mode != 'ACT':
            return False, f'terminal_mode_{mode.lower()}'

        # Safety: cumulative timeout (hard safety limit)
        elapsed = time.time() - self.start_time
        if elapsed >= self.cumulative_timeout:
            logging.warning(f"[MODE:ACT] [ACT LOOP] Cumulative timeout reached ({elapsed:.2f}s)")
            return False, 'timeout'

        # Safety: max iterations (hard cap)
        if self.iteration_number >= self.max_iterations:
            logging.info(f"[MODE:ACT] [ACT LOOP] Max iterations reached ({self.max_iterations})")
            return False, 'max_iterations'

        # Safety: history token budget (prevents unbounded prompt growth)
        if self.act_history and max_history_tokens > 0:
            history_text = self.get_history_context()
            estimated_tokens = int(len(history_text.split()) * 1.3)
            if estimated_tokens > max_history_tokens:
                logging.info(f"[MODE:ACT] [ACT LOOP] History token budget exceeded ({estimated_tokens} > {max_history_tokens})")
                return False, 'history_token_budget'

        # Can continue ACT mode
        return True, None

    def get_critic_telemetry(self) -> dict:
        """Return critic telemetry if a critic was attached, else empty dict.

        Returns:
            Dict of critic telemetry metrics, or an empty dict when no critic
            was injected.
        """
        if self._critic is not None:
            return self._critic.get_telemetry()
        return {}

    def get_loop_telemetry(self) -> dict:
        """Return loop metrics for telemetry logging.

        Returns:
            Dict containing iterations_used, max_iterations, elapsed_seconds,
            and actions_total.
        """
        return {
            'iterations_used': self.iteration_number,
            'max_iterations': self.max_iterations,
            'elapsed_seconds': round(time.time() - self.start_time, 1),
            'actions_total': len(self.act_history),
        }

    def append_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Append action results to history for context injection.

        Large results (>=1000 estimated tokens) are offloaded to the scratchpad
        when scratchpad_enabled is True. The inline result is truncated to ~200
        words with a reference to the scratchpad entry.

        Args:
            results: List of action result dictionaries from dispatcher
        """
        gated = []
        for result in results:
            result_text = result.get('result', '')
            if not isinstance(result_text, str):
                gated.append(result)
                continue
            estimated_tokens = int(len(result_text.split()) * 1.3)
            if estimated_tokens >= 1000 and self.scratchpad_enabled and self.loop_id:
                self._scratchpad_counter += 1
                sp_id = f"sp_{self._scratchpad_counter:03d}"
                words = result_text.split()
                summary = ' '.join(words[:200])
                self._write_to_scratchpad({
                    'id': sp_id,
                    'source': result.get('action_type', 'unknown'),
                    'iteration': self.iteration_number,
                    'summary': summary,
                    'full_content': result_text,
                    'query_hint': result.get('notes', ''),
                })
                result = dict(result)
                result['result'] = ' '.join(words[:200]) + f' ... [full result in notes as {sp_id}]'
                result['scratchpad_ref'] = sp_id
            gated.append(result)
        self.act_history.extend(gated)

    def get_history_context(self, max_history_tokens: int = 24000) -> str:
        """
        Format ACT history for injection into {{act_history}} prompt placeholder.

        When the full history exceeds max_history_tokens (estimated via word_count * 1.3),
        older entries are moved to the scratchpad and a retrieval hint is prepended.

        Returns:
            Formatted history string showing executed actions
        """
        if not self.act_history:
            return "(none)"

        history = self.act_history

        def _format_entry(idx, result):
            action_type = result['action_type']
            status = result['status']
            result_text = result['result']
            exec_time = result['execution_time']
            card_note = ""
            if result_text == "__CARD_ONLY__":
                display_text = "(card emitted — no text available)"
            elif isinstance(result_text, dict) and result_text.get("card_emitted"):
                display_text = "(card emitted — no text available)"
            elif isinstance(result_text, str) and result_text.startswith("__CARD_EMITTED__\n"):
                card_note = " [card delivered to user]"
                display_text = _strip_tool_markers(result_text.split("\n", 1)[1])
            else:
                display_text = _strip_tool_markers(str(result_text)) if result_text else "(empty)"
            return f"{idx}. [{action_type}] {status.upper()}{card_note}: {display_text} ({exec_time:.2f}s)"

        # Build full history first
        lines = ["## Internal Cognitive Actions"]
        for idx, result in enumerate(history, 1):
            lines.append(_format_entry(idx, result))

        full_text = "\n".join(lines)

        # Check token estimate — prune oldest entries to scratchpad when over budget
        if max_history_tokens > 0:
            estimated_tokens = int(len(full_text.split()) * 1.3)
            if estimated_tokens > max_history_tokens and len(history) > 3:
                # Find the split point: keep as many recent entries as fit in budget,
                # always keeping a minimum of 3. Start from the minimum (last 3) and
                # expand forward until budget is exceeded.
                keep_start = len(history) - 3  # index of first kept entry (minimum)
                for candidate_start in range(len(history) - 3, 0, -1):
                    trial = ["## Internal Cognitive Actions"]
                    for idx, result in enumerate(history[candidate_start:], candidate_start + 1):
                        trial.append(_format_entry(idx, result))
                    if int(len("\n".join(trial).split()) * 1.3) <= max_history_tokens:
                        keep_start = candidate_start
                        break

                pruned = history[:keep_start]
                remaining = history[keep_start:]

                for result in pruned:
                    result_text = result.get('result', '')
                    self._scratchpad_counter += 1
                    self._write_to_scratchpad({
                        'id': f"sp_pruned_{self._scratchpad_counter:03d}",
                        'source': 'pruned',
                        'iteration': result.get('iteration_number', self.iteration_number),
                        'summary': str(result_text)[:200],
                        'full_content': str(result_text),
                        'query_hint': result.get('action_type', ''),
                    })

                pruned_count = len(pruned)
                lines = [
                    "## Internal Cognitive Actions",
                    f"[entries 1-{pruned_count} moved to notes — use notes skill to query]",
                ]
                for idx, result in enumerate(remaining, pruned_count + 1):
                    lines.append(_format_entry(idx, result))
                return "\n".join(lines)

        return full_text

    def execute_actions(self, topic: str, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute actions via dispatcher. Multiple actions run sequentially with output chaining.
        Fire-and-forget actions (memorize, focus) dispatch to background threads and return
        a synthetic success immediately — they never block the next iteration.

        Args:
            topic: Current conversation topic
            actions: List of action specifications

        Returns:
            List of action results from dispatcher (order preserved)
        """
        logging.info(f"[MODE:ACT] [ACT LOOP] Iteration {self.iteration_number}: executing {len(actions)} action(s)")

        # Reuse dispatcher across iterations (P5 fix) — lazy-build on first call
        if self._dispatcher is None:
            from services.act_dispatcher_service import ActDispatcherService
            self._dispatcher = ActDispatcherService(timeout=self.per_action_timeout)

        results = []
        accumulated = {}  # outputs from completed actions, keyed by downstream param name

        for i, action in enumerate(actions):
            action_type = action.get('type', '')

            # Fire-and-forget: dispatch to background, return synthetic result immediately
            if action_type in FIRE_AND_FORGET:
                enriched = {**self.context_extras, **action}
                self._dispatch_async(topic, enriched)
                results.append({
                    'action_type': action_type,
                    'status': 'success',
                    'result': '(running in background)',
                    'execution_time': 0.0,
                    'confidence': 0.92,
                    'notes': 'fire-and-forget',
                })
                logging.debug(f"[MODE:ACT] [ACT LOOP] Step {i+1}/{len(actions)} → {action_type} (fire-and-forget)")
                continue

            # Enrich action params with context_extras as defaults, then accumulated outputs
            enriched = {**self.context_extras, **action}
            for field, value in accumulated.items():
                if not enriched.get(field):
                    enriched[field] = value

            logging.debug(f"[MODE:ACT] [ACT LOOP] Step {i+1}/{len(actions)} → {action.get('type', 'unknown')}")
            result = self._dispatcher.dispatch_action(topic, enriched)
            results.append(result)

            logging.info(
                f"[MODE:ACT] [ACT LOOP] Action {result['action_type']}: "
                f"{result['status']} ({result['execution_time']:.2f}s)"
            )
            # Log result content for schedule actions and for any error/unexpected results
            result_str = str(result.get('result', ''))
            if result['action_type'] == 'schedule' or result_str.startswith('Error:'):
                logging.info(
                    f"[MODE:ACT] [ACT LOOP] {result['action_type']} result: {result_str!r:.200}"
                )

            # Generic output chaining: merge all scalar values from successful dict
            # results into accumulated context for downstream actions (P6 fix)
            if result.get("status") == "success" and isinstance(result.get("result"), dict):
                for key, value in result["result"].items():
                    if isinstance(value, (str, int, float, bool)) and key not in ('status', 'card_emitted'):
                        accumulated[key] = value

        return results

    def _dispatch_async(self, topic: str, action: Dict[str, Any]) -> None:
        """Dispatch an action to a background daemon thread (fire-and-forget)."""
        def _run():
            try:
                self._dispatcher.dispatch_action(topic, action)
            except Exception as e:
                logging.warning(f"[MODE:ACT] [ACT LOOP] Fire-and-forget {action.get('type')} failed: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"act-ff-{action.get('type')}")
        t.start()

    def log_iteration(
        self,
        started_at: float,
        completed_at: float,
        chosen_mode: str,
        chosen_confidence: float,
        actions_executed: List[Dict] = None,
        frontal_cortex_response: Dict = None,
        termination_reason: Optional[str] = None,
        alternative_paths: List[Dict] = None,
        decision_data: Dict = None
    ) -> None:
        """
        Log iteration data to iteration_logs list for batch write.

        Args:
            started_at: Iteration start timestamp (time.time())
            completed_at: Iteration end timestamp (time.time())
            chosen_mode: Final selected mode
            chosen_confidence: Confidence in chosen path
            actions_executed: Actions that were executed (if ACT mode)
            frontal_cortex_response: Full response from LLM
            termination_reason: Reason if loop terminates
            alternative_paths: Alternative paths from LLM (optional)
            decision_data: Decision gate results dict (optional)
        """
        execution_time_ms = (completed_at - started_at) * 1000
        actions_executed = actions_executed or []
        alternative_paths = alternative_paths or []
        frontal_cortex_response = frontal_cortex_response or {}

        iteration_log = {
            'iteration_number': self.iteration_number,
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at)),
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(completed_at)),
            'execution_time_ms': execution_time_ms,
            'chosen_mode': chosen_mode,
            'chosen_confidence': chosen_confidence,
            'alternative_paths': alternative_paths,
            'termination_reason': termination_reason,
            'actions_executed': actions_executed,
            'action_count': len(actions_executed),
            'action_success_count': sum(1 for a in actions_executed if a.get('status') == 'success'),
            'frontal_cortex_response': frontal_cortex_response,
            'cumulative_cost': self.cumulative_cost,
        }

        # Include net_value from decision_data if available
        if decision_data:
            iteration_log['net_value'] = decision_data.get('net_value', 0.0)

        self.iteration_logs.append(iteration_log)
        logging.debug(f"[MODE:{chosen_mode}] [ACT LOOP] Logged iteration {self.iteration_number}, total logs: {len(self.iteration_logs)}")
