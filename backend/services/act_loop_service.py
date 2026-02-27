"""
ACT Loop Service - Fatigue-based cognitive iteration manager.

Manages ACT loop with fatigue-based termination. Each action type has a different
base cost, and fatigue grows non-linearly with iteration depth.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import logging


ACTION_FATIGUE_COSTS = {
    'introspect': 0.5, 'memorize': 0.8, 'recall': 1.0, 'associate': 1.0,
}


class ActLoopService:
    """Manages ACT loop with fatigue-based termination and concurrent execution."""

    def __init__(
        self,
        config: dict,
        cumulative_timeout: float = 60.0,
        per_action_timeout: float = 10.0,
        max_iterations: int = 7,
        cortex_iteration_service=None
    ):
        """
        Initialize ACT loop service.

        Args:
            config: Configuration dict with cost parameters
            cumulative_timeout: Maximum total execution time (seconds, safety limit)
            per_action_timeout: Maximum time per individual action (seconds)
            max_iterations: Hard cap on iteration count (default 7)
            cortex_iteration_service: Service for exploration bonus calculation
        """
        self.config = config
        self.cumulative_timeout = cumulative_timeout
        self.per_action_timeout = per_action_timeout
        self.max_iterations = max_iterations
        self.cortex_iteration_service = cortex_iteration_service

        # Loop state
        self.loop_id = None  # Set by caller before loop starts
        self.iteration_number = 0  # 0-based iteration counter
        self.start_time = time.time()
        self.cumulative_cost = 0.0
        self.current_confidence = 0.0  # Tracks confidence across iterations

        # Fatigue state
        self.fatigue = 0.0
        self.fatigue_budget = config.get('fatigue_budget', 10.0)
        self.fatigue_growth_rate = config.get('fatigue_growth_rate', 0.3)
        self.fatigue_costs = config.get('fatigue_costs', {})  # per-action overrides

        # History tracking
        self.act_history = []  # Action results for context injection
        self.iteration_logs = []  # Iteration data for batch PostgreSQL write
        self.context_extras = {}  # Extra params merged into every action dispatch

    def can_continue(self, mode: str = 'ACT', max_history_tokens: int = 4000, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Determine if ACT loop should continue based on fatigue, timeout, iteration cap,
        and history token budget.

        Fatigue is the primary termination signal. Timeout, max_iterations, and
        max_history_tokens are safety caps.

        Args:
            mode: Selected mode from decision gate (default 'ACT')
            max_history_tokens: Maximum estimated tokens for act_history context.
                Prevents unbounded prompt growth across iterations.
                Estimated via word_count * 1.3 heuristic.

        Returns:
            Tuple of (can_continue: bool, termination_reason: str | None)
        """
        # Terminal modes always exit (RESPOND, CLARIFY, etc)
        if mode != 'ACT':
            return False, f'terminal_mode_{mode.lower()}'

        # Primary: fatigue budget
        if self.fatigue >= self.fatigue_budget:
            logging.info(f"[MODE:ACT] [ACT LOOP] Fatigue exhausted ({self.fatigue:.2f}/{self.fatigue_budget})")
            return False, 'fatigue_exhausted'

        # Safety: cumulative timeout (hard safety limit)
        elapsed = time.time() - self.start_time
        if elapsed >= self.cumulative_timeout:
            logging.warning(f"[MODE:ACT] [ACT LOOP] Cumulative timeout reached ({elapsed:.2f}s)")
            return False, 'timeout'

        # Safety: max iterations (hard cap, should rarely trigger)
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

    def accumulate_fatigue(self, actions_executed: list, iteration_number: int) -> float:
        """Accumulate fatigue from executed actions with non-linear growth."""
        added = 0.0
        for result in actions_executed:
            action_type = result.get('action_type', 'unknown')
            base_cost = self.fatigue_costs.get(
                action_type, ACTION_FATIGUE_COSTS.get(action_type, 1.0)
            )
            cost = base_cost * (1.0 + self.fatigue_growth_rate * iteration_number)
            added += cost
        self.fatigue += added
        return added

    @staticmethod
    def estimate_net_value(actions_executed: list, iteration_number: int) -> float:
        """Heuristic net value of actions — logs to cortex_iterations for strategy analysis."""
        value = 0.0
        for result in actions_executed:
            if result['status'] == 'success':
                result_text = str(result.get('result', ''))
                if len(result_text) > 50:
                    value += 1.0
                else:
                    value += 0.3
            elif result['status'] == 'timeout':
                value -= 0.5
            else:
                value -= 0.3
        return value * (1.0 / (1.0 + 0.2 * iteration_number))

    def get_fatigue_telemetry(self) -> dict:
        """Return fatigue metrics for telemetry logging."""
        return {
            'fatigue_total': round(self.fatigue, 2),
            'fatigue_budget': self.fatigue_budget,
            'fatigue_utilization': round(self.fatigue / self.fatigue_budget, 3) if self.fatigue_budget > 0 else 0,
            'iterations_used': self.iteration_number,
            'max_iterations': self.max_iterations,
            'elapsed_seconds': round(time.time() - self.start_time, 1),
            'actions_total': len(self.act_history),
            'budget_headroom': round(self.fatigue_budget - self.fatigue, 2),
        }

    def append_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Append action results to history for context injection.

        Args:
            results: List of action result dictionaries from dispatcher
        """
        self.act_history.extend(results)

    def get_history_context(self, max_history_tokens: int = 4000) -> str:
        """
        Format ACT history for injection into {{act_history}} prompt placeholder.

        When the full history exceeds max_history_tokens (estimated via word_count * 1.3),
        truncates older entries but keeps the most recent 3 intact.

        Returns:
            Formatted history string showing executed actions
        """
        if not self.act_history:
            return "(none)"

        history = self.act_history

        # Build full history first
        lines = ["## Internal Cognitive Actions"]
        for idx, result in enumerate(history, 1):
            action_type = result['action_type']
            status = result['status']
            result_text = result['result']
            exec_time = result['execution_time']

            display_text = "(card emitted)" if result_text == "__CARD_ONLY__" else result_text
            lines.append(f"{idx}. [{action_type}] {status.upper()}: {display_text} ({exec_time:.2f}s)")

        full_text = "\n".join(lines)

        # Check token estimate
        if max_history_tokens > 0:
            estimated_tokens = int(len(full_text.split()) * 1.3)
            if estimated_tokens > max_history_tokens and len(history) > 3:
                # Keep most recent 3 entries, summarize older ones
                truncated_count = len(history) - 3
                recent = history[-3:]
                lines = [
                    "## Internal Cognitive Actions",
                    f"[{truncated_count} earlier action(s) truncated for brevity]",
                ]
                for idx, result in enumerate(recent, truncated_count + 1):
                    action_type = result['action_type']
                    status = result['status']
                    result_text = result['result']
                    exec_time = result['execution_time']
                    display_text = "(card emitted)" if result_text == "__CARD_ONLY__" else result_text
                    lines.append(f"{idx}. [{action_type}] {status.upper()}: {display_text} ({exec_time:.2f}s)")
                return "\n".join(lines)

        return full_text

    def execute_actions(self, topic: str, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute actions via dispatcher. Multiple actions run sequentially with output chaining.

        Args:
            topic: Current conversation topic
            actions: List of action specifications

        Returns:
            List of action results from dispatcher (order preserved)
        """
        from services.act_dispatcher_service import ActDispatcherService

        logging.info(f"[MODE:ACT] [ACT LOOP] Iteration {self.iteration_number}: executing {len(actions)} action(s)")

        dispatcher = ActDispatcherService(timeout=self.per_action_timeout)
        results = []
        accumulated = {}  # outputs from completed actions, keyed by downstream param name

        for i, action in enumerate(actions):
            # Enrich action params with context_extras as defaults, then accumulated outputs
            enriched = {**self.context_extras, **action}
            for field, value in accumulated.items():
                if not enriched.get(field):
                    enriched[field] = value

            logging.debug(f"[MODE:ACT] [ACT LOOP] Step {i+1}/{len(actions)} → {action.get('type', 'unknown')}")
            result = dispatcher.dispatch_action(topic, enriched)
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

            # Harvest outputs for downstream chaining
            if result.get("status") == "success" and isinstance(result.get("result"), dict):
                out = result["result"]
                if "timezone" in out:
                    accumulated["timezone"] = out["timezone"]
                if "city" in out:
                    accumulated["location"] = out["city"]
                if "latitude" in out and "longitude" in out:
                    accumulated["lat"] = out["latitude"]
                    accumulated["lon"] = out["longitude"]

        return results

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

        # Include full cost breakdown if decision_data is available
        if decision_data:
            cost_breakdown = decision_data.get('cost_breakdown', {})
            iteration_log.update({
                'iteration_cost': cost_breakdown.get('iteration_cost', 0.0),
                'diminishing_cost': cost_breakdown.get('diminishing_cost', 0.0),
                'uncertainty_cost': cost_breakdown.get('uncertainty_cost', 0.0),
                'action_base_cost': cost_breakdown.get('action_base_cost', 0.0),
                'total_cost': decision_data.get('total_cost', 0.0),
                'efficiency_score': decision_data.get('efficiency', 0.0),
                'net_value': decision_data.get('net_value', 0.0),
                'decision_override': decision_data.get('decision_override', False),
                'overridden_mode': decision_data.get('overridden_mode'),
                'exploration_bonus': decision_data.get('exploration_bonus', 0.0),
            })

        self.iteration_logs.append(iteration_log)
        logging.debug(f"[MODE:{chosen_mode}] [ACT LOOP] Logged iteration {self.iteration_number}, total logs: {len(self.iteration_logs)}")

    def create_fatigue_response(self, last_response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback response when cognitive fatigue hits (low efficiency).

        Args:
            last_response_data: Most recent response from frontal cortex

        Returns:
            Response dict with mode=RESPOND and fatigue explanation
        """
        partial_response = last_response_data.get('response', '')

        fallback_text = (
            "I've explored multiple approaches and reached my cognitive limit. "
            "Based on my current understanding: "
        )

        if partial_response:
            fallback_text += partial_response
        else:
            fallback_text += "I can provide a tentative answer, but clarification would help."

        return {
            'mode': 'RESPOND',
            'modifiers': ['FATIGUE'],
            'response': fallback_text,
            'generation_time': last_response_data.get('generation_time', 0),
            'actions': [],
            'confidence': last_response_data.get('confidence', 0.3)
        }
