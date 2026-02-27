"""
Plan Decomposition Service — LLM-powered goal → step DAG decomposition.

Decomposes a high-level goal into a structured step graph where each step
declares dependencies. The DAG is validated (no cycles, valid refs) and
stored in the persistent task's progress JSONB.

Step statuses: pending → ready → in_progress → completed | failed | skipped
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PLAN DECOMPOSITION]"


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


class PlanDecompositionService:
    """Decomposes goals into executable step DAGs via LLM."""

    def __init__(self):
        from services.config_service import ConfigService
        self.config = ConfigService.resolve_agent_config("plan-decomposition")
        self.prompt_template = ConfigService.get_agent_prompt("plan-decomposition")
        self.min_steps = self.config.get('min_steps', 2)
        self.max_steps = self.config.get('max_steps', 8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.auto_accept_confidence = self.config.get('auto_accept_confidence', 0.9)

    # ── Public API ────────────────────────────────────────────────

    def decompose(self, goal: str, scope: Optional[str] = None,
                  memory_context: str = '') -> Optional[Dict[str, Any]]:
        """
        Decompose a goal into a step DAG via LLM.

        Returns plan dict with 'steps', 'decomposition_confidence', etc.
        Returns None on failure.
        """
        prompt = self._build_prompt(goal, scope, memory_context)
        raw = self._call_llm(prompt)
        if not raw:
            return None

        plan = self._parse_response(raw)
        if not plan:
            return None

        steps = plan.get('steps', [])

        # Validate DAG structure
        if not self.validate_dag(steps):
            logger.warning(f"{LOG_PREFIX} DAG validation failed for goal: {goal[:80]}")
            return None

        # Validate step quality
        quality_issues = self.validate_step_quality(steps)
        if quality_issues:
            logger.warning(f"{LOG_PREFIX} Step quality issues: {quality_issues}")
            return None

        # Check step count bounds
        if len(steps) < self.min_steps or len(steps) > self.max_steps:
            logger.warning(
                f"{LOG_PREFIX} Step count {len(steps)} outside bounds "
                f"[{self.min_steps}, {self.max_steps}]"
            )
            return None

        confidence = plan.get('decomposition_confidence', 0.0)
        if confidence < self.confidence_threshold:
            logger.info(
                f"{LOG_PREFIX} Confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold}"
            )
            return None

        # Initialize step statuses
        for step in steps:
            step['status'] = 'pending'
            step.setdefault('result_summary', '')
            step.setdefault('started_at', None)
            step.setdefault('completed_at', None)

        cost_class = self.estimate_cost({'steps': steps})

        return {
            'version': 1,
            'steps': steps,
            'decomposed_at': datetime.now(timezone.utc).isoformat(),
            'decomposition_confidence': confidence,
            'cost_class': cost_class,
            'blocked_on': None,
            'blocked_reason': None,
        }

    @staticmethod
    def validate_dag(steps: List[Dict]) -> bool:
        """
        Validate step DAG: no cycles, valid dependency references, at least one root.

        Uses Kahn's algorithm for topological sort (cycle detection).
        """
        if not steps:
            return False

        step_ids = {s['id'] for s in steps}

        # Check all dependency references are valid
        for step in steps:
            for dep in step.get('depends_on', []):
                if dep not in step_ids:
                    logger.warning(f"{LOG_PREFIX} Invalid dependency ref: {dep}")
                    return False

        # At least one root (step with no dependencies)
        roots = [s for s in steps if not s.get('depends_on')]
        if not roots:
            logger.warning(f"{LOG_PREFIX} No root steps (all have dependencies)")
            return False

        # Kahn's algorithm — cycle detection via topological sort
        in_degree = {s['id']: len(s.get('depends_on', [])) for s in steps}
        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        sorted_count = 0

        # Build adjacency: step_id → list of dependents
        dependents = {s['id']: [] for s in steps}
        for s in steps:
            for dep in s.get('depends_on', []):
                dependents[dep].append(s['id'])

        while queue:
            node = queue.popleft()
            sorted_count += 1
            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if sorted_count != len(steps):
            logger.warning(f"{LOG_PREFIX} Cycle detected in step DAG")
            return False

        return True

    @staticmethod
    def validate_step_quality(steps: List[Dict]) -> List[str]:
        """
        Reject garbage plans: too-short descriptions, too-long descriptions,
        or semantically duplicate steps.

        Returns list of rejection reasons (empty = valid).
        """
        issues = []

        for step in steps:
            desc = step.get('description', '').strip()
            word_count = len(desc.split())

            if word_count < 4:
                issues.append(
                    f"Step {step['id']}: description too short "
                    f"({word_count} words, min 4)"
                )
            elif word_count > 30:
                issues.append(
                    f"Step {step['id']}: description too long "
                    f"({word_count} words, max 30)"
                )

        # Check for semantic duplicates (Jaccard > 0.7)
        for i, a in enumerate(steps):
            for b in steps[i + 1:]:
                sim = _jaccard_similarity(
                    a.get('description', ''),
                    b.get('description', ''),
                )
                if sim > 0.7:
                    issues.append(
                        f"Steps {a['id']} and {b['id']} are semantically "
                        f"duplicate (Jaccard={sim:.2f})"
                    )

        return issues

    @staticmethod
    def get_ready_steps(plan: Dict) -> List[Dict]:
        """
        Return steps where status='pending' and all depends_on are
        completed or skipped.

        Ordered by: shallowest depth first, cheapest first (no tools),
        shortest description, then step id.
        """
        steps = plan.get('steps', [])
        if not steps:
            return []

        # Build status lookup
        status_map = {s['id']: s.get('status', 'pending') for s in steps}
        resolved = {'completed', 'skipped'}

        # Compute dependency depth for each step
        depth_cache: Dict[str, int] = {}

        def _depth(step_id: str) -> int:
            if step_id in depth_cache:
                return depth_cache[step_id]
            step = next((s for s in steps if s['id'] == step_id), None)
            if not step or not step.get('depends_on'):
                depth_cache[step_id] = 0
                return 0
            d = 1 + max(_depth(dep) for dep in step['depends_on'])
            depth_cache[step_id] = d
            return d

        ready = []
        for step in steps:
            if step.get('status') != 'pending':
                continue
            deps = step.get('depends_on', [])
            if all(status_map.get(d) in resolved for d in deps):
                ready.append(step)

        # Sort: shallowest depth → cheapest (no tools) → shortest desc → step id
        ready.sort(key=lambda s: (
            _depth(s['id']),
            0 if not s.get('tools_needed') else 1,
            len(s.get('description', '')),
            s['id'],
        ))

        return ready

    @staticmethod
    def update_step_status(
        plan: Dict,
        step_id: str,
        new_status: str,
        result_summary: str = '',
        skip_reason: str = '',
        skipped_by: str = '',
        failure_reason: str = '',
        retryable: bool = False,
    ) -> Dict:
        """
        Mutate step status and recalculate plan metadata.

        Skipped steps count toward progress. Updates blocked_on/blocked_reason
        if applicable.
        """
        steps = plan.get('steps', [])
        step = next((s for s in steps if s['id'] == step_id), None)
        if not step:
            logger.warning(f"{LOG_PREFIX} Step {step_id} not found in plan")
            return plan

        now = datetime.now(timezone.utc).isoformat()
        step['status'] = new_status

        if new_status == 'in_progress' and not step.get('started_at'):
            step['started_at'] = now
        elif new_status in ('completed', 'failed', 'skipped'):
            step['completed_at'] = now

        if result_summary:
            step['result_summary'] = result_summary
        if skip_reason:
            step['skip_reason'] = skip_reason
        if skipped_by:
            step['skipped_by'] = skipped_by
        if failure_reason:
            step['failure_reason'] = failure_reason
        if new_status == 'failed':
            step['retryable'] = retryable

        # Recalculate blocked state
        plan = PlanDecompositionService._update_blocked_state(plan)

        return plan

    @staticmethod
    def estimate_cost(plan: Dict) -> str:
        """
        Classify plan as 'cheap' (only innate skills) or 'expensive'
        (requires external tools).
        """
        for step in plan.get('steps', []):
            if step.get('tools_needed'):
                return 'expensive'
        return 'cheap'

    @staticmethod
    def get_plan_coverage(plan: Dict) -> float:
        """Calculate coverage: (completed + skipped) / total."""
        steps = plan.get('steps', [])
        if not steps:
            return 0.0
        resolved = sum(
            1 for s in steps
            if s.get('status') in ('completed', 'skipped')
        )
        return resolved / len(steps)

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _update_blocked_state(plan: Dict) -> Dict:
        """
        Check if the plan is blocked: no ready steps but some still pending.
        """
        steps = plan.get('steps', [])
        status_map = {s['id']: s.get('status', 'pending') for s in steps}
        resolved = {'completed', 'skipped'}
        terminal = {'completed', 'skipped', 'failed'}

        pending_steps = [s for s in steps if s.get('status') == 'pending']
        if not pending_steps:
            plan['blocked_on'] = None
            plan['blocked_reason'] = None
            return plan

        # Check if any pending step has all deps resolved
        any_ready = False
        for step in pending_steps:
            deps = step.get('depends_on', [])
            if all(status_map.get(d) in resolved for d in deps):
                any_ready = True
                break

        if any_ready:
            plan['blocked_on'] = None
            plan['blocked_reason'] = None
        else:
            # Find the blocking dependencies
            blockers = set()
            for step in pending_steps:
                for dep in step.get('depends_on', []):
                    dep_status = status_map.get(dep)
                    if dep_status == 'failed':
                        blockers.add(dep)
            plan['blocked_on'] = list(blockers) if blockers else None
            plan['blocked_reason'] = (
                f"dependencies {list(blockers)} failed"
                if blockers else "waiting on in-progress dependencies"
            )

        return plan

    def _build_prompt(self, goal: str, scope: Optional[str],
                      memory_context: str) -> str:
        """Fill the prompt template with context."""
        prompt = self.prompt_template
        prompt = prompt.replace('{{goal}}', goal)
        prompt = prompt.replace('{{scope}}', scope or 'No specific scope defined')
        prompt = prompt.replace('{{memory_context}}', memory_context or 'None available')

        # Gather available skills
        skills_text = self._get_available_skills()
        prompt = prompt.replace('{{available_skills}}', skills_text)

        # Gather available tools
        tools_text = self._get_available_tools()
        prompt = prompt.replace('{{available_tools}}', tools_text)

        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM for plan decomposition."""
        try:
            from services.llm_service import create_llm_service
            llm = create_llm_service(self.config)
            response = llm.send_message(
                "You are a planning agent. Decompose goals into step graphs.",
                prompt,
            )
            return response.text
        except Exception as e:
            logger.error(f"{LOG_PREFIX} LLM call failed: {e}", exc_info=True)
            return None

    @staticmethod
    def _parse_response(raw: str) -> Optional[Dict]:
        """Parse LLM JSON response, stripping think tags if present."""
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if '<think>' in cleaned:
            cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"{LOG_PREFIX} Failed to parse LLM response: {e}")
            return None

    @staticmethod
    def _get_available_skills() -> str:
        """List innate skills for the prompt."""
        skills = [
            ('recall', 'Memory retrieval and search'),
            ('memorize', 'Explicit memory encoding'),
            ('introspect', 'Self-reflection and meta-cognition'),
            ('associate', 'Semantic relationship discovery'),
            ('schedule', 'Reminders and task scheduling'),
            ('autobiography', 'Personal narrative synthesis'),
            ('list', 'List management and organization'),
            ('focus', 'Focus session management'),
            ('persistent_task', 'Background task management'),
        ]
        return '\n'.join(f'- **{name}**: {desc}' for name, desc in skills)

    @staticmethod
    def _get_available_tools() -> str:
        """List available external tools from the tool registry."""
        try:
            from services.database_service import get_shared_db_service
            from services.tool_registry_service import ToolRegistryService
            db = get_shared_db_service()
            registry = ToolRegistryService(db)
            tools = registry.list_tools()
            if not tools:
                return 'No external tools configured.'
            lines = []
            for t in tools:
                name = t.get('name', 'unknown')
                desc = t.get('description', 'No description')
                lines.append(f'- **{name}**: {desc}')
            return '\n'.join(lines)
        except Exception:
            return 'No external tools configured.'
