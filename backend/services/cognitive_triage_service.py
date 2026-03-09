"""
Cognitive Triage Service — LLM-based routing with self-eval guardrails.

3-step branching flow:
  1. Empty-input guard (~0ms)
  2. LLM cognitive triage (~100-300ms, lightweight model)
  3. Self-eval sanity check (~0ms, deterministic rules)

The triage LLM reasons about the prompt with context and tool summaries,
returning a structured decision. The self-eval applies deterministic
guardrails to catch obvious errors.

Timeout fallback: if LLM exceeds 500ms, falls back to simple heuristics.
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TRIAGE]"

# Cognitive primitives — always selected for ACT regardless of prompt compliance
_PRIMITIVES = ['recall', 'memorize', 'introspect']
_VALID_SKILLS = {'recall', 'memorize', 'introspect', 'associate', 'schedule', 'list', 'focus', 'autobiography', 'persistent_task', 'document', 'read'}
_CONTEXTUAL_SKILLS = _VALID_SKILLS - set(_PRIMITIVES)  # innate skills that don't need external tools
MAX_CONTEXTUAL_SKILLS = 3   # caps contextual skills; never truncates primitives

_FACTUAL_QUESTION = re.compile(
    r'\b(what|where|when|who|how\s+much|how\s+many|is\s+it|are\s+they|did\s+they|does\s+it)\b.*\?',
    re.IGNORECASE
)

_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\'\)]+',
    re.IGNORECASE
)

_TOOL_INVOCATION_WORDS = frozenset([
    'search', 'find', 'check', 'look up', 'look it up', 'get me', 'fetch',
    'just look', 'pull up', 'google', 'browse', 'open', 'visit',
    'go to', 'read this', 'check this', 'use',
])


_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences or preamble if present."""
    import json
    text = text.strip()
    # Fast path: already valid JSON
    if text.startswith('{'):
        return json.loads(text)
    # Strip markdown code fences
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())
    # Find first { ... } substring
    start = text.find('{')
    if start != -1:
        return json.loads(text[start:])
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def _is_factual_question(text: str) -> bool:
    """Heuristic: does this look like a factual question needing real-world data?"""
    return bool(_FACTUAL_QUESTION.search(text))


def _is_tool_invocation_command(text: str) -> bool:
    """Heuristic: does this look like an imperative tool-invocation command?

    Catches patterns like 'Search arXiv for...', 'Find papers about...', 'Use X to...'
    that don't end with '?' and therefore bypass the factual-question failsafe.
    """
    lower = text.lower()
    return any(w in lower for w in _TOOL_INVOCATION_WORDS)


def _contains_url(text: str) -> bool:
    """Detect presence of an HTTP(S) URL in the message."""
    return bool(_URL_PATTERN.search(text))


@dataclass
class TriageContext:
    context_warmth: float
    memory_confidence: float
    working_memory_turns: int
    gist_count: int
    fact_count: int
    previous_mode: str
    previous_tools: List[str]
    tool_summaries: str
    working_memory_summary: str


@dataclass
class TriageResult:
    branch: str                   # 'social' | 'respond' | 'clarify' | 'act'
    mode: str                     # 'ACKNOWLEDGE' | 'RESPOND' | 'CLARIFY' | 'ACT' | 'IGNORE' | 'CANCEL'
    tools: List[str]              # tool names, only meaningful for ACT
    skills: List[str]             # innate skill names selected for ACT
    confidence_internal: float    # 0-1: confidence memory is sufficient
    confidence_tool_need: float   # 0-1: confidence external tools required
    freshness_risk: float         # 0-1: risk answer needs recent/real-time data
    decision_entropy: float       # abs(confidence_internal - confidence_tool_need)
    reasoning: str
    triage_time_ms: float
    fast_filtered: bool           # True if regex path, no LLM
    self_eval_override: bool
    self_eval_reason: str
    effort_estimate: str = 'moderate'  # trivial | light | moderate | deep


class CognitiveTriageService:
    """3-step branching triage: empty guard → LLM → self-eval → dispatch."""

    def triage(self, text: str, context: TriageContext) -> TriageResult:
        """Main entry point. Empty guard → LLM triage → self-eval."""
        start = time.time()

        # 1. Empty-input guard (~0ms)
        if not text.strip():
            result = TriageResult(
                branch='social', mode='IGNORE', tools=[], skills=[],
                confidence_internal=1.0, confidence_tool_need=0.0,
                freshness_risk=0.0, decision_entropy=0.0,
                reasoning='empty_input', triage_time_ms=(time.time() - start) * 1000,
                fast_filtered=True, self_eval_override=False, self_eval_reason='',
                effort_estimate='trivial',
            )
            return result

        # 2. LLM cognitive triage (~100-300ms)
        result = self._cognitive_triage(text, context)

        # 3. Self-eval sanity check (~0ms, deterministic)
        result = self._self_evaluate(result, text, context)

        result.triage_time_ms = (time.time() - start) * 1000
        return result

    def _cognitive_triage(self, text: str, context: TriageContext) -> TriageResult:
        """LLM call with cognitive-triage.md prompt. Falls back to heuristics on timeout."""
        import json

        try:
            prompt_template = self._load_prompt()
            prompt = (
                prompt_template
                .replace('{{prompt}}', text)
                .replace('{{warmth}}', f'{context.context_warmth:.2f}')
                .replace('{{memory_confidence}}', f'{context.memory_confidence:.2f}')
                .replace('{{fact_count}}', str(context.fact_count))
                .replace('{{turns}}', str(context.working_memory_turns))
                .replace('{{previous_mode}}', context.previous_mode or 'RESPOND')
                .replace('{{working_memory_summary}}', context.working_memory_summary or 'None')
                .replace('{{tool_summaries_grouped}}', context.tool_summaries or 'No tools registered.')
                .replace('{{active_tasks_summary}}', self._get_active_tasks_summary())
            )

            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            data = _extract_json(response_text)

            mode = data.get('mode', 'RESPOND').upper()
            tools = data.get('tools', [])
            if isinstance(tools, str):
                tools = [tools] if tools else []

            confidence_internal = float(data.get('confidence_internal', 0.5))
            confidence_tool_need = float(data.get('confidence_tool_need', 0.5))
            freshness_risk = float(data.get('freshness_risk', 0.0))

            branch = self._mode_to_branch(mode)

            # Parse and validate skills
            raw_skills = data.get('skills', [])
            if isinstance(raw_skills, str):
                raw_skills = [raw_skills] if raw_skills else []

            # Whitelist filter — drop hallucinated or misspelled names
            skills = [s for s in raw_skills if isinstance(s, str) and s in _VALID_SKILLS]

            # De-duplicate while preserving first-seen order
            skills = list(dict.fromkeys(skills))

            # Enforce cognitive primitives for ACT — do not rely on prompt compliance alone
            if mode == 'ACT':
                for p in reversed(_PRIMITIVES):
                    if p not in skills:
                        skills.insert(0, p)

            # Canonical ordering — primitives first, contextual sorted and capped
            primitives_in = [s for s in _PRIMITIVES if s in skills]
            contextual = sorted(s for s in skills if s not in _PRIMITIVES)[:MAX_CONTEXTUAL_SKILLS]
            skills = primitives_in + contextual

            # Extract and validate effort estimate
            effort_estimate = data.get('effort_estimate', 'moderate')
            if effort_estimate not in ('trivial', 'light', 'moderate', 'deep'):
                effort_estimate = 'moderate'

            logger.info(f"{LOG_PREFIX} mode={mode} tools={[t for t in tools if isinstance(t, str)]} skills={skills} effort={effort_estimate}")

            return TriageResult(
                branch=branch,
                mode=mode,
                tools=[t for t in tools if isinstance(t, str)],
                skills=skills,
                confidence_internal=min(1.0, max(0.0, confidence_internal)),
                confidence_tool_need=min(1.0, max(0.0, confidence_tool_need)),
                freshness_risk=min(1.0, max(0.0, freshness_risk)),
                decision_entropy=abs(confidence_internal - confidence_tool_need),
                reasoning=data.get('reasoning', ''),
                triage_time_ms=0.0,
                fast_filtered=False,
                self_eval_override=False,
                self_eval_reason='',
                effort_estimate=effort_estimate,
            )

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM triage failed ({type(e).__name__}: {e}), using heuristic fallback")
            return self._heuristic_fallback(text, context)

    def _heuristic_fallback(self, text: str, context: TriageContext) -> TriageResult:
        """Deterministic fallback when LLM times out or fails."""
        lower = text.lower()

        # Innate skill detection — these never need external tools
        _innate_skill = self._detect_innate_skill(lower)
        if _innate_skill:
            return TriageResult(
                branch='act',
                mode='ACT',
                tools=[],
                skills=list(_PRIMITIVES) + [_innate_skill],
                confidence_internal=0.3,
                confidence_tool_need=0.8,
                freshness_risk=0.2,
                decision_entropy=0.0,
                reasoning=f'heuristic_fallback_innate_{_innate_skill}',
                triage_time_ms=0.0,
                fast_filtered=True,
                self_eval_override=False,
                self_eval_reason='',
            )

        looks_like_command = any(
            w in lower for w in ['search', 'find', 'check', 'look up', 'look it up', 'get me', 'fetch',
                                  'just look', 'pull up', 'google', 'browse', 'open', 'visit',
                                  'go to', 'read this', 'check this']
        ) or _contains_url(lower)
        # Factual real-time signals: questions about events, results, current status, news
        looks_like_realtime = any(
            w in lower for w in ['who won', 'who is', 'who are', 'what happened', 'what is the current',
                                  'what are the latest', 'latest', 'current', 'today', 'tonight',
                                  'this week', 'this year', 'results', 'score', 'winner', 'winners',
                                  'news', 'update', 'live', 'now', 'recently']
        )

        if (looks_like_command or looks_like_realtime) and context.tool_summaries:
            mode = 'ACT'
            branch = 'act'
            freshness = 0.8
        else:
            mode = 'RESPOND'
            branch = 'respond'
            freshness = 0.2

        return TriageResult(
            branch=branch,
            mode=mode,
            tools=[],
            skills=list(_PRIMITIVES) if branch == 'act' else [],
            confidence_internal=0.3 if branch == 'act' else 0.6,
            confidence_tool_need=0.8 if branch == 'act' else 0.2,
            freshness_risk=freshness,
            decision_entropy=0.0,
            reasoning='heuristic_fallback',
            triage_time_ms=0.0,
            fast_filtered=True,
            self_eval_override=False,
            self_eval_reason='',
        )

    @staticmethod
    def _detect_innate_skill(lower: str) -> Optional[str]:
        """Detect innate skill intent from message keywords. Returns skill name or None."""
        # Schedule / reminder patterns
        if any(w in lower for w in ['remind me', 'set a reminder', 'set reminder',
                                     'schedule a', 'schedule this', 'alarm for',
                                     'every morning', 'every evening', 'every day at',
                                     'every week', 'notify me', 'alert me']):
            return 'schedule'
        # List patterns
        if any(w in lower for w in ['add to my list', 'add to the list', 'remove from my list',
                                     'shopping list', 'to-do list', 'todo list',
                                     'check off', 'cross off']):
            return 'list'
        # Focus patterns
        if any(w in lower for w in ['start a focus', 'focus session', 'deep work',
                                     'am i focused', 'end focus']):
            return 'focus'
        # Persistent task patterns
        if any(w in lower for w in ['research this over', 'background task',
                                     'work on this over', 'task status']):
            return 'persistent_task'
        # Document patterns
        if any(w in lower for w in ['my document', 'my warranty', 'uploaded file',
                                     'in the document', 'in my file', 'search my documents',
                                     'document library', 'what does my', 'look up my']):
            return 'document'
        return None

    def _mode_to_branch(self, mode: str) -> str:
        """Map LLM mode string to branch name."""
        return {
            'ACT': 'act',
            'RESPOND': 'respond',
            'CLARIFY': 'clarify',
            'ACKNOWLEDGE': 'social',
            'IGNORE': 'social',
            'CANCEL': 'social',
        }.get(mode, 'respond')

    def _self_evaluate(self, result: TriageResult, text: str, ctx: TriageContext) -> TriageResult:
        """Deterministic sanity check rules. ~0ms."""

        # Rule 1: ACT without tools → keep if contextual skill, defer to ACT loop if tools exist, else downgrade
        if result.branch == 'act' and not result.tools:
            has_contextual_skill = any(s in _CONTEXTUAL_SKILLS for s in result.skills)
            if has_contextual_skill:
                # LLM selected an innate action skill (schedule, list, etc.) — no external tool needed
                if not result.skills:
                    result.skills = list(_PRIMITIVES)
                result.self_eval_override = True
                result.self_eval_reason = 'act_innate_skill'
            elif ctx.tool_summaries:
                # Tools exist but none named — defer tool selection to ACT loop
                if not result.skills:
                    result.skills = list(_PRIMITIVES)
                result.self_eval_override = True
                result.self_eval_reason = 'act_tool_deferred_to_loop'
            else:
                # No external tools — but innate skills don't need them.
                # Triage LLM sometimes omits skills[] when tool_summaries is empty.
                # Recover by running the keyword heuristic directly on the text.
                _recovered = self._detect_innate_skill(text.lower())
                if _recovered and _recovered in _CONTEXTUAL_SKILLS:
                    if _recovered not in result.skills:
                        result.skills.append(_recovered)
                    for _p in _PRIMITIVES:
                        if _p not in result.skills:
                            result.skills.insert(0, _p)
                    result.self_eval_override = True
                    result.self_eval_reason = 'act_innate_skill_recovered'
                else:
                    # Genuinely no tools and no innate skill — downgrade to RESPOND
                    result.branch = 'respond'
                    result.mode = 'RESPOND'
                    result.self_eval_override = True
                    result.self_eval_reason = 'act_no_tools_available'
                    # Log capability gap — user wanted ACT but no tools matched
                    try:
                        from services.self_model_service import SelfModelService
                        SelfModelService().log_capability_gap(
                            request_summary=text[:200],
                            detection_source="triage",
                            confidence=0.5,
                        )
                    except Exception:
                        pass

        # Rule 2: RESPOND on high-freshness question → escalate to ACT
        # Only escalate for genuinely time-sensitive questions (≥0.7 freshness).
        # General knowledge questions (recipes, definitions, how-to) stay in RESPOND.
        if (result.branch == 'respond'
                and result.freshness_risk >= 0.7
                and _is_factual_question(text)
                and ctx.tool_summaries):
            result.branch = 'act'
            result.mode = 'ACT'
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            result.self_eval_override = True
            result.self_eval_reason = 'act_failsafe'

        # Rule 2b: RESPOND on imperative tool-invocation command → escalate to ACT
        # Commands like "Search arXiv for..." don't end with '?' so Rule 2 misses them.
        if (result.branch == 'respond'
                and _is_tool_invocation_command(text)
                and ctx.tool_summaries):
            result.branch = 'act'
            result.mode = 'ACT'
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            result.self_eval_override = True
            result.self_eval_reason = 'act_tool_command'

        # Rule 5: URL in message → escalate to ACT (URL is an unambiguous external-action signal)
        if result.branch == 'respond' and _contains_url(text) and ctx.tool_summaries:
            result.branch = 'act'
            result.mode = 'ACT'
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            result.self_eval_override = True
            result.self_eval_reason = 'act_url_detected'

        # Rule 3: LLM classified as social but message has a substantive question
        if result.branch == 'social' and '?' in text and len(text.split()) > 3:
            if result.mode not in ('CANCEL', 'IGNORE'):
                result.branch = 'respond'
                result.mode = 'RESPOND'
                result.self_eval_override = True
                result.self_eval_reason = 'social_with_question'

        # Rule 4: Anti-oscillation — only suppress if SAME TOOL re-selected
        if (result.branch == 'act'
                and ctx.previous_mode == 'ACT'
                and ctx.previous_tools
                and set(result.tools) == set(ctx.previous_tools)):
            result.confidence_tool_need *= 0.7
            if result.confidence_tool_need < 0.4:
                result.branch = 'respond'
                result.mode = 'RESPOND'
                result.self_eval_override = True
                result.self_eval_reason = 'anti_oscillation_same_tool'

        # Rule 6: Bidirectional effort proportionality
        if result.branch == 'act' and not result.self_eval_override:
            from services.innate_skills.registry import SKILL_EFFORT
            _EFFORT_RANK = {'trivial': 0, 'light': 1, 'moderate': 2, 'deep': 3}
            request_rank = _EFFORT_RANK.get(result.effort_estimate, 2)

            # Direction A: Overpowered — deep tools/skills for trivial request
            if request_rank <= 1:  # trivial or light request
                deep_skills = [s for s in result.skills
                               if s in _CONTEXTUAL_SKILLS
                               and _EFFORT_RANK.get(SKILL_EFFORT.get(s, 'moderate'), 2) >= 3]
                if deep_skills:
                    result.skills = [s for s in result.skills if s not in deep_skills]
                    result.self_eval_override = True
                    result.self_eval_reason = 'effort_proportionality_overpowered'

            # Direction B: Underpowered — only trivial/light tools for deep request
            if request_rank >= 3:  # deep request
                max_skill_rank = max(
                    (_EFFORT_RANK.get(SKILL_EFFORT.get(s, 'moderate'), 2)
                     for s in result.skills if s in _CONTEXTUAL_SKILLS),
                    default=0,
                )
                if max_skill_rank <= 1 and 'persistent_task' not in result.skills:
                    result.skills.append('persistent_task')
                    result.self_eval_override = True
                    result.self_eval_reason = 'effort_proportionality_underpowered'

        return result

    @staticmethod
    def _get_active_tasks_summary() -> str:
        """Get a brief summary of active persistent tasks for triage context."""
        try:
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            db = get_shared_db_service()
            service = PersistentTaskService(db)
            # Get account_id
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM master_account LIMIT 1")
                row = cursor.fetchone()
                account_id = row[0] if row else 1

            active = service.get_active_tasks(account_id)
            if not active:
                return 'None'

            lines = []
            for t in active:
                progress = t.get('progress', {}) or {}
                coverage = progress.get('coverage_estimate', 0)
                lines.append(f'- [{t["status"]}] "{t["goal"][:60]}" ({coverage:.0%})')
            return '\n'.join(lines)
        except Exception:
            return 'None'

    def _load_prompt(self) -> str:
        import os
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        path = os.path.join(prompts_dir, 'cognitive-triage.md')
        with open(path, 'r') as f:
            return f.read()

    def _get_llm(self):
        from services.llm_service import create_llm_service
        from services.config_service import ConfigService
        agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
        return create_llm_service(agent_cfg)
