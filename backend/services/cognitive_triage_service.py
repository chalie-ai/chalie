"""
Cognitive Triage Service — LLM-based routing with social fast filter and self-eval.

Replaces the 12-step linear pipeline with a 4-step branching flow:
  1. Social filter (~1ms, regex) → fast template response
  2. LLM cognitive triage (~100-300ms, lightweight model)
  3. Self-eval sanity check (~0ms, deterministic rules)
  4. Branch dispatch → RESPOND, CLARIFY, or ACT

The triage LLM reasons about the prompt with context and tool summaries,
returning a structured decision. The self-eval applies deterministic
guardrails to catch obvious errors.

Timeout fallback: if LLM exceeds 500ms, falls back to simple heuristics.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TRIAGE]"

# Cognitive primitives — always selected for ACT regardless of prompt compliance
_PRIMITIVES = ['recall', 'memorize', 'introspect']
_VALID_SKILLS = {'recall', 'memorize', 'introspect', 'associate', 'schedule', 'list', 'focus', 'autobiography'}
MAX_CONTEXTUAL_SKILLS = 3   # caps contextual skills; never truncates primitives

# Innate action skill patterns — ACT is required even when no external tool is listed
_INNATE_ACTION_PATTERNS = re.compile(
    r'\b(remind\s+me|set\s+(a\s+)?(reminder|alarm|schedule)|schedule\s+(a\s+)?(reminder|task|message)|'
    r'add\s+.{1,40}\s+to\s+(my\s+)?(list|shopping|to.?do)|remove\s+.{1,40}\s+from\s+(my\s+)?(list)|'
    r'cancel\s+(my\s+|all\s+|the\s+)?(reminder|alarm|schedule|task|notification|event|appointment)s?|'
    r'delete\s+(my\s+|all\s+|the\s+)?(reminder|alarm|schedule|task|notification|event|appointment)s?|'
    r'turn\s+off\s+(my\s+|all\s+|the\s+)?(reminder|alarm|schedule|task|notification|event|appointment)s?|'
    r'every\s+(morning|evening|day|hour|week|month|\d+\s+minutes?|few\s+hours?))\b',
    re.IGNORECASE
)

# Social filter regex patterns (reused from IntentClassifierService)
_GREETING_PATTERNS = re.compile(
    r'^(hey|hi|hello|yo|sup|what\'?s\s*up|howdy|hiya|heya|greetings|'
    r'good\s*(morning|afternoon|evening))\b',
    re.IGNORECASE
)

_POSITIVE_FEEDBACK = re.compile(
    r'\b(thanks|thank\s+you|great|perfect|awesome|exactly|that\s+works|correct|'
    r'good|nice|helpful|got\s+it|understood)\b',
    re.IGNORECASE
)

_CANCEL_PATTERNS = [
    re.compile(r'\bnever\s*mind\b', re.IGNORECASE),
    re.compile(r'\bignore\s+that\b', re.IGNORECASE),
    re.compile(r'\bstop\s+(searching|looking|checking)\b', re.IGNORECASE),
    re.compile(r'\bforget\s+(it|about\s+it|that)\b', re.IGNORECASE),
    re.compile(r'\bcancel\b(?!\s+(?:(?:my|all|the|a|this|that)\s+)?(?:reminder|alarm|schedule|task|notification|event|recurring|appointment)s?\b)', re.IGNORECASE),
    re.compile(r'\bdon\'?t\s+bother\b', re.IGNORECASE),
]

_SELF_RESOLVED_PATTERNS = [
    re.compile(r'\b(found|figured|sorted|solved|got)\s+(it|that|this)\s*(out|now|myself)?\b', re.IGNORECASE),
    re.compile(r'\b(all\s+good|no\s+worries|no\s+need)\b', re.IGNORECASE),
]

_FACTUAL_QUESTION = re.compile(
    r'\b(what|where|when|who|how\s+much|how\s+many|is\s+it|are\s+they|did\s+they|does\s+it)\b.*\?',
    re.IGNORECASE
)


def _is_factual_question(text: str) -> bool:
    """Heuristic: does this look like a factual question needing real-world data?"""
    return bool(_FACTUAL_QUESTION.search(text))


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


class CognitiveTriageService:
    """4-step branching triage: social filter → LLM → self-eval → dispatch."""

    def triage(self, text: str, context: TriageContext) -> TriageResult:
        """Main entry point. Social filter → LLM triage → self-eval."""
        start = time.time()

        # 1. Fast social filter (~1ms, regex)
        social = self._social_filter(text)
        if social:
            social.triage_time_ms = (time.time() - start) * 1000
            return social

        # 2. LLM cognitive triage (~100-300ms)
        result = self._cognitive_triage(text, context)

        # 3. Self-eval sanity check (~0ms, deterministic)
        result = self._self_evaluate(result, text, context)

        result.triage_time_ms = (time.time() - start) * 1000
        return result

    def _social_filter(self, text: str) -> Optional[TriageResult]:
        """Regex fast exit. Returns TriageResult or None if not social."""
        stripped = text.strip()
        if not stripped:
            return self._make_social('IGNORE')

        if _GREETING_PATTERNS.match(stripped):
            # Greetings with questions → let LLM decide
            if '?' in stripped and len(stripped.split()) > 3:
                return None
            return self._make_social('ACKNOWLEDGE')

        if _POSITIVE_FEEDBACK.search(stripped) and len(stripped.split()) <= 8:
            return self._make_social('ACKNOWLEDGE')

        for pattern in _CANCEL_PATTERNS:
            if pattern.search(stripped):
                return self._make_social('CANCEL')

        for pattern in _SELF_RESOLVED_PATTERNS:
            if pattern.search(stripped):
                return self._make_social('IGNORE')

        return None

    def _make_social(self, mode: str) -> TriageResult:
        """Create a social filter result."""
        return TriageResult(
            branch='social',
            mode=mode,
            tools=[],
            skills=[],
            confidence_internal=1.0,
            confidence_tool_need=0.0,
            freshness_risk=0.0,
            decision_entropy=0.0,
            reasoning=f'social_filter_{mode.lower()}',
            triage_time_ms=0.0,
            fast_filtered=True,
            self_eval_override=False,
            self_eval_reason='',
        )

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
            )

            llm = self._get_llm()
            response_text = llm.send_message("", prompt).text
            data = json.loads(response_text)

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

            logger.info(f"{LOG_PREFIX} mode={mode} tools={[t for t in tools if isinstance(t, str)]} skills={skills}")

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
            )

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM triage failed ({type(e).__name__}: {e}), using heuristic fallback")
            return self._heuristic_fallback(text, context)

    def _heuristic_fallback(self, text: str, context: TriageContext) -> TriageResult:
        """Deterministic fallback when LLM times out or fails."""
        lower = text.lower()
        looks_like_command = any(
            w in lower for w in ['search', 'find', 'check', 'look up', 'look it up', 'get me', 'fetch',
                                  'just look', 'pull up', 'google', 'browse']
        )
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

        # Rule 1: ACT without any tools → innate skill if matched, otherwise assign default tool or downgrade
        if result.branch == 'act' and not result.tools:
            if _INNATE_ACTION_PATTERNS.search(text):
                # Innate action skill (schedule/list) — no external tool needed, keep ACT
                # Ensure primitives are present (may arrive here from heuristic_fallback)
                if not result.skills:
                    result.skills = list(_PRIMITIVES)
                result.self_eval_override = True
                result.self_eval_reason = 'act_innate_skill'
            else:
                default_tool = self._pick_default_tool(ctx.tool_summaries)
                if default_tool:
                    result.tools = [default_tool]
                    result.self_eval_override = True
                    result.self_eval_reason = 'act_default_tool_assigned'
                else:
                    result.branch = 'respond'
                    result.mode = 'RESPOND'
                    result.self_eval_override = True
                    result.self_eval_reason = 'act_without_tools'

        # Rule 1b: RESPOND but text is clearly an innate action request → escalate to ACT
        if result.branch == 'respond' and _INNATE_ACTION_PATTERNS.search(text):
            result.branch = 'act'
            result.mode = 'ACT'
            result.tools = []
            result.self_eval_override = True
            result.self_eval_reason = 'act_innate_skill_failsafe'
            # Ensure primitives + detect likely contextual skill from keyword match
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            lower = text.lower()
            if not any(s for s in result.skills if s not in _PRIMITIVES):
                if any(w in lower for w in ['remind', 'schedule', 'alarm', 'every morning', 'every day']):
                    result.skills.append('schedule')
                elif any(w in lower for w in ['add to', 'remove from', 'my list', 'shopping']):
                    result.skills.append('list')

        # Rule 2: RESPOND on high-freshness question → escalate to ACT
        # Freshness risk alone is sufficient — if the answer requires live data, use tools.
        if (result.branch == 'respond'
                and result.freshness_risk > 0.5
                and _is_factual_question(text)
                and ctx.tool_summaries):
            result.branch = 'act'
            result.mode = 'ACT'
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            result.self_eval_override = True
            result.self_eval_reason = 'act_failsafe'

        # Rule 3: Social filter missed a substantive question
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

        return result

    def _pick_default_tool(self, tool_summaries: str) -> str:
        """Pick the first available tool from summaries (format: "- tool_name: ...")."""
        if not tool_summaries:
            return ''
        import re
        m = re.search(r'-\s+(\w+):', tool_summaries)
        return m.group(1) if m else ''

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
