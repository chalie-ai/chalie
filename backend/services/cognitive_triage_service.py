"""
Cognitive Triage Service — LLM-based routing with self-eval guardrails.

3-step branching flow:
  1. Empty-input guard (~0ms)
  2. LLM cognitive triage (~100-300ms, lightweight model)
  3. Self-eval sanity check (~0ms, deterministic rules)

The triage LLM reasons about the prompt with context and tool summaries,
returning a structured decision. The self-eval applies deterministic
guardrails to catch obvious errors.

Timeout fallback: if LLM fails, falls back to ONNX classification.
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
_VALID_SKILLS = {
    'recall', 'memorize', 'introspect', 'associate', 'schedule', 'list',
    'focus', 'autobiography', 'persistent_task', 'document', 'read', 'reflect',
    'needs_external_tool',
}
_CONTEXTUAL_SKILLS = _VALID_SKILLS - set(_PRIMITIVES) - {'needs_external_tool'}
MAX_CONTEXTUAL_SKILLS = 3   # caps contextual skills from LLM; ONNX predictions bypass this cap

_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\'\)]+',
    re.IGNORECASE
)


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


def _contains_url(text: str) -> bool:
    """Detect presence of an HTTP(S) URL in the message."""
    return bool(_URL_PATTERN.search(text))


@dataclass
class TriageContext:
    """Input context supplied to the triage LLM for routing decisions.

    Attributes:
        context_warmth: Float 0-1 measuring how active the current conversation is.
        memory_confidence: Float 0-1 confidence that internal memory is sufficient.
        working_memory_turns: Number of turns currently held in working memory.
        gist_count: Number of gist summaries available for this topic.
        fact_count: Number of stored facts available for this topic.
        previous_mode: The routing mode used in the prior turn (e.g. 'RESPOND').
        previous_tools: Tool names selected in the prior ACT turn.
        tool_summaries: Grouped human-readable summary of available tools.
        working_memory_summary: Text summary of recent working memory exchanges.
    """

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
    """Output of the cognitive triage pipeline.

    Attributes:
        branch: High-level dispatch branch: 'ignore' | 'respond' | 'clarify' | 'act'.
        mode: Routing mode: 'RESPOND' | 'CLARIFY' | 'ACT' | 'IGNORE' | 'CANCEL'.
        tools: Tool names selected for ACT mode (empty for other modes).
        skills: Innate skill names selected for ACT mode.
        confidence_internal: Float 0-1 confidence that internal memory is sufficient.
        confidence_tool_need: Float 0-1 confidence that external tools are required.
        triage_time_ms: Wall-clock milliseconds taken for the full triage pipeline.
        fast_filtered: True if the result came from the empty-input fast path (no LLM called).
        self_eval_override: True if a self-eval rule overrode the LLM decision.
        self_eval_reason: Short label for which self-eval rule fired.
        effort_estimate: Effort tier: 'trivial' | 'light' | 'moderate' | 'deep'.
    """

    branch: str                   # 'ignore' | 'respond' | 'clarify' | 'act'
    mode: str                     # 'RESPOND' | 'CLARIFY' | 'ACT' | 'IGNORE' | 'CANCEL'
    tools: List[str]              # tool names, only meaningful for ACT
    skills: List[str]             # innate skill names selected for ACT
    confidence_internal: float    # 0-1: confidence memory is sufficient
    confidence_tool_need: float   # 0-1: confidence external tools required
    triage_time_ms: float
    fast_filtered: bool           # True if empty-input path, no LLM
    self_eval_override: bool
    self_eval_reason: str
    effort_estimate: str = 'moderate'  # trivial | light | moderate | deep


class CognitiveTriageService:
    """3-step branching triage: empty guard → LLM → self-eval → dispatch."""

    def triage(self, text: str, context: TriageContext) -> TriageResult:
        """Main entry point. Empty guard → LLM triage → ONNX skill override → self-eval."""
        start = time.time()

        # 1. Empty-input guard (~0ms)
        if not text.strip():
            result = TriageResult(
                branch='ignore', mode='IGNORE', tools=[], skills=[],
                confidence_internal=1.0, confidence_tool_need=0.0,
                triage_time_ms=(time.time() - start) * 1000,
                fast_filtered=True, self_eval_override=False, self_eval_reason='',
                effort_estimate='trivial',
            )
            return result

        # 2. LLM cognitive triage (~100-300ms)
        result = self._cognitive_triage(text, context)

        # 2b. ONNX skill selector (~5ms) — override LLM skill selection when available
        llm_skills = list(result.skills)
        result = self._apply_onnx_skills(result, text, llm_skills)

        # 3. Self-eval sanity check (~0ms, deterministic)
        original_mode = result.mode
        result = self._self_evaluate(result, text, context)

        # Log self-eval overrides to interaction_log for constraint learning
        if result.self_eval_override:
            self._log_self_eval_override(result, original_mode, text)

        result.triage_time_ms = (time.time() - start) * 1000
        return result



    def _cognitive_triage(self, text: str, context: TriageContext) -> TriageResult:
        """Call the LLM with the cognitive-triage prompt and parse the response.

        Falls back to ONNX classification if the LLM times out or returns invalid JSON.

        Args:
            text: The raw user message text.
            context: TriageContext carrying memory signals and tool summaries.

        Returns:
            TriageResult parsed from the LLM response, or an ONNX fallback.
        """
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
            # freshness_risk is parsed for prompt compliance but not stored in TriageResult

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
                triage_time_ms=0.0,
                fast_filtered=False,
                self_eval_override=False,
                self_eval_reason='',
                effort_estimate=effort_estimate,
            )

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM triage failed ({type(e).__name__}: {e}), using ONNX fallback")
            return self._onnx_fallback(text, context)

    def _onnx_fallback(self, text: str, context: TriageContext) -> TriageResult:
        """Build a TriageResult from ONNX mode-tiebreaker when the LLM triage fails.

        Only determines the mode. Skill selection is left to _apply_onnx_skills()
        which runs immediately after in the triage() pipeline — no double classification.
        Falls back to RESPOND if ONNX is unavailable.

        Args:
            text: The raw user message text.
            context: TriageContext carrying tool summaries and conversation state.

        Returns:
            TriageResult with ONNX-predicted mode, skills populated by downstream step.
        """
        mode = 'RESPOND'
        reasoning = 'onnx_fallback'

        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            if svc.is_available("mode-tiebreaker"):
                label, confidence = svc.predict("mode-tiebreaker", text)
                if label is not None:
                    mode = label.upper()
                    if mode not in ('RESPOND', 'ACT', 'CLARIFY', 'IGNORE'):
                        mode = 'RESPOND'
                    reasoning = f'onnx_fallback_mode={mode}({confidence:.2f})'
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ONNX fallback failed: {e}")

        branch = self._mode_to_branch(mode)
        skills = list(_PRIMITIVES) if mode == 'ACT' else []

        return TriageResult(
            branch=branch,
            mode=mode,
            tools=[],
            skills=skills,
            confidence_internal=0.5,
            confidence_tool_need=0.5 if mode == 'ACT' else 0.2,
            triage_time_ms=0.0,
            fast_filtered=True,
            self_eval_override=False,
            self_eval_reason='',
        )

    def _mode_to_branch(self, mode: str) -> str:
        """Map an LLM mode string to the corresponding dispatch branch name.

        Args:
            mode: Uppercase mode string from the LLM response (e.g. 'ACT').

        Returns:
            Lowercase branch name string (e.g. 'act'), defaulting to 'respond'.
        """
        return {
            'ACT': 'act',
            'RESPOND': 'respond',
            'CLARIFY': 'clarify',
            'IGNORE': 'ignore',
            'CANCEL': 'ignore',
        }.get(mode, 'respond')

    def _apply_onnx_skills(self, result: TriageResult, text: str, llm_skills: list) -> TriageResult:
        """Replace LLM skill selection with ONNX predictions when available.

        The ONNX skill-selector is a multi-label classifier trained on 13 skills.
        It runs in ~5ms vs ~100-300ms for the LLM triage. When available, its
        predictions replace the LLM's skill[] selection for ACT mode routing.

        The LLM triage still determines mode/branch/tools/confidence — the ONNX
        model only handles skill selection.

        Shadow logging: both LLM and ONNX skill predictions are logged for
        comparison to monitor alignment.
        """
        if result.branch != 'act':
            return result

        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            if not svc.is_available("skill-selector"):
                return result

            # Build input in the same format as training data
            input_text = f"{text}\nSkills:"
            onnx_skills = svc.predict_multi_label("skill-selector", input_text)

            if not onnx_skills:
                return result

            # Extract skill names (predict_multi_label returns [(label, confidence), ...])
            onnx_skill_names = [s for s, _ in onnx_skills]

            # Shadow log: compare LLM vs ONNX skill predictions
            self._log_skill_shadow(text, llm_skills, onnx_skill_names, onnx_skills)

            # Build final skill list: ensure cognitive primitives, then ONNX contextual skills
            # ONNX predictions are NOT capped at MAX_CONTEXTUAL_SKILLS — the model was
            # trained to predict the full set independently
            primitives = [p for p in _PRIMITIVES if p not in onnx_skill_names]
            result.skills = primitives + onnx_skill_names

            # If ONNX predicts needs_external_tool, ensure we don't strip tools
            if 'needs_external_tool' in onnx_skill_names:
                result.skills = [s for s in result.skills if s != 'needs_external_tool']
                # Boost tool confidence so self-eval doesn't downgrade
                result.confidence_tool_need = max(result.confidence_tool_need, 0.6)

            logger.debug(
                f"{LOG_PREFIX} ONNX skills: {onnx_skill_names} "
                f"(replaced LLM: {llm_skills})"
            )

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} ONNX skill selector unavailable: {e}")

        return result

    @staticmethod
    def _log_skill_shadow(text: str, llm_skills: list, onnx_skills: list, onnx_raw: list):
        """Log LLM vs ONNX skill predictions for shadow comparison."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            InteractionLogService(db).log_event(
                event_type='skill_selector_shadow',
                payload={
                    'message_preview': text[:100],
                    'llm_skills': llm_skills,
                    'onnx_skills': onnx_skills,
                    'onnx_confidences': {s: round(c, 3) for s, c in onnx_raw},
                    'match': set(llm_skills) == set(onnx_skills),
                },
                source='cognitive_triage',
            )
        except Exception:
            pass

    def _self_evaluate(self, result: TriageResult, text: str, ctx: TriageContext) -> TriageResult:
        """Apply deterministic guard-rail rules to catch obvious LLM routing errors.

        Runs in ~0ms. Mutates and returns the result with override flags set when
        a rule fires.

        Args:
            result: TriageResult from the LLM (or ONNX fallback).
            text: The raw user message text.
            ctx: TriageContext for tool availability and conversation state.

        Returns:
            Potentially-modified TriageResult with self_eval_override/reason set.
        """

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
                # Recover via ONNX skill-selector if available.
                _recovered_skill = None
                try:
                    from services.onnx_inference_service import get_onnx_inference_service
                    _svc = get_onnx_inference_service()
                    if _svc.is_available("skill-selector"):
                        _onnx_preds = _svc.predict_multi_label("skill-selector", f"{text}\nSkills:")
                        if _onnx_preds:
                            for _s, _ in _onnx_preds:
                                if _s in _CONTEXTUAL_SKILLS:
                                    _recovered_skill = _s
                                    break
                except Exception:
                    pass
                if _recovered_skill:
                    if _recovered_skill not in result.skills:
                        result.skills.append(_recovered_skill)
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

        # Rule 5: URL in message → escalate to ACT
        if result.branch == 'respond' and _contains_url(text) and ctx.tool_summaries:
            result.branch = 'act'
            result.mode = 'ACT'
            if not result.skills:
                result.skills = list(_PRIMITIVES)
            result.self_eval_override = True
            result.self_eval_reason = 'act_url_detected'

        # Rule 3: LLM classified as ignore but message has a substantive question
        # CANCEL is protected (user explicitly cancelled); IGNORE with a real question gets upgraded
        if result.branch == 'ignore' and result.mode != 'CANCEL' and '?' in text and len(text.split()) > 3:
            result.branch = 'respond'
            result.mode = 'RESPOND'
            result.self_eval_override = True
            result.self_eval_reason = 'ignore_with_question'

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
    def _log_self_eval_override(result, original_mode: str, text: str):
        """Log triage self-eval override to interaction_log for constraint learning."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            InteractionLogService(db).log_event(
                event_type='triage_override',
                payload={
                    'rule': result.self_eval_reason,
                    'original_mode': original_mode,
                    'final_mode': result.mode,
                    'message_preview': text[:100],
                },
                source='cognitive_triage',
            )
        except Exception:
            pass

    @staticmethod
    def _get_active_tasks_summary() -> str:
        """Retrieve a formatted summary of active persistent tasks for triage context.

        Returns:
            Newline-separated list of active task lines, or 'None' if empty or on error.
        """
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
        """Load the cognitive-triage.md prompt template from disk.

        Returns:
            Full contents of the prompt file as a string.
        """
        import os
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
        path = os.path.join(prompts_dir, 'cognitive-triage.md')
        with open(path, 'r') as f:
            return f.read()

    def _get_llm(self):
        """Instantiate and return the LLM service configured for cognitive triage.

        Returns:
            LLM service instance backed by the 'cognitive-triage' agent config.
        """
        from services.llm_service import create_llm_service
        from services.config_service import ConfigService
        agent_cfg = ConfigService.resolve_agent_config('cognitive-triage')
        return create_llm_service(agent_cfg)
