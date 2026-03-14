"""
AmbientToolAction — Proactive external tool invocation from drift thoughts.

Bridges drift thoughts to external tools (Reddit, News, Wikipedia, etc.).
When a drift thought is relevant to an ambient-capable tool, this action
invokes the tool and stores high-signal findings as gists for later use.

Priority: 6 (same as SEED_THREAD — ties broken by score)
Low priority because ambient lookups are speculative — they should not
compete with direct user-facing actions.

Gates:
  1. Phase gate: Spark must be in connected or graduated
  2. Tool gate: At least one ambient tool must exist
  3. Relevance gate: Thought embedding similarity to tool docs > threshold
  4. Rate limit: Max 1 invocation per cooldown per tool
  5. Activation energy >= min_activation
"""

import logging
import math
import time
from typing import Optional, Tuple, List

from services.memory_client import MemoryClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)
LOG_PREFIX = "[AMBIENT_TOOL]"
_NS = "ambient_tool"


def _key(suffix: str) -> str:
    return f"{_NS}:{suffix}"


class AmbientToolAction(AutonomousAction):
    """Proactive ambient tool invocation driven by cognitive drift thoughts.

    Evaluates each drift thought against available ambient-capable tools
    (e.g., Reddit, News, Wikipedia) using embedding similarity.  When a
    thought is sufficiently relevant and all gate conditions are met, the
    matching tool is invoked and its findings are persisted as gists for
    future drift or recall cycles.

    Priority: 6 (same as SEED_THREAD — ties broken by score).  Lower than
    direct user-facing actions because ambient lookups are speculative.
    """

    def __init__(self, config: dict = None):
        """Initialize the ambient tool action with configurable thresholds.

        Args:
            config: Optional configuration dict.  Recognised keys:

                - ``relevance_threshold`` (float, default 0.35): Minimum
                  cosine similarity between a drift thought embedding and a
                  tool's documentation to pass the relevance gate.
                - ``min_activation`` (float, default 0.5): Minimum activation
                  energy on the ``ThoughtContext`` required to proceed.
                - ``per_tool_cooldown`` (int, default 43200): Seconds between
                  consecutive invocations of the same tool (12 h).
                - ``signal_threshold`` (float, default 0.4): Minimum signal
                  score to store a finding gist.
                - ``llm_timeout`` (float, default 8.0): Seconds to wait for
                  the query-generation LLM call.
                - ``surface_high_signal`` (bool, default False): When True,
                  high-signal findings are surfaced to the user in addition
                  to being stored as gists.
        """
        # Disabled: ambient tool invocation from drift produced low-quality content
        # (world events, places) on weak similarity matching. Re-enable when
        # signal-driven reasoning provides better context.
        super().__init__(name='AMBIENT_TOOL', enabled=False, priority=6)
        config = config or {}
        self.store = MemoryClientService.create_connection()

        self.relevance_threshold = config.get('relevance_threshold', 0.35)
        self.min_activation = config.get('min_activation', 0.5)
        self.per_tool_cooldown = config.get('per_tool_cooldown', 43200)  # 12h
        self.signal_threshold = config.get('signal_threshold', 0.4)
        self.llm_timeout = config.get('llm_timeout', 8.0)
        self.surface_high_signal = config.get('surface_high_signal', False)

        self._embedding_service = None
        self._tool_cache = None
        self._tool_cache_ts = 0
        self._pending_tool = None
        self._pending_relevance = 0.0

    @property
    def embedding_service(self):
        """Lazily-initialised shared :class:`EmbeddingService` instance.

        The service is imported and constructed on first access so that the
        heavy sentence-transformer model is not loaded until actually needed.

        Returns:
            The singleton embedding service obtained via
            :func:`~services.embedding_service.get_embedding_service`.
        """
        if self._embedding_service is None:
            from services.embedding_service import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def _get_ambient_tools(self) -> List[dict]:
        """Cached ambient tool lookup (refresh every 5 min)."""
        now = time.time()
        if self._tool_cache is None or (now - self._tool_cache_ts) > 300:
            try:
                from services.tool_registry_service import ToolRegistryService
                self._tool_cache = ToolRegistryService().get_ambient_tools()
                self._tool_cache_ts = now
            except Exception:
                self._tool_cache = []
        return self._tool_cache or []

    # ── Gates ──────────────────────────────────────────────────

    def _phase_gate(self) -> bool:
        """Same as SuggestAction — connected or graduated."""
        try:
            from services.spark_state_service import SparkStateService
            phase = SparkStateService().get_phase()
            return phase in ('connected', 'graduated')
        except Exception:
            return False

    def _tool_gate(self) -> Tuple[bool, List[dict]]:
        """At least one ambient tool must exist."""
        tools = self._get_ambient_tools()
        return (len(tools) > 0, tools)

    def _relevance_gate(
        self, thought: ThoughtContext, tools: List[dict]
    ) -> Tuple[bool, Optional[dict], float]:
        """Score thought embedding vs each tool's documentation."""
        if not thought.thought_embedding:
            return (False, None, 0.0)

        best_tool = None
        best_sim = 0.0

        for tool in tools:
            doc_text = tool['manifest'].get('documentation', '')
            if not doc_text:
                continue
            try:
                doc_emb = self.embedding_service.generate_embedding(doc_text[:500])
                sim = self._cosine_similarity(thought.thought_embedding, doc_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_tool = tool
            except Exception:
                continue

        if best_sim < self.relevance_threshold or not best_tool:
            return (False, None, best_sim)
        return (True, best_tool, best_sim)

    def _rate_limit_gate(self, tool_name: str) -> bool:
        """Max 1 invocation per cooldown period per tool."""
        last_key = _key(f'last_invoke:{tool_name}')
        last = self.store.get(last_key)
        if last and (time.time() - float(last)) < self.per_tool_cooldown:
            return False
        return True

    # ── Main interface ─────────────────────────────────────────

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Evaluate whether an ambient tool invocation should be scheduled.

        Runs five sequential gates (phase, tool availability, embedding
        relevance, per-tool rate limit, and activation energy).  If all
        gates pass, the best matching tool and its relevance score are cached
        on the instance for the subsequent :meth:`execute` call.

        Args:
            thought: The current drift thought context, including its
                embedding, activation energy, and seed topic.

        Returns:
            A ``(score, eligible)`` tuple.  ``score`` is a float in [0, 1]
            representing action priority (``relevance * 0.5``).  ``eligible``
            is ``True`` only when every gate passes.  Returns ``(0.0, False)``
            whenever any gate fails.
        """
        self.last_gate_result = None

        # Gate 1: Phase
        if not self._phase_gate():
            self.last_gate_result = {'gate': 'phase', 'reason': 'not in connected/graduated phase'}
            return (0.0, False)

        # Gate 2: Tools exist
        tool_passes, tools = self._tool_gate()
        if not tool_passes:
            self.last_gate_result = {'gate': 'tools', 'reason': 'no ambient tools available'}
            return (0.0, False)

        # Gate 3: Relevance
        rel_passes, best_tool, rel_score = self._relevance_gate(thought, tools)
        if not rel_passes:
            self.last_gate_result = {'gate': 'relevance', 'reason': 'thought not relevant to any tool'}
            return (0.0, False)

        # Gate 4: Rate limit
        if not self._rate_limit_gate(best_tool['name']):
            self.last_gate_result = {'gate': 'rate_limit', 'reason': f"tool '{best_tool['name']}' on cooldown"}
            return (0.0, False)

        # Gate 5: Activation energy
        if thought.activation_energy < self.min_activation:
            self.last_gate_result = {'gate': 'activation_energy', 'reason': f"energy {thought.activation_energy:.2f} < {self.min_activation}"}
            return (0.0, False)

        # Store for execute()
        self._pending_tool = best_tool
        self._pending_relevance = rel_score

        # Score: modest, this is speculative
        score = rel_score * 0.5
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Invoke the pending ambient tool and persist findings as a gist.

        Expects :meth:`should_execute` to have been called first so that
        ``_pending_tool`` is populated.  The method:

        1. Generates a focused search query from the drift thought via a
           short-timeout LLM call.
        2. Invokes the tool through :class:`~services.tool_registry_service.ToolRegistryService`.
        3. Updates the per-tool rate-limit timestamp in the memory store.
        4. Stores the raw finding as a gist for future drift/recall.

        Args:
            thought: The drift thought context that triggered this action.

        Returns:
            An :class:`~services.autonomous_actions.base.ActionResult` with
            ``action_name='AMBIENT_TOOL'``.  ``success`` is ``True`` when the
            tool was invoked without error; ``details`` always contains at
            minimum a ``'reason'`` key on failure.
        """
        tool = self._pending_tool
        if not tool:
            return ActionResult(action_name='AMBIENT_TOOL', success=False,
                                details={'reason': 'no_pending_tool'})

        tool_name = tool['name']

        # 1. Generate search query from thought
        query = self._generate_query(thought, tool)
        if not query:
            return ActionResult(action_name='AMBIENT_TOOL', success=False,
                                details={'reason': 'query_generation_failed'})

        # 2. Invoke tool
        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()

            result_text = registry.invoke(tool_name, thought.seed_topic, {'query': query})
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Tool invocation failed: {e}")
            return ActionResult(action_name='AMBIENT_TOOL', success=False,
                                details={'reason': 'invocation_failed', 'error': str(e)[:100]})

        # 3. Update rate limit
        self.store.set(_key(f'last_invoke:{tool_name}'), str(time.time()))

        # 4. Store finding in working memory for context continuity
        self._store_finding_in_wm(thought, tool_name, query, result_text)

        logger.info(f"{LOG_PREFIX} Invoked {tool_name} for '{query}' (relevance={self._pending_relevance:.2f})")

        return ActionResult(
            action_name='AMBIENT_TOOL',
            success=True,
            details={
                'tool': tool_name,
                'query': query,
                'relevance': self._pending_relevance,
            },
        )

    # ── Helpers ────────────────────────────────────────────────

    def _generate_query(self, thought: ThoughtContext, tool: dict) -> Optional[str]:
        """Use lightweight LLM to extract a search query from the drift thought."""
        import threading

        result_holder: list = [None]
        done = threading.Event()

        def _generate():
            try:
                from services.config_service import ConfigService
                from services.llm_service import create_llm_service

                try:
                    config = ConfigService.resolve_agent_config("autonomous-ambient-tool")
                except Exception:
                    config = ConfigService.resolve_agent_config("frontal-cortex")

                config = dict(config)
                config['format'] = ''

                prompt = ConfigService.get_agent_prompt("ambient-tool-query")
                prompt = prompt.replace('{{thought_content}}', thought.thought_content)
                prompt = prompt.replace('{{seed_topic}}', thought.seed_topic or '')
                prompt = prompt.replace('{{tool_name}}', tool['name'])
                prompt = prompt.replace('{{tool_description}}', tool['manifest'].get('description', ''))

                llm = create_llm_service(config)
                response = llm.send_message(prompt, "Generate a search query.").text
                text = response.strip().strip('"').strip("'").strip()

                if text and 3 < len(text) < 200:
                    result_holder[0] = text
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Query generation failed: {e}")
            finally:
                done.set()

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        done.wait(timeout=self.llm_timeout)
        return result_holder[0]

    def _store_finding_in_wm(self, thought: ThoughtContext, tool_name: str, query: str, result_text: str) -> None:
        """Store ambient tool finding in working memory for context continuity."""
        try:
            from services.working_memory_service import WorkingMemoryService
            topic = thought.seed_topic or 'ambient_discovery'
            wm = WorkingMemoryService()
            wm.append_turn(topic, 'system', f"[ambient/{tool_name}] {query}: {result_text[:500]}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} WM storage failed: {e}")

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
