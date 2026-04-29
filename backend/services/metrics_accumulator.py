import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class MetricsAccumulator:
    """Per-turn token, tool, and timing accounting for one MessageProcessor.send().

    Started at processor entry (before the thinking gate) so exploration and
    compaction LLM calls are captured. Ends when snapshot() is called at
    dispatch time. All LLM calls in the turn contribute via accumulate();
    every handleTool() call contributes via record_tool().

    Timing model
    ------------
    Three turn-level buckets, each in milliseconds:
      * ``llm_ms``   — sum of every LLMResponse.latency_ms recorded via
        ``record_llm_call`` across the turn (ACT iterations + thinking
        exploration + compaction).
      * ``pre_llm_ms`` — sum of all pre-LLM stages, accumulated via
        ``stage(name)`` context manager. Stages reported as a sub-dict so
        the worst offender is identifiable without parsing.
      * ``post_llm_ms`` — derived at snapshot time as
        ``wall - pre_llm_ms - llm_ms``. Captures everything not in the
        other two buckets (post-tool records, store, postTurn, etc.).

    Per-iteration breakdown is captured by wrapping each ACT loop body in
    ``iteration()``. The currently-open iteration is the bucket that
    ``record_llm_call`` and the iteration-scoped stages append to.
    """

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_thinking: int = 0
    tokens_cache_read: int = 0
    tokens_cache_create: int = 0
    tool_counts: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    tokens_total_complete: bool = True

    # Timing
    llm_ms: int = 0
    llm_calls: int = 0
    stages: dict = field(default_factory=dict)
    iterations: list = field(default_factory=list)
    _current_iter: dict = field(default=None, repr=False)

    # ── Token / tool accumulation (existing) ────────────────────────────────

    def accumulate(self, response) -> None:
        if response is None:
            return
        in_ = getattr(response, 'tokens_input', None)
        out_ = getattr(response, 'tokens_output', None)
        if in_ is None or out_ is None:
            self.tokens_total_complete = False
        for attr in ('tokens_input', 'tokens_output', 'tokens_thinking',
                     'tokens_cache_read', 'tokens_cache_create'):
            v = getattr(response, attr, None)
            if v is not None:
                setattr(self, attr, getattr(self, attr) + int(v))

    def record_tool(self, tool_name: str) -> None:
        if not tool_name:
            return
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        if self._current_iter is not None:
            tools = self._current_iter.setdefault('tools', [])
            tools.append({'name': tool_name})

    def merge(self, other: 'MetricsAccumulator') -> None:
        """Fold a sub-processor's counts into this one.

        Used when a sub-processor (compaction, exploration) runs its own
        send() inside the parent turn — token/tool counts and LLM time
        are absorbed so per-turn metrics reflect the full cost.
        Sub-processor stages and iterations are NOT merged: the parent
        owns the timeline.
        """
        if other is None:
            return
        for attr in ('tokens_input', 'tokens_output', 'tokens_thinking',
                     'tokens_cache_read', 'tokens_cache_create'):
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        for tool_name, count in other.tool_counts.items():
            self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + count
        if not other.tokens_total_complete:
            self.tokens_total_complete = False
        # Absorb LLM time so the parent's llm_ms reflects child calls too.
        self.llm_ms += other.llm_ms
        self.llm_calls += other.llm_calls

    # ── Timing API ──────────────────────────────────────────────────────────

    @contextmanager
    def stage(self, name: str):
        """Time a named pre-LLM (or post-LLM) stage.

        Adds the elapsed ms to ``stages[name]`` (cumulative across calls
        with the same name within the turn). If invoked while an
        iteration is active, the time is also added to that iteration's
        per-stage breakdown so per-iter ACT cost is attributable.

        Never raises — timing must not break the request path.
        """
        t0 = time.monotonic()
        try:
            yield
        finally:
            try:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                self.stages[name] = self.stages.get(name, 0) + elapsed_ms
                if self._current_iter is not None:
                    iter_stages = self._current_iter.setdefault('stages', {})
                    iter_stages[name] = iter_stages.get(name, 0) + elapsed_ms
            except Exception:
                pass

    def add_stage_ms(self, name: str, ms: int) -> None:
        """Record a stage measured externally (e.g. file-tags wait in WS)."""
        if ms is None or ms < 0:
            return
        try:
            self.stages[name] = self.stages.get(name, 0) + int(ms)
        except Exception:
            pass

    @contextmanager
    def iteration(self, iter_idx: int):
        """Open a per-ACT-iteration bucket. Tools / LLM calls / stages
        recorded while this is active are attributed to ``iter_idx``."""
        bucket = {
            'iter': iter_idx,
            'llm_ms': 0,
            'llm_calls': 0,
            'tools': [],
            'stages': {},
            'wall_ms': 0,
        }
        t0 = time.monotonic()
        prev = self._current_iter
        self._current_iter = bucket
        try:
            yield bucket
        finally:
            try:
                bucket['wall_ms'] = int((time.monotonic() - t0) * 1000)
                self.iterations.append(bucket)
            finally:
                self._current_iter = prev

    def record_llm_call(self, latency_ms: int) -> None:
        """Add an LLM round-trip to the turn-level + per-iteration totals."""
        if latency_ms is None or latency_ms < 0:
            return
        try:
            ms = int(latency_ms)
        except (TypeError, ValueError):
            return
        self.llm_ms += ms
        self.llm_calls += 1
        if self._current_iter is not None:
            self._current_iter['llm_ms'] += ms
            self._current_iter['llm_calls'] += 1

    # ── Snapshot ────────────────────────────────────────────────────────────

    def snapshot(self, end_time: float = None) -> dict:
        end = end_time if end_time is not None else time.time()
        wall_ms = max(0, int((end - self.start_time) * 1000))
        total = (self.tokens_input + self.tokens_output + self.tokens_thinking
                 + self.tokens_cache_read + self.tokens_cache_create)
        pre_llm_ms = sum(self.stages.values())
        post_llm_ms = max(0, wall_ms - pre_llm_ms - self.llm_ms)
        result = {
            'tokens_total': total,
            'tools': dict(self.tool_counts),
            'response_time_s': round(end - self.start_time, 3),
            'timing': {
                'wall_ms': wall_ms,
                'pre_llm_ms': pre_llm_ms,
                'llm_ms': self.llm_ms,
                'post_llm_ms': post_llm_ms,
                'llm_calls': self.llm_calls,
                'stages': dict(self.stages),
                'iterations': list(self.iterations),
            },
        }
        if not self.tokens_total_complete:
            result['tokens_total_complete'] = False
        return result
