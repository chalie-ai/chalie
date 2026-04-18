import time
from dataclasses import dataclass, field


@dataclass
class MetricsAccumulator:
    """Per-turn token and tool accounting for a single MessageProcessor.send() call.

    Started at processor entry (before the thinking gate) so exploration and
    compaction LLM calls are captured. Ends when snapshot() is called at
    dispatch time. All LLM calls in the turn contribute via accumulate();
    every handleTool() call contributes via record_tool().
    """

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_thinking: int = 0
    tokens_cache_read: int = 0
    tokens_cache_create: int = 0
    tool_counts: dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    tokens_total_complete: bool = True

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
            if v:
                setattr(self, attr, getattr(self, attr) + int(v))

    def record_tool(self, tool_name: str) -> None:
        if not tool_name:
            return
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1

    def snapshot(self, end_time: float = None) -> dict:
        end = end_time if end_time is not None else time.time()
        total = (self.tokens_input + self.tokens_output + self.tokens_thinking
                 + self.tokens_cache_read + self.tokens_cache_create)
        result = {
            'tokens_total': total,
            'tools': dict(self.tool_counts),
            'response_time_s': round(end - self.start_time, 3),
        }
        if not self.tokens_total_complete:
            result['tokens_total_complete'] = False
        return result
