"""
Workers package — Background queue-consumer and scheduled worker callables.

Each worker module exposes a single callable that is registered with
``PromptQueue`` or launched as a long-running background process by the
service runtime.  This package re-exports the primary worker functions for
convenient import by the consumer and service layers.

Exported workers:
    - ``digest_worker``: Main LLM response pipeline (prompt-queue consumer).
    - ``memory_chunker_worker``: Post-exchange memory extraction pipeline.
    - ``episodic_memory_worker``: Episodic memory consolidation pipeline.
    - ``semantic_consolidation_worker``: Semantic vector consolidation pipeline.
    - ``rest_api_worker``: REST API request handler worker.
    - ``tool_worker``: On-demand tool execution worker.
"""

from .digest_worker import digest_worker
from .memory_chunker_worker import memory_chunker_worker
from .episodic_memory_worker import episodic_memory_worker
from .semantic_consolidation_worker import semantic_consolidation_worker
from .rest_api_worker import rest_api_worker
from .tool_worker import tool_worker

__all__ = ['digest_worker', 'memory_chunker_worker', 'episodic_memory_worker', 'semantic_consolidation_worker', 'rest_api_worker', 'tool_worker']
