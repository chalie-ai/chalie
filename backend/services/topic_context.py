"""
TopicContext — single source of truth for topic/thread identity during context assembly.

Replaces loose ``topic: str`` and ``thread_id: str`` parameters with a structured
object that tracks failures, carries classification metadata, and provides
a consistent identifier for working memory lookup.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional


@dataclass
class TopicContext:
    """Structured context for a single topic/thread during assembly."""

    topic: str
    thread_id: Optional[str] = None
    confidence: float = 1.0
    is_new_topic: bool = False
    is_boundary: bool = False
    boundary_trigger: str = ""
    message_embedding: Any = None
    just_reset_from_silence: bool = False
    _failed_sections: List[Tuple[str, str]] = field(default_factory=list, repr=False)

    def record_failure(self, section: str, error: Exception) -> None:
        """Record a context assembly section failure for observability."""
        self._failed_sections.append((section, str(error)))

    @property
    def failed_sections(self) -> List[Tuple[str, str]]:
        """Return a copy of the failed sections list."""
        return list(self._failed_sections)

    @property
    def wm_identifier(self) -> str:
        """Working memory lookup key — thread_id takes precedence over topic."""
        return self.thread_id if self.thread_id else self.topic

    @classmethod
    def from_classification(cls, result: dict, thread_id: str = None) -> 'TopicContext':
        """Create from a topic_classifier.classify() result dict."""
        diagnostics = result.get('boundary_diagnostics') or {}
        return cls(
            topic=result.get('topic', ''),
            thread_id=thread_id,
            confidence=result.get('confidence', 1.0),
            is_new_topic=result.get('is_new_topic', False),
            is_boundary=bool(diagnostics.get('triggered', False)),
            boundary_trigger=diagnostics.get('trigger', ''),
            message_embedding=result.get('message_embedding'),
            just_reset_from_silence=result.get('just_reset_from_silence', False),
        )
