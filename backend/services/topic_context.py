"""
TopicContext — single source of truth for topic/thread identity during context assembly.

Replaces loose ``topic: str`` and ``thread_id: str`` parameters with a structured
object that tracks failures, carries embedding metadata, and provides
a consistent identifier for working memory lookup.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional


@dataclass
class TopicContext:
    """Structured context for a single topic/thread during assembly.

    Fields:
        topic: Primary topic/thread key used for transcript and compaction lookup.
        thread_id: Canonical thread identifier; takes precedence over ``topic``
            when constructing the working-memory key.
        confidence: Reserved for caller-supplied confidence score; defaults to 1.0.
        is_new_topic: Whether this message starts a new topic/thread.
        message_embedding: Pre-computed embedding vector for the current message,
            forwarded to downstream services (e.g. episodic memory).
        _failed_sections: Internal list of (section_name, error_str) pairs
            recorded via :meth:`record_failure`.
    """

    topic: str
    thread_id: Optional[str] = None
    confidence: float = 1.0
    is_new_topic: bool = False
    message_embedding: Any = None
    _failed_sections: List[Tuple[str, str]] = field(default_factory=list, repr=False)

    def record_failure(self, section: str, error: Exception) -> None:
        """Record a context assembly section failure for observability.

        Args:
            section: Human-readable name of the assembly section that failed.
            error: The exception that was raised.
        """
        self._failed_sections.append((section, str(error)))

    @property
    def failed_sections(self) -> List[Tuple[str, str]]:
        """Return a copy of the failed sections list.

        Returns:
            List of ``(section_name, error_string)`` tuples for all recorded failures.
        """
        return list(self._failed_sections)

    @property
    def wm_identifier(self) -> str:
        """Working memory lookup key — thread_id takes precedence over topic.

        Returns:
            ``thread_id`` when set, otherwise ``topic``.
        """
        return self.thread_id if self.thread_id else self.topic
