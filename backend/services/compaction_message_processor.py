"""CompactionMessageProcessor — LLM-call wrappers for channel + subagent compaction.

Three subclasses, all sharing ``_CompactionProcessorBase``:

  ContinuityCompactionProcessor
      UMP channel checkpoint summarisation. Replaces ``FullCompactionProcessor``.
      Produces ``<summary>`` body parsed by ``_extract_compaction_summary``.

  SubagentTrailCompactionProcessor
      Subagent mid-ACT tool-trail summarisation. Replaces both the old
      ``TrailCompactionProcessor`` (UMP variant) and is the canonical path for
      subagents. Output is a dense per-tool block injected inline; no channel
      summary, no ``<summary>`` tags.

Both share ``_CompactionProcessorBase``:
  - LOG_LABEL = 'compaction' (labels LLM calls for observability)
  - ALWAYS_AVAILABLE / DISCOVERABLE = [] (compaction never calls tools)
  - SKIP_TRANSCRIPT_WRITE = True   (no transcript pollution)
  - _check_threshold returns False (compaction never compacts itself — recursion guard)
  - get_user_definition returns '' (no human user)
  - get_user_prompt returns self._raw_input (caller pre-formats the input)

Callers:
  MessageProcessor._run_full_compaction → ContinuityCompactionProcessor
  SubagentProcessor._handle_overflow    → SubagentTrailCompactionProcessor
"""

from services.message_processor import MessageProcessor
from services.system_message_prompt import (
    ContinuityCompactionSystemPrompt,
    SubagentTrailCompactionSystemPrompt,
)


class _CompactionProcessorBase(MessageProcessor):
    LOG_LABEL = 'compaction'
    ALWAYS_AVAILABLE: list[str] = []
    DISCOVERABLE: list[str] = []
    SKIP_TRANSCRIPT_WRITE = True

    def get_user_definition(self) -> str:
        return ''

    def get_user_prompt(self) -> str:
        return self._raw_input

    def get_system_prompt(self) -> str:
        # Compaction has no human user — skip the empty user-definition prefix
        # that the base would otherwise stitch on with a leading '\n\n'.
        return self.SYSTEM_PROMPT_CLASS().get_prompt()

    def _check_threshold(self, _system_prompt: str, _tools: list, _user_body: str) -> bool:
        # Compaction never compacts itself. Hard-disable the gate so a large
        # input (which is the common case) cannot trigger recursion.
        return False


class ContinuityCompactionProcessor(_CompactionProcessorBase):
    """Full channel compaction — continuity-first summary producing <summary> tags."""

    CHANNEL = 'compaction'
    ROLE = 'compaction'
    SYSTEM_PROMPT_CLASS = ContinuityCompactionSystemPrompt


class SubagentTrailCompactionProcessor(_CompactionProcessorBase):
    """Subagent mid-ACT trail compaction — dense per-tool blocks, no channel summary."""

    CHANNEL = 'subagent_compaction'
    ROLE = 'compaction'
    SYSTEM_PROMPT_CLASS = SubagentTrailCompactionSystemPrompt
