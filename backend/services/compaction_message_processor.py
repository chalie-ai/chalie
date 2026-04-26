"""CompactionMessageProcessor — unified LLM-call wrapper for full + trail compaction.

Replaces both the inline LLM calls in MessageProcessor._run_full_compaction /
_run_stage1_tool_compaction AND the deleted standalone compaction_service.

Two subclasses:
  - FullCompactionProcessor   → channel checkpoint summarisation (ex-Stage 2 / ex-backstop)
  - TrailCompactionProcessor  → mid-ACT tool-trail summarisation (ex-Stage 1)

Both share `_CompactionProcessorBase`:
  - JOB = 'frontal-cortex-unified' (same as parent message type per north star p.547)
  - NATIVE_TOOLS = []              (compaction never calls tools)
  - SKIP_TRANSCRIPT_WRITE = True   (no transcript pollution)
  - _check_threshold returns False (compaction never compacts itself — recursion guard)
  - getUserDefinition returns ''   (no human user)
  - getUserPrompt returns self._raw_input (caller pre-formats the input)

Callers:
  MessageProcessor._run_full_compaction        → FullCompactionProcessor
  MessageProcessor._run_stage1_tool_compaction → TrailCompactionProcessor
"""

from services.message_processor import MessageProcessor
from services.system_message_prompt import (
    CompactionFullSystemPrompt,
    CompactionTrailSystemPrompt,
)


class _CompactionProcessorBase(MessageProcessor):
    JOB = 'frontal-cortex-unified'
    NATIVE_TOOLS: list[str] = []
    SKIP_TRANSCRIPT_WRITE = True

    def getUserDefinition(self) -> str:
        return ''

    def getUserPrompt(self) -> str:
        return self._raw_input

    def getSystemPrompt(self) -> str:
        # Compaction has no human user — skip the empty user-definition prefix
        # that the base would otherwise stitch on with a leading '\n\n'.
        return self.SYSTEM_PROMPT_CLASS().getPrompt()

    def _check_threshold(self, _user_msg: str, _context_limit: int) -> bool:
        # Compaction never compacts itself. Hard-disable the gate so a large
        # input (which is the common case) cannot trigger recursion.
        return False


class FullCompactionProcessor(_CompactionProcessorBase):
    CHANNEL = 'compaction'
    ROLE = 'compaction'
    SYSTEM_PROMPT_CLASS = CompactionFullSystemPrompt


class TrailCompactionProcessor(_CompactionProcessorBase):
    CHANNEL = 'compaction_trail'
    ROLE = 'compaction'
    SYSTEM_PROMPT_CLASS = CompactionTrailSystemPrompt
