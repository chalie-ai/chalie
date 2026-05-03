"""
SubagentAbility — Spawn a focused subagent to execute a delegated task.

Three types, each with its own tool surface, system prompt, and default timeout:
  web_surfer   (60m): search + browse the open web and return a cited summary.
  summariser   (10m): read and compress long documents or web pages.
  general_purpose (30m): parallelise different long-running work.

Default is fire-and-forget (wait=false). When the subagent finishes it steers
the parent agent's ACT loop (Case A: parent mid-turn) or spawns a fresh
user-channel turn via SubagentReturnProcessor (Case B: parent idle).

When wait=true the parent ACT iteration blocks until the subagent finishes,
capped at 300 s regardless of type default.
"""

import json
import logging
import threading
import uuid

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SUBAGENT SKILL]"

# ── Per-type system prompts ───────────────────────────────────────────────────

_WEB_SURFER_PROMPT = """\
You are a web research subagent. Your job is to search the open web,
crawl pages, and return a focused, well-cited summary of what you found.

Tools available: read, search, browser, news, memory, find_tools.

Workflow:
- Start with `search` for breadth. Use `news` for recent events.
- Use `browser` to actually open and read pages that look promising.
- Use `read` for direct URLs or text content.
- Save intermediate findings to memory so progress survives a timeout.

Output:
- Lead with the answer. Supporting detail follows.
- Cite sources inline (URL or domain) for every factual claim.
- Be concise — no fluff, no narrative scaffolding, no apology paragraphs.
- If a source contradicts another, surface the contradiction.
- If the question is unanswerable from public web, say so explicitly.\
"""

_SUMMARISER_PROMPT = """\
You are a summarisation subagent. Your job is to read long content
(documents, web pages, transcripts) and return a tight summary.

Tools available: read, search, document, find_tools.

Workflow:
- Use `read` for URLs, `document` for files already in Chalie's store.
- If you need related context, use `search`.
- Use `find_tools` only if a more specialised tool is needed.

Output:
- Lead with a one-sentence TL;DR.
- Then the structured summary the parent asked for (bullet list,
  sectioned, table — match the parent's request).
- Quote the source verbatim only when the wording matters.
- Preserve numerical facts and named entities exactly.
- No commentary or evaluation unless the parent asked for it.\
"""

_GENERAL_PURPOSE_PROMPT = """\
You are a general-purpose subagent. The parent agent delegated this
task to you so it can do other work in parallel.

Tools available: memory, find_tools. Use `find_tools` to discover any
additional capabilities you need.

Workflow:
- Read the task brief carefully. The parent may have included context,
  data, or constraints — honour them.
- Save intermediate findings to memory so progress survives a timeout.
- If a tool fails, try alternatives before giving up.

Output:
- Be concise and structured. Match the output shape the parent asked
  for.
- No fluff, no narrative, no restating the task.
- If you hit a blocker, say so plainly with what you tried.\
"""

_SHARED_GUARDRAILS = """\
Output guardrails (apply always):
- Lead with the answer. Supporting detail follows.
- Be concise. No fluff, no narrative scaffolding ("As an AI...",
  "Let me think about...", "I'll now...").
- Do not restate the task.
- No emoji or decorative formatting unless the task explicitly asks.
- If you hit a blocker, say so plainly with what you tried.\
"""

# ── Type registry ─────────────────────────────────────────────────────────────

SUBAGENT_TYPES: dict[str, dict] = {
    "web_surfer": {
        "native_tools": ["read", "search", "browser", "news", "memory", "find_tools"],
        "default_timeout": 3600,   # 60 min
        "description": (
            "Search and crawl web pages and return a summary. "
            "Has web browsing capabilities."
        ),
        "system_prompt": _WEB_SURFER_PROMPT,
    },
    "summariser": {
        "native_tools": ["read", "search", "document", "find_tools"],
        "default_timeout": 600,    # 10 min
        "description": "Read and summarise long documents or web pages.",
        "system_prompt": _SUMMARISER_PROMPT,
    },
    "general_purpose": {
        "native_tools": ["memory", "find_tools"],
        "default_timeout": 1800,   # 30 min
        "description": (
            "Parallelise different long-running work — duplicate yourself "
            "to do different types of work in parallel."
        ),
        "system_prompt": _GENERAL_PURPOSE_PROMPT,
    },
}


# ── Envelope helpers ──────────────────────────────────────────────────────────

def _build_envelope(response_text: str, agent_type: str, status: str = "success") -> str:
    """Wrap the subagent output in the canonical envelope format."""
    if status == "success":
        return (
            f"[subagent.complete(type={agent_type})]\n"
            "The subagent has completed the task. Subagent's response:\n\n"
            f"{response_text}\n"
            "[end:subagent.complete]"
        )
    return (
        f"[subagent.complete(type={agent_type}, status=failure)]\n"
        f"The subagent failed. Reason: {response_text}.\n\n"
        "Decide whether to retry, escalate to the user, or fall back.\n"
        "[end:subagent.complete]"
    )


def _deliver_envelope(envelope: str, parent_ref: object) -> None:
    """Deliver the subagent envelope to the parent agent.

    Case A — parent turn is active (UserMessageProcessor on the call stack):
      Append the envelope to parent._pending_steers. The next iteration's
      getUserPrompt() drains _pending_steers into _act_trail.

    Case B — parent idle or non-UMP:
      Spawn a daemon thread that runs SubagentReturnProcessor(envelope).send()
      which produces a normal user-channel ACT turn and emits a 'message' event.

    The discriminator is ``isinstance(parent_ref, UserMessageProcessor)``.
    ``current_processor()`` only returns a non-None UMP while ``send()`` is
    on the call stack (it sets the contextvar on entry and resets on exit),
    so the type check alone is proof that the parent is mid-turn. We do NOT
    gate on ``_current_iteration > 0`` — iteration 0 is the first ACT pass
    and is just as much "mid-turn" as iteration 1+. Gating on > 0 would
    misroute a fast subagent completing during the parent's first iteration
    to Case B and spawn an unnecessary synthetic turn.
    """
    from services.user_message_processor import UserMessageProcessor

    if isinstance(parent_ref, UserMessageProcessor):
        # Case A: mid-ACT injection via _pending_steers
        try:
            parent_ref._pending_steers.append(envelope)
            logger.info(
                "%s Delivered envelope to parent _pending_steers (iteration=%d)",
                LOG_PREFIX, parent_ref._current_iteration,
            )
        except Exception as exc:
            logger.error(
                "%s Failed to append to parent _pending_steers: %s", LOG_PREFIX, exc
            )
            _spawn_return_processor(envelope)
        return

    # Case B: parent idle — synthetic user-channel turn
    _spawn_return_processor(envelope)


def _spawn_return_processor(envelope: str) -> None:
    """Spawn a daemon thread that runs SubagentReturnProcessor with the envelope."""

    def _run():
        try:
            from services.user_message_processor import SubagentReturnProcessor
            SubagentReturnProcessor(raw_input=envelope).send()
            logger.info("%s SubagentReturnProcessor completed", LOG_PREFIX)
        except Exception as exc:
            logger.error(
                "%s SubagentReturnProcessor failed: %s", LOG_PREFIX, exc, exc_info=True
            )

    t = threading.Thread(target=_run, daemon=True, name="subagent-return")
    t.start()


# ── Ability ───────────────────────────────────────────────────────────────────

class SubagentAbility(Ability):
    NAME = "subagent"
    SUMMARY = """\
Spawn a subagent to handle long-running, parallel, or context-heavy work.
Each subagent gets its own tool surface based on `type`:

- web_surfer (60m): search and crawl web pages; can browse. Use for
  multi-source web research and live page lookups.
- summariser (10m): read and summarise long documents or web pages.
  Use to compress large content before pulling it into your context.
- general_purpose (30m): parallelise different long-running work — spawn
  multiple subagents to do different things at once.

When NOT to use:
- For a single quick lookup with a known URL or query — use the tool
  directly.
- For tasks under ~30 seconds that fit in your turn — answer inline.
- When you need the result NOW and inline is faster than spawning.

Briefing rules:
- The subagent has none of your conversation context. State the task
  fully.
- Include success criteria and any data you already have.
- Be specific about output shape (summary length, format, key fields).
- Launch as many bounded subagents as you need; parallelise aggressively.\
"""
    EXAMPLES = [
        "research the top 3 health benefits of cold water swimming as a background task",
        "do a deep dive on the competitive landscape for electric vehicles",
        "investigate thoroughly how LLM context windows affect reasoning quality",
        "comprehensive analysis of the housing market in Malta over the last 5 years",
        "research multiple sources on the best diet for endurance athletes",
        "background task: compare React vs Vue for a large enterprise project",
        "deep research on the history and current state of quantum computing",
        "find out everything about carbon capture technology and write me a brief",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Detailed task brief. The subagent has none of your "
                    "conversation context. State the task fully, include any "
                    "context, data, success criteria, and output shape "
                    "(summary length, bullet list vs prose, key fields)."
                ),
            },
            "agent_type": {
                "type": "string",
                "enum": ["web_surfer", "summariser", "general_purpose"],
                "description": (
                    "web_surfer (60m): search and crawl web pages; can browse. "
                    "Use for multi-source web research and live page lookups.\n"
                    "summariser (10m): read and summarise long documents or web "
                    "pages. Use to compress large content before pulling it into "
                    "your context.\n"
                    "general_purpose (30m): parallelise different long-running "
                    "work — duplicate yourself to do different types of work in "
                    "parallel."
                ),
                "default": "general_purpose",
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "Set this to true to make the request synchronous. "
                    "Request will be hard capped to 5 minutes, else leave it "
                    "False and you will be notified automatically once the "
                    "request is completed."
                ),
                "default": False,
            },
        },
        "required": ["prompt"],
    }
    # Parent-dispatch timeout. wait=false returns in ms so 305 is harmless;
    # wait=true blocks the parent ACT iteration up to 300s (the wait-true cap)
    # plus a small buffer for envelope render + record. Anything below 305
    # short-circuits a legitimate sync run with "Action exceeded Ns timeout"
    # before SubagentProcessor finishes — the result column then carries the
    # generic timeout string instead of the canonical [subagent(...)] tag.
    TIMEOUT = 305

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"text": _skill_tag("subagent", error="prompt-required")}

        agent_type = params.get("agent_type", "general_purpose")
        if agent_type not in SUBAGENT_TYPES:
            return {"text": _skill_tag("subagent", error=f"invalid-type:{agent_type}")}

        wait = params.get("wait", False)
        timeout = SUBAGENT_TYPES[agent_type]["default_timeout"]
        if wait:
            timeout = min(timeout, 300)

        sub_id = uuid.uuid4().hex

        if wait:
            return self._run_sync(prompt, agent_type, timeout, sub_id)
        return self._run_async(prompt, agent_type, timeout, sub_id)

    # ── Execution paths ───────────────────────────────────────────────────────

    def _run_async(self, prompt: str, agent_type: str, timeout: int, sub_id: str) -> dict:
        """Fire-and-forget: spawn daemon thread, deliver result via steer/return."""
        from services.message_processor import current_processor

        parent_ref = current_processor()

        def _run():
            try:
                from services.subagent_processor import SubagentProcessor

                response_text = SubagentProcessor(
                    raw_input=prompt,
                    metadata={"sub_id": sub_id},
                    agent_type=agent_type,
                    max_timeout_override=timeout,
                ).send()
                response_text = (response_text or "").strip()
                if not response_text:
                    response_text = "Subagent completed but produced no output."

                envelope = _build_envelope(response_text, agent_type, status="success")
                logger.info("%s Subagent %s complete — delivering envelope", LOG_PREFIX, sub_id[:8])
            except Exception as exc:
                logger.error(
                    "%s Subagent %s failed: %s", LOG_PREFIX, sub_id[:8], exc, exc_info=True
                )
                envelope = _build_envelope(str(exc), agent_type, status="failure")

            _deliver_envelope(envelope, parent_ref)

        t = threading.Thread(target=_run, daemon=True, name=f"subagent-{sub_id[:8]}")
        t.start()

        body = json.dumps({
            "success": True,
            "sub_id": sub_id,
            "type": agent_type,
            "response": "Working on it. I'll notify you when done.",
        })
        return {"text": _skill_tag("subagent", body, sub_id=sub_id[:8], wait=False)}

    def _run_sync(self, prompt: str, agent_type: str, timeout: int, sub_id: str) -> dict:
        """Blocking: run SubagentProcessor inline and return its result."""
        from services.subagent_processor import SubagentProcessor

        try:
            response_text = SubagentProcessor(
                raw_input=prompt,
                metadata={"sub_id": sub_id},
                agent_type=agent_type,
                max_timeout_override=timeout,
            ).send()
            response_text = (response_text or "").strip()
        except Exception as exc:
            return {
                "text": _skill_tag(
                    "subagent", sub_id=sub_id[:8], wait=True, error=str(exc)
                )
            }

        return {
            "text": _skill_tag(
                "subagent",
                response_text,
                sub_id=sub_id[:8],
                wait=True,
            )
        }
