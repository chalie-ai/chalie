"""
SubagentAbility — Spawn a focused subagent to execute a delegated task.

Three types, each with its own tool surface, system prompt, and default timeout:
  web_surfer   (60m): search + browse the open web and return a cited summary.
  summariser   (10m): read and compress long documents or web pages.
  general_purpose (30m): parallelise different long-running work.

Default is fire-and-forget (wait=false). When the subagent finishes it
delivers the result via the chat chokepoint (_start_turn with hidden_input),
which either steers the active UMP or starts a fresh user-channel turn.

When wait=true the parent ACT iteration blocks until the subagent finishes,
capped per-type by ``wait_cap``: web_surfer caps at 30 min (web research is
slow), summariser and general_purpose cap at 5 min. Prefer wait=false for
web_surfer — multi-site crawls regularly run for many minutes and blocking
the parent that long stalls every other reply.
"""

import json
import logging
import threading
import uuid

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag
from services.rich_media_parser import strip_spans
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SUBAGENT SKILL]"

# ── Active subagent registry ──────────────────────────────────────────────────
#
# Tracks all async subagents that are currently running so the stop endpoint
# can signal cooperative cancellation. Entries are added before thread.start()
# and removed in the thread's finally block, guaranteeing no leaked entries.
# Each entry: {"processor": SubagentProcessor, "agent_type": str,
#               "description": str, "started_at": str}

_active_subagents: dict[str, dict] = {}
_subagent_lock = threading.Lock()


def get_active_subagents() -> dict[str, dict]:
    """Return a snapshot of the active subagent registry (thread-safe)."""
    with _subagent_lock:
        return dict(_active_subagents)


def cancel_subagent(sub_id: str) -> bool:
    """Set the cancel_event on a running subagent. Returns True if found."""
    with _subagent_lock:
        entry = _active_subagents.get(sub_id)
    if entry is None:
        return False
    entry["processor"]._cancel_event.set()
    return True

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

# default_timeout = the absolute wall-clock the SubagentProcessor is allowed
#                   to run before its own deadline kicks in (fire-and-forget).
# wait_cap        = the maximum the parent ACT iteration may block when the
#                   caller passes wait=true. Always <= default_timeout.
#                   web_surfer's cap is intentionally generous (30 min) because
#                   real-world multi-site crawls regularly exceed 5 min and a
#                   sync subagent that times out at 300s is useless. Other
#                   types stay at 300s — anything that needs longer should
#                   dispatch async.
SUBAGENT_TYPES: dict[str, dict] = {
    "web_surfer": {
        "native_tools": ["read", "search", "browser", "news", "memory", "find_tools"],
        "default_timeout": 3600,   # 60 min
        "wait_cap": 1800,          # 30 min sync cap — web research is slow
        "description": (
            "Search and crawl web pages and return a summary. "
            "Has web browsing capabilities."
        ),
        "system_prompt": _WEB_SURFER_PROMPT,
    },
    "summariser": {
        "native_tools": ["read", "search", "document", "find_tools"],
        "default_timeout": 600,    # 10 min
        "wait_cap": 300,           # 5 min sync cap
        "description": "Read and summarise long documents or web pages.",
        "system_prompt": _SUMMARISER_PROMPT,
    },
    "general_purpose": {
        "native_tools": ["memory", "find_tools"],
        "default_timeout": 1800,   # 30 min
        "wait_cap": 300,           # 5 min sync cap
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


# ── Ability ───────────────────────────────────────────────────────────────────

class SubagentAbility(Ability):
    NAME = "subagent"
    SUMMARY = """\
ONLY use this tool if multi-step actions are needed OR you need to process
a large volume of data (call summariser in this case). For a single tool
lookup, call the tool directly — do not wrap it in a subagent. Multiple
calls to the same tool (e.g. weather for two cities) is NOT multi-step —
call the tool once per item.

Each subagent gets its own tool surface based on `agent_type`:

- web_surfer (60m): search and crawl web pages; can browse. Use for
  multi-source web research and live page lookups. Web research is SLOW —
  multi-site crawls routinely take many minutes. **Strongly prefer
  wait=false** (fire-and-forget). The result is delivered back to you
  asynchronously when ready, so you can keep talking to the user or fan
  out more subagents in parallel.
- summariser (10m): read and summarise long documents or web pages.
  Use to compress large content before pulling it into your context.
- general_purpose (30m): parallelise different long-running work — spawn
  multiple subagents to do different things at once.

Sync vs async (wait):
- wait=false (default): returns immediately with an ack. The completed
  envelope arrives later as a steer or a fresh turn. Use this for any
  web_surfer call and for anything you don't need to act on inside this
  same turn.
- wait=true: blocks the parent ACT iteration until the subagent finishes.
  Capped per type — web_surfer 30 min, summariser/general_purpose 5 min.
  Only use this when the subagent's answer is the next thing you must
  reason about and the task is genuinely fast.

Parallelism:
- Fan out aggressively. Three wait=false web_surfers (one per source) beat
  one monolith trying to crawl everything sequentially.

When NOT to use:
- For a single quick lookup with a known URL or query — use the tool
  directly.
- For tasks under ~30 seconds that fit in your turn — answer inline.

Briefing rules:
- The subagent has none of your conversation context. State the task
  fully.
- Include success criteria and any data you already have.
- Be specific about output shape (summary length, format, key fields).\
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
                    "Use for multi-source web research and live page lookups. "
                    "Web research is SLOW (multi-site crawls regularly take "
                    "many minutes) — strongly prefer wait=false so you are not "
                    "blocked.\n"
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
                    "false (default, recommended): fire-and-forget. Returns an "
                    "ack immediately and the completed envelope arrives back "
                    "later as a steer or a fresh turn. ALWAYS use this for "
                    "web_surfer — multi-site crawls regularly run for many "
                    "minutes and blocking your turn that long is wasteful. "
                    "true: synchronous — blocks this ACT iteration until the "
                    "subagent finishes. Hard capped per type: web_surfer 30 "
                    "min, summariser and general_purpose 5 min. Only use when "
                    "the subagent's answer is the very next thing you need to "
                    "reason about."
                ),
                "default": False,
            },
        },
        "required": ["prompt"],
    }
    # Parent-dispatch timeout. wait=false returns in ms; wait=true blocks the
    # parent ACT iteration up to the per-type wait_cap (currently max 1800s
    # for web_surfer) plus a small buffer for envelope render + record.
    # This ceiling MUST be >= max(wait_cap) across SUBAGENT_TYPES, otherwise
    # the dispatcher kills a legitimate sync run with "Action exceeded Ns
    # timeout" before SubagentProcessor finishes — the result column would
    # then carry the generic timeout string instead of the canonical
    # [subagent(...)] tag.
    TIMEOUT = 1805

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"text": _skill_tag("subagent", error="prompt-required")}

        agent_type = params.get("agent_type", "general_purpose")
        if agent_type not in SUBAGENT_TYPES:
            return {"text": _skill_tag("subagent", error=f"invalid-type:{agent_type}")}

        wait = params.get("wait", False)
        type_entry = SUBAGENT_TYPES[agent_type]
        timeout = type_entry["default_timeout"]
        if wait:
            # wait=true caps per-type via wait_cap. web_surfer = 1800s (web
            # research is genuinely slow); other types = 300s. Always <=
            # default_timeout so the SubagentProcessor's own deadline still
            # bounds it.
            timeout = min(timeout, type_entry["wait_cap"])

        sub_id = uuid.uuid4().hex
        # Mutate in-place so the ACT loop's reference to params includes sub_id
        # when it writes the tool_calls row.
        params["sub_id"] = sub_id

        if wait:
            return self._run_sync(prompt, agent_type, timeout, sub_id)
        return self._run_async(prompt, agent_type, timeout, sub_id)

    # ── Execution paths ───────────────────────────────────────────────────────

    def _run_async(self, prompt: str, agent_type: str, timeout: int, sub_id: str) -> dict:
        """Fire-and-forget: spawn daemon thread, deliver result via chat chokepoint."""
        from services.subagent_processor import SubagentProcessor
        from services.websocket_broker import WebSocketBroker

        cancel_event = threading.Event()
        description = SUBAGENT_TYPES[agent_type]["description"]
        proc = SubagentProcessor(
            raw_input=prompt,
            metadata={"sub_id": sub_id},
            agent_type=agent_type,
            max_timeout_override=timeout,
            cancel_event=cancel_event,
        )

        with _subagent_lock:
            _active_subagents[sub_id] = {
                "processor": proc,
                "agent_type": agent_type,
                "description": description,
                "started_at": utc_now().isoformat(),
            }

        def _run():
            broker = WebSocketBroker()
            status = "done"
            envelope = ""
            try:
                broker.broadcast({
                    "type": "subagent_start",
                    "sub_id": sub_id,
                    "agent_type": agent_type,
                    "description": description,
                })

                response_text = proc.send()
                response_text = strip_spans((response_text or "").strip()).strip()
                if not response_text:
                    response_text = "Subagent completed but produced no output."

                if cancel_event.is_set():
                    status = "cancelled"
                    envelope = _build_envelope(
                        "Subagent was cancelled before completing.", agent_type, status="failure"
                    )
                else:
                    envelope = _build_envelope(response_text, agent_type, status="success")
                logger.info(
                    "%s Subagent %s complete (status=%s) — delivering envelope",
                    LOG_PREFIX, sub_id[:8], status,
                )
            except Exception as exc:
                status = "error"
                logger.error(
                    "%s Subagent %s failed: %s", LOG_PREFIX, sub_id[:8], exc, exc_info=True
                )
                envelope = _build_envelope(str(exc), agent_type, status="failure")
            finally:
                with _subagent_lock:
                    _active_subagents.pop(sub_id, None)
                broker.broadcast({
                    "type": "subagent_end",
                    "sub_id": sub_id,
                    "status": status,
                })

            try:
                from api.chat import dispatch_message
                dispatch_message(envelope, source='subagent', hidden_input=True, intercept=True)
            except Exception as exc:
                logger.error(
                    "%s Subagent %s delivery failed: %s", LOG_PREFIX, sub_id[:8], exc, exc_info=True
                )

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
            response_text = strip_spans((response_text or "").strip()).strip()
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
