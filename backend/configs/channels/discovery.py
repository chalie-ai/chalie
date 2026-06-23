from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)
LOG_PREFIX = "[DISCOVERY_CONFIG]"

# The task handed to the loop each tick. Deliberately open-ended: it states the
# intent (find something the user would care about, save it for later) and trusts
# the model to choose how to search and what is worth keeping.
DISCOVERY_PROMPT = (
    "Based on what you know about me and what we've discussed recently, search "
    "the web and read the news — did anything happen that I'd find interesting? "
    "Recall your earlier discoveries first so you don't repeat yourself. If, and "
    "only if, you find something genuinely interesting and relevant to me, record "
    "a memory about it so we can discuss it later. If nothing stands out, that's a "
    "fine outcome — record nothing and say so."
)

_DISCOVERY_SYSTEM_BODY = (
    "You are running as a quiet background research process — not a chat. No one "
    "is waiting on a reply. Your job is to proactively notice things in the world "
    "that this specific person would genuinely care about, given what you know "
    "about them and what they've discussed lately.\n\n"
    "How you work:\n"
    "- Reach the web through the discovery tools: use find_tools to bring up the "
    "web search and browsing delegates, then follow leads worth following.\n"
    "- Be selective. A generic headline is noise. Something tied to their actual "
    "interests, work, plans, or people is signal. When in doubt, keep nothing.\n"
    "- Recall your past discoveries before saving, and don't record the same "
    "finding twice.\n"
    "- Save a keeper by storing a memory (the memory tool, action=store): a short "
    "key and a value that captures the finding and why it matters to them. It is "
    "filed as a discovery automatically — you don't pick the kind.\n"
    "- This is transparent: the person can see these runs. Write your findings "
    "plainly, as if leaving them a note to read later."
)


class PersistDiscoveryRunHook(PostTurnHook):
    """Record one discovery run after the loop's final assistant row is persisted.

    The run row anchors the Brain "Auto Research" views: the grounding the loop
    ran against (captured in metadata at dispatch) plus a preview of what it said.
    The full output is read live from the transcript by turn, never duplicated.
    """

    def run(self, mp: "MessageProcessor", result_text: str) -> None:
        from services.discovery_runs import record_run
        meta = mp._metadata or {}
        record_run(
            turn_id=mp.turn_id,
            user_summary=cast("str", meta.get("user_summary") or ""),
            compacted_summary=cast("str", meta.get("compacted_summary") or ""),
            researched=(result_text or "").strip(),
        )


class DiscoveryConfig(ProcessorConfig):
    """Proactive autonomous-research channel.

    Fired at most once every few hours by the subconscious worker. Carries the
    framework discovery tools (DEFAULT_ALWAYS_AVAILABLE pins find_tools + memory),
    so it reaches the web through the search/browse delegates and records keepers
    as discovery memories. Grounding (user summary + latest compaction at dispatch
    time) is threaded in via metadata, not re-derived here.
    """

    def __init__(self) -> None:
        super().__init__(
            channel="discovery",
            role="discovery",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn_hooks=(PersistDiscoveryRunHook(),),
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return (
            "The user is 'discovery' — a background research process that "
            "represents your own initiative to look out for this person."
        )

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        """Grounding (user summary + recent discussion) + the task + ACT trail."""
        meta = mp._metadata or {}
        parts: list[str] = []
        summary = cast("str", meta.get("user_summary") or "")
        if summary:
            parts.append(f"## About the user\n{summary}")
        compacted = cast("str", meta.get("compacted_summary") or "")
        if compacted:
            parts.append(f"## Recently discussed\n{compacted}")
        parts.append(mp._raw_input)
        try:
            trail = mp._render_act_trail()
            if trail:
                parts.append(trail)
        except Exception as exc:
            logger.debug("%s _render_act_trail failed: %s", LOG_PREFIX, exc)
        return "\n\n".join(parts)

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        return f"{self.get_user_definition(mp)}\n\n{_DISCOVERY_SYSTEM_BODY}"
