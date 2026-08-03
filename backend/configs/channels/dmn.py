from __future__ import annotations

from abilities.memory import MemoryAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from models.episode import Episode
from services.processor_config import ProcessorConfig

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel

_EPISODE_RETRIEVAL_WEIGHT_FLOOR = 0.3
_DMN_EPISODE_LOOKBACK_DAYS = 30
_DMN_EPISODE_LIMIT = 50


class DmnConfig(ProcessorConfig):
    """DMN background channel.  No after-turn hooks (metrics are emitted by the gateway).

    DMN carries the framework discovery tools (DEFAULT_ALWAYS_AVAILABLE includes
    find_tools), so it can discover and spawn any DISCOVERABLE tool — including
    the web_search / web_browse / vision delegates — when a reflection genuinely
    needs the web. The raw web tools (browser/search/news) and the pattern
    writers stay non-discoverable globally, so DMN reaches the web only through
    the delegates, never directly.
    """

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.DMN.value,
            role="proactive_thought",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            memory_seed=False,
        )

    @staticmethod
    def recent_salient_user_episodes() -> str:
        """Recent, non-decayed ``user``-channel episodes as a numbered list —
        ``N. [ts] (salience=…) gist`` — or ``''`` when none / on error (the read
        must never crash the DMN turn).

        @todo: Refactor — a config must not read the DB (§2.4). The read now
        routes through the Episode model, but it still doesn't belong in a config:
        this static is a deliberate stop-gap until episodic prompt-context is folded
        onto the spine as a proper service; it stays here (not a loose module
        function) so there is exactly one home for the DMN reads.
        """
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            episodes = Episode.recent_salient(
                "user",
                weight_floor=_EPISODE_RETRIEVAL_WEIGHT_FLOOR,
                lookback_days=_DMN_EPISODE_LOOKBACK_DAYS,
                limit=_DMN_EPISODE_LIMIT,
            )
            lines = []
            for i, ep in enumerate(episodes, 1):
                ts = (ep.created_at or "")[:16].replace("T", " ")
                lines.append(f"{i}. [{ts}] (salience={ep.salience}) {ep.gist or ''}")
            return "\n".join(lines)
        except Exception as exc:
            _log.warning("[DMN_CONFIG] recent_salient_user_episodes failed: %s", exc)
            return ""

    @property
    def system_prompt(self) -> str:
        return f"""The user is 'proactive_thought' — a special background process that represents your own reflections on recent activity.

## Scope
The user has provided a synthesis about themselves under `About the User` and relevant episodic memories regarding conversations you had with `Chalie`. Your goal is to find open threads, recurring concerns, goals and aspirations the user has and ACT upon them.

## How to ACT

* Use the supplied tools to learn more topics the user discusses so that the next time they discuss such a topic you are aware of latest news, research, etc... You can use the `{WebSearchAbility.NAME}` and `{WebBrowseAbility.NAME}` tools for this. Save your findings using the `{MemoryAbility.NAME}` tool so that you can reference them later.
* Analyse patterns where the user seemed genuinely satisfied or dissatisfied with your responses or approach and store feedback to not repeat the same mistakes or reinforce good behaviour. Use the `{MemoryAbility.NAME}` tool for this.

## When to stop

Aim for 2–3 substantive findings per tick — quality over quantity. Once you have saved meaningful insights via the `{MemoryAbility.NAME}` tool, conclude with a brief one-line summary of what you saved. Do not pad with redundant tool calls or speculative topics."""
