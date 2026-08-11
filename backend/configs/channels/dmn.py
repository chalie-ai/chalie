from __future__ import annotations

from abilities.recall import Recall
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from services.processor_config import ProcessorConfig

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel

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

    @property
    def system_prompt(self) -> str:
        return f"""The user is 'proactive_thought' — a special background process that represents your own reflections on recent activity.

## Scope
The user has provided a synthesis about themselves under `About the User`. Your goal is to find open threads, recurring concerns, goals and aspirations the user has and ACT upon them.

## How to ACT

* Use the supplied tools to learn more topics the user discusses so that the next time they discuss such a topic you are aware of latest news, research, etc... You can use the `{WebSearchAbility.NAME}` and `{WebBrowseAbility.NAME}` tools for this. The background consolidator stores findings — during this turn you only `{Recall.NAME}` for context.
* Analyse patterns where the user seemed genuinely satisfied or dissatisfied with your responses or approach. The background consolidator stores feedback — during this turn you only `{Recall.NAME}` for context.

## When to stop

Aim for 2–3 substantive findings per tick — quality over quantity. The background consolidator stores insights automatically. Conclude with a brief one-line summary of what you recalled and observed. Do not pad with redundant tool calls or speculative topics."""
