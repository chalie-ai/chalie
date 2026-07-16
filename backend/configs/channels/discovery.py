from __future__ import annotations

from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum

# The task handed to the loop each tick — written as a user asking for a research
# pass, grounded on the fresh main spine (DiscoveryConfig reads the ``user``
# channel's history via ``read_channel="user"``). The system prompt is the
# inherited UserConfig persona/voice prompt; this is the user-turn input only.
DISCOVERY_PROMPT = (
    "Can you run a background research pass for me? Ground it in our recent "
    "conversation above, then search the web and read the news for anything "
    "that's happened that I'd actually find interesting.\n\n"
    "Use find_tools to bring up the web search and browsing delegates, and "
    "follow whatever leads are worth following. Be selective — a generic "
    "headline is noise; something tied to my actual interests, work, plans, or "
    "people is signal. Recall your earlier discoveries first so you don't "
    "record the same thing twice. If you find something worth keeping, save it "
    "with the memory tool (action=store): a short key, what it is, and why it "
    "matters to me — it gets filed automatically, you don't need to pick a "
    "kind. If nothing stands out, that's a fine outcome — just say so and "
    "record nothing.\n\n"
    "Write whatever you find plainly, as if you're leaving me a note to read "
    "later."
)


class DiscoveryConfig(UserConfig):
    """Proactive autonomous-research channel.

    A thin split of UserConfig: it **reads** the ``user`` channel's cross-turn
    history (so it sees exactly what a user turn sees) but **writes** its rows
    to the ``discovery`` channel. The system prompt, user definition, world
    state, memory seed and prompt assembly are all inherited from UserConfig —
    only the routing identity and the silent-background-loop overrides differ.
    Fired at most once every few hours by the subconscious worker.
    """

    SUPPORTS_ASYNC = False
    BROADCASTS_STATE = False
    # Re-declared, NOT inherited. Subclassing UserConfig would otherwise bill this
    # loop's spend as "chat" — but the user never asked for it and never sees it;
    # Chalie runs it on its own initiative, so it is system spend. Every override
    # below follows the same rule: inherit UserConfig's *thinking*, never its
    # user-facing identity. A future UserConfig subclass must make the same call
    # deliberately rather than inherit a bill.
    USAGE_TYPE = "system"

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "channel", Channel.DISCOVERY.value)
        object.__setattr__(self, "read_channel", Channel.USER.value)
        object.__setattr__(self, "broadcast_to", None)
        object.__setattr__(self, "prompt_channel", Channel.USER.value)

    def type(self) -> ConfigTypeEnum:
        return ConfigTypeEnum.DISCOVERY
