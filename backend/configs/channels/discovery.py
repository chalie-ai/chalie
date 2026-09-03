from __future__ import annotations

from abilities.find_skills import FindSkillsAbility
from abilities.find_tools import FindToolsAbility
from abilities.mcp_manager import McpManagerAbility
from abilities.recall import Recall
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum

# The task handed to the loop each tick — written as a user asking for a research
# pass, grounded on the fresh main spine (DiscoveryConfig reads the ``user``
# channel's history via ``read_channel="user"``). The system prompt is the
# inherited UserConfig persona/voice prompt, unmodified; the user turn is the
# research task only. Persisting what the pass found is the settle-triggered
# memory step's job, never the pass's own.
DISCOVERY_PROMPT = (
    f"Can you run a background research pass for me? Ground it in our recent "
    f"conversation above, then search the web and read the news for anything "
    f"that's happened that I'd actually find interesting.\n\n"
    f"Use {FindToolsAbility.NAME} to bring up the web search and browsing delegates, and "
    f"follow whatever leads are worth following. Be selective — a generic "
    f"headline is noise; something tied to my actual interests, work, plans, or "
    f"people is signal. If nothing stands out, that's a fine outcome — just say so "
    f"and record nothing.\n\n"
    f"Write whatever you find plainly, as if you're leaving me a note to read "
    f"later."
)


class DiscoveryConfig(UserConfig):
    """Proactive autonomous-research channel.

    A thin split of UserConfig: it **reads** the ``user`` channel's cross-turn
    history (so it sees exactly what a user turn sees) but **writes** its rows
    to the ``discovery`` channel. The system prompt is the inherited UserConfig
    persona/voice prompt; the user turn is the research task only
    (``DISCOVERY_PROMPT``). Only the routing identity, the tool surface, the
    stable turn_id, and the silent-background-loop overrides differ from
    UserConfig. Fired at most once every few hours by the subconscious worker.

    Memory: the pass itself only reads — ``recall`` grounds the research in
    what is already known; it never writes memory inline. Each settled fire
    triggers the memory step, which forks into the research thread and
    distills the findings into Graph/Map. All of a box's discovery fires share
    one stable ``turn_id`` (``external_turn_id = True`` + the cron persists
    the id it allocated on the first fire), so the research log clusters as
    one thread.
    """

    BROADCASTS_STATE = False
    RENDERS_HTML = False
    # Re-declared, NOT inherited. Subclassing UserConfig would otherwise bill this
    # loop's spend as "foreground" — but the user never asked for it and never sees it;
    # Chalie runs it on its own initiative, so it is background spend. Every override
    # below follows the same rule: inherit UserConfig's *thinking*, never its
    # user-facing identity. A future UserConfig subclass must make the same call
    # deliberately rather than inherit a bill.
    USAGE_TYPE = "background"

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "channel", Channel.DISCOVERY.value)
        object.__setattr__(self, "read_channel", Channel.USER.value)
        object.__setattr__(self, "prompt_channel", Channel.USER.value)
        # Tool surface: the framework discovery pair + MCP manager, plus recall
        # so the pass can ground its research in what is already known. Memory
        # WRITES happen in the settle-triggered memory step, never inline.
        object.__setattr__(
            self,
            "always_available",
            [
                FindSkillsAbility.NAME,
                FindToolsAbility.NAME,
                McpManagerAbility.NAME,
                Recall.NAME,
            ],
        )
        # The discovery turn_id is a stable, box-owned key (persisted by the cron
        # job on first fire): the first research pass opens the turn, every later
        # fire forks into it, so the whole research log is one thread.
        object.__setattr__(self, "external_turn_id", True)

    def type(self) -> ConfigTypeEnum:
        return ConfigTypeEnum.DISCOVERY
