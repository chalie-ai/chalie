from __future__ import annotations

from abilities.delete_graph import DeleteGraph
from abilities.find_skills import FindSkillsAbility
from abilities.find_tools import FindToolsAbility
from abilities.mcp_manager import McpManagerAbility
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum

# The task handed to the loop each tick — written as a user asking for a research
# pass, grounded on the fresh main spine (DiscoveryConfig reads the ``user``
# channel's history via ``read_channel="user"``). The system prompt is the
# inherited UserConfig persona/voice prompt; this is the user-turn input only.
#
# Recall-before / save-after: the loop must recall what it already knows before
# recording anything, and persist only via the Memory v3 tools (Graph for living
# facts, Map for narrative lineage) — never the legacy in-conversation memory.
DISCOVERY_PROMPT = (
    f"Can you run a background research pass for me? Ground it in our recent "
    f"conversation above, then search the web and read the news for anything "
    f"that's happened that I'd actually find interesting.\n\n"
    f"Use {FindToolsAbility.NAME} to bring up the web search and browsing delegates, and "
    f"follow whatever leads are worth following. Be selective — a generic "
    f"headline is noise; something tied to my actual interests, work, plans, or "
    f"people is signal.\n\n"
    f"Before recording anything, call {Recall.NAME} on the topics you're about to "
    f"write so you don't duplicate or contradict what's already stored. Then, if "
    f"you find something worth keeping, save it: {SaveGraph.NAME} for a durable fact "
    f"(a short subject key + the current value) or {SaveMap.NAME} for a narrative "
    f"thread that extends something earlier. Use {DeleteGraph.NAME} only to retire a "
    f"fact that this pass overturns. If nothing stands out, that's a fine outcome "
    f"— just say so and record nothing.\n\n"
    f"Write whatever you find plainly, as if you're leaving me a note to read "
    f"later."
)


class DiscoveryConfig(UserConfig):
    """Proactive autonomous-research channel.

    A thin split of UserConfig: it **reads** the ``user`` channel's cross-turn
    history (so it sees exactly what a user turn sees) but **writes** its rows
    to the ``discovery`` channel. The system prompt, user definition, world
    state, memory seed and prompt assembly are all inherited from UserConfig —
    only the routing identity, the tool surface, the stable turn_id, and the
    silent-background-loop overrides differ. Fired at most once every few hours
    by the subconscious worker.

    Memory v3: discovery writes its findings through the Graph/Map tools
    (``recall`` / ``save_graph`` / ``save_map`` / ``delete_graph``) instead of
    the legacy in-conversation memory. All of a box's discovery fires share one
    stable ``turn_id`` (``external_turn_id = True`` + the cron persists the id
    it allocated on the first fire), so the research log clusters as one thread.
    """

    SUPPORTS_ASYNC = False
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
        # Memory v3 tool surface: keep the framework discovery pair + MCP manager,
        # drop the legacy in-conversation memory, and pin the Graph/Map tools so
        # the research pass can recall + persist findings directly.
        object.__setattr__(
            self,
            "always_available",
            [
                FindSkillsAbility.NAME,
                FindToolsAbility.NAME,
                McpManagerAbility.NAME,
                Recall.NAME,
                SaveGraph.NAME,
                SaveMap.NAME,
                DeleteGraph.NAME,
            ],
        )
        # The discovery turn_id is a stable, box-owned key (persisted by the cron
        # job on first fire): the first research pass opens the turn, every later
        # fire forks into it, so the whole research log is one thread.
        object.__setattr__(self, "external_turn_id", True)

    def type(self) -> ConfigTypeEnum:
        return ConfigTypeEnum.DISCOVERY
