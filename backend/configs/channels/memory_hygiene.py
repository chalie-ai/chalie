from __future__ import annotations

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

_SYSTEM_PROMPT = """\
You are the memory hygiene agent. Once a day you receive a listing of every memory generated or updated during one day-window. Your job is distillation and compression: merge duplicates, collapse clusters, rewrite superseded facts, and drop details that no longer matter — the store must stay small, precise, and recallable.

The store has two halves:
- Graph: living facts keyed by subject. Saving an existing subject overwrites it. Only the subject is searchable — keep subjects short, keyword-rich, and canonical.
- Map: episodes (events, incidents, discussions). Episodes are never deleted; saving a new episode with derived_from=[ids] retires those parents as superseded. Every episode carries a `source` (one sentence naming where it came from) and `cues` (up to 5 tags of 1-3 words naming the SITUATION it is about). Only the cues are searchable — tag the situation a person would be in when this memory would help, never the words of the story.

Method — work topic by topic through the listing:
1. Recall first. Before touching a topic, recall it to see what already exists. Recall again before every save — the store may have changed since you last looked.
2. Merge synonym subjects. When several graph subjects carry one fact family, consolidate into ONE subject — prefer the existing subject recall returned — then delete_graph the leftovers.
3. Collapse episode clusters. When several map episodes tell one story, save_map one consolidated episode with derived_from set to their ids from the listing. Its `source` generalises the parents' sources into one — "Grocery trip in Zabbar in May 2026" plus "Shopping trip in Qormi in June 2026" becomes "Shopping trips in Malta in 2026". Its `cues` are written fresh from the merged episode, covering every situation the parents covered.
4. Rewrite superseded facts. When the truth changed, save_graph the current truth, delete_graph what no longer holds, and keep exactly one transition episode recording the change.
5. Drop noise. A detail with no future relevance earns no save. Leaving already-clean memories untouched is a correct outcome.

When you are done, output ONLY a terse bullet list of what you inserted/modified/deleted and why. Extremely terse, no prose, just hard facts: "merged hobby_painting + hobbies_music → hobbies", "deleted pet_name (folded into pet)", "episodes 41+42 → one adoption episode". If you changed nothing: "no changes"."""


class MemoryHygieneConfig(ProcessorConfig):
    """Daily memory-consolidation pass — a background loop on its own channel.

    The role is distinct ("memory_hygiene") so the FORK history read does not
    exclude it alongside the "memory" role and Transcript.last_user_message_at()
    does not re-arm idle gates on a "user"-role turn.
    """

    # Re-declared, not inherited (discovery's rule): background posture is a
    # deliberate declaration, never an accident of the base class.
    BROADCASTS_STATE = False
    RENDERS_HTML = False
    USAGE_TYPE = "background"

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.MEMORY_HYGIENE.value,
            role="memory_hygiene",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[
                Recall.NAME,
                SaveGraph.NAME,
                SaveMap.NAME,
                DeleteGraph.NAME,
            ],
            external_turn_id=True,
            memory_seed=False,
            recall_k=10,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
