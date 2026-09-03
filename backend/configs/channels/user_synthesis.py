from __future__ import annotations

from abilities.recall import Recall
from abilities.save_synthesis import SaveSynthesis
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

_SYSTEM_PROMPT = """\
You are reviewing recent interactions between yourself the assistant (Chalie) and the user.
Create a terse summary in bullet point form about the user and valuable information to carry forward in your memory.
This will be the only information about the user carried forward and you will have no other memory of the user outside this summary.

Your goal; Capture hard facts about the user such as name, location, preferences, tone of voice, observed patterns, etc...
Focus; on the user ONLY. Everything that happened in the conversation is irrelevant, your focus is about observing the user so that you or the next agent get a terse profile about the user they are talking to.

How to work:
1. Read the latest summary (if one is not present, this is the first conversation with the user)
2. Process the transcript provided and extract the patterns, facts, preferences, etc...
3. Update (or generate) the summary to reflect the new information provided
4. Use the `recall` tool to cross-reference the user information that you have already and enrich / update based on the information from the `recall`
5. Call the `save_synthesis` tool to persist the new summary

IMPORTANT:
- A summary should NEVER exceed 200 words, compress heavily
- If the current synthesis is correct and requires no changes on the observed transcript, just exit, you will be called again with new transcripts when they come
- Do NOT invent or alter facts. ALWAYS validate your summary against `recall`, if you can't verify the claim DO NOT include it in the summary
- Focus mostly on long-term facts: Name, family, pets, household, goals, aspirations, recurring events, etc..."""


class UserSynthesisConfig(ProcessorConfig):
    """Compaction-fired user-synthesis turn — distils the folded conversation
    window into a new ``user_synthesis`` version.

    The caller (UserSynthesisGenerator) assembles the raw input — the current
    synthesis plus the folded transcript — so the prompt spine passes it
    straight through. ``suppress_history=True`` keeps the turn context-gate
    exempt (a ContextLimit re-raises instead of recursing into compaction) and
    ``skip_transcript=True`` keeps it out of the memory step; the role is
    distinct so history reads never conflate it with the "memory" role.
    """

    def __init__(self) -> None:
        super().__init__(
            channel=Channel.USER_SYNTHESIS.value,
            role="user_synthesis",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[Recall.NAME, SaveSynthesis.NAME],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
