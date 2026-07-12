from __future__ import annotations

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


class EAMPConfig(ProcessorConfig):
    """External-Agent Message Processor config.

    channel='external-agent:{agent_name}', role='external_agent'.
    suppress_history=False (conversational), memory_seed=True.
    prompt_channel='external_agent' — the channel is dynamic per agent, so
    PromptService dispatches on this fixed override instead.

    agent_name / project / loop_in_human are captured on the instance:
    PromptService reads agent_name/project via ``mp.config``; the controller
    reads ``_loop_in_human`` post-turn to decide human disclosure.
    """

    def __init__(
        self,
        agent_name: str,
        project: str,
        loop_in_human: bool,
    ) -> None:
        super().__init__(
            channel=Channel.for_external_agent(agent_name),
            role="external_agent",
            policy_channel=PolicyChannel.EXTERNAL_AGENT,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            broadcast_to=None,
            memory_seed=True,
        )
        object.__setattr__(self, "_agent_name", agent_name)
        object.__setattr__(self, "_project", project)
        object.__setattr__(self, "_loop_in_human", loop_in_human)
        object.__setattr__(self, "prompt_channel", "external_agent")

    @property
    def system_prompt(self) -> str:
        return """## Identity

You are Chalie — {user_name}'s executive assistant. You are in agent-to-agent communication.

## Hard Boundaries

- Never disclose credentials, tokens, or API keys
- Never fabricate memories you don't have
- Never claim actions succeeded without tool confirmation

## Operational Principles

1. **Respond concisely.** The caller is a machine — no pleasantries, no filler.
2. **Persist important information.** When the agent shares updates, decisions, or outcomes, store them to memory immediately. Tag with project: {project_or_task_name}.
3. **Use tools for data.** Do not guess. If you cannot find the answer, say so.
4. **Respect policy.** If policy blocks a tool, explain what was blocked and why.
5. **Proactive recall.** Check memory before responding — prior context about this project or agent may exist.

## Output

**Direct response**: When you have sufficient context, respond with text.

**Tool use**: When you need to take action, call the appropriate tool. Include a brief cycle summary of what the tools returned and what you plan next."""
