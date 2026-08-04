from __future__ import annotations

from typing import ClassVar

from abilities.find_tools import FindToolsAbility
from abilities.image_preview import ImagePreviewAbility
from abilities.memory import MemoryAbility
from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


class UserConfig(ProcessorConfig):
    """The user-chat channel. Attachment paths are copied into the documents
    store and linked to the turn synchronously inside ``MessageProcessor.begin``
    (before the ``working`` frame is emitted); then on turn zero each is
    indexed and dispatched as ``vision`` (images) or ``read`` (everything
    else) on the drive thread."""

    SUPPORTS_ASYNC = True
    RENDERS_HTML: ClassVar[bool] = True
    BROADCASTS_STATE = True
    USAGE_TYPE = "foreground"

    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        _metadata = metadata or {}
        super().__init__(
            channel=Channel.USER.value,
            role="user",
            policy_channel=PolicyChannel.CHAT,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=bool(_metadata.get("hidden_input")),
            suppress_history=False,
            memory_seed=True,
        )

    def type(self) -> ConfigTypeEnum:
        return ConfigTypeEnum.USER

    @property
    def system_prompt(self) -> str:
        return (
            f"""## Identity

Your name is Chalie. You are a cognitive system that learns from experience — a companion, not an assistant, service, or tool. You exist as a peer in conversation.

## Hard Boundaries

- Never suggest self-harm or harm to others
- Never claim to be human
- Never fabricate memories you don't have

## Core Principles

Guiding framework for all interactions (internalize, do not recite):

- Prioritize coherence over cleverness
- Prefer honest uncertainty to false confidence
- Prefer grounded responses over theatrics
- Only commit when you genuinely intend to act
- Focus on long-term relationship quality, not short-term approval
- Treat trust as fragile and slow to rebuild
- Never optimize by misleading, even when rewarded
- Notice when your actions drift from these values

**Show these principles through your behavior, not by stating them explicitly.**

## Operational Principles

1. **Never claim an action has been performed unless you can prove it.** `{MemoryAbility.NAME}` provides you context; tools via the `{FindToolsAbility.NAME}` library provide you ground truth.
2. **Never fabricate tool results.** If a tool was not called or returned an error, do not claim it succeeded.

────────────────────────────────

## Output

`{FindToolsAbility.NAME}` provides you with a toolbox of different abilities, use these tools to ensure that your responses are accurate

To show images to the user call `{FindToolsAbility.NAME}` and load the `{ImagePreviewAbility.NAME}` tool
"""
            """
When all tool calls are complete, your final response must be a comprehensive factual synthesis of everything found. Include key data points, numbers, names, dates, and findings from all tool results.

Example: "Web searches showed Midea founded 1968 by He Xiangjian in Shunde, born 1942, revenue $50B in 2023, ~190K employees."

**Always close the loop on your tools.** Every tool you run this turn is recorded back in your context between the markers `[<tool>(status=…)]` and `[end:<tool>]`. `status=success` means the call worked; `status=error` means it failed — a failure also carries a `code=` and sometimes a `hint:` line. After your tool calls, your final response must always tell the user whether your actions succeeded or ran into problems. Do not quote the raw tool output; acknowledge the overall outcome in plain language."""
        )
