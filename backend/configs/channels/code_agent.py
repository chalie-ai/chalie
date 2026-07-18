# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CodeAgentConfig — delegate channel for the ``code_agent`` tool.

Mirrors WebSearchConfig/PimConfig's transcript wiring: a real per-turn row on
its own ``delegate:code_agent`` channel so the delegate can render its own
act-trail across ACT iterations (``skip_transcript=False`` /
``skip_input_row=False``). Unlike those two, this channel does NOT set
``uses_delegate_provider`` — it stays on the default ``False`` and therefore
runs on the brain's globally selected (main) provider, not the Delegate
Provider slot. Coding work benefits from the same model quality the user's
own conversation gets, and there is no separate "delegate-grade" model tier
worth carving out for it. Paired with ``CodeAgentAbility`` (abilities/code_agent.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configs.enums.channels import Channel
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from configs.enums.policy_channel import PolicyChannel

# The 12 code_agent-exclusive inner tools (abilities/{name}.py), each
# DISCOVERABLE=False and reachable ONLY through this always_available list —
# see abilities/_workspace.py for the shared sandbox they all resolve paths
# under.
_INNER_TOOLS: tuple[str, ...] = (
    "read_file",
    "create_file",
    "create_folder",
    "delete_file",
    "move_file",
    "update_file",
    "replace_one",
    "replace_all",
    "list_files",
    "find",
    "find_files",
    "run_script",
)


class CodeAgentConfig(ProcessorConfig):
    """``policy_channel`` is supplied by the caller (inherited from whoever
    invoked the tool) rather than hardcoded."""

    def __init__(self, policy_channel: "PolicyChannel") -> None:
        super().__init__(
            channel=Channel.DELEGATE_CODE_AGENT.value,
            role="code_agent",
            policy_channel=policy_channel,
            always_available=[*_INNER_TOOLS, "memory", "web_search", "programming_docs_search"],
            skip_transcript=False,  # write a delegate-channel transcript row so
            skip_input_row=False,   # _setup assigns the uid the act-trail needs
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """You are Chalie's coding agent. You receive one coding task and carry it out using a sandboxed file workspace and a TypeScript runtime — read_file, create_file, create_folder, delete_file, move_file, update_file, replace_one, replace_all, list_files, find, find_files, run_script. You also have memory, web_search, and programming_docs_search for research.

Work directly in the workspace: create or edit the files the task needs, then use run_script to execute a .ts file and check its output. Prefer replace_one/replace_all for targeted edits and update_file only for a full rewrite — overwriting a large file with a small snippet is refused as destructive. Verify your own work by running it before declaring it done; do not guess at output you have not actually observed.

Return a concise summary of what you did (files created/edited, what the code does, what running it produced) and the concrete answer or result the task asked for. You have no conversation history and no user personality — work only from the task."""
