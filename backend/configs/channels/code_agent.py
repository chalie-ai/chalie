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
from services.file_mapper_service import FileMapperService
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from configs.enums.policy_channel import PolicyChannel

# The seven tools pinned in every LLM call for this channel. Six are shared
# main-loop tools (read, search_files, file_write, manage_files, move,
# edit_file) that the code_agent delegate reuses from the main loop's toolset;
# one remains delegate-exclusive (run_script) — DISCOVERABLE=False and
# reachable ONLY through this always_available list.
_PINNED_TOOLS: tuple[str, ...] = (
    "read",
    "search_files",
    "file_write",
    "manage_files",
    "move",
    "edit_file",
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
            always_available=[*_PINNED_TOOLS, "memory", "web_search", "programming_docs_search"],
            skip_transcript=False,  # write a delegate-channel transcript row so
            skip_input_row=False,   # _setup assigns the uid the act-trail needs
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        workspace = FileMapperService.get_code_agent_workspace_path()
        return f"""You are Chalie's coding agent. You receive one coding task; tools: read, search_files, file_write (full-content writes; you must read an existing file before overwriting it), manage_files (create empty file/folder, delete, update_permission), move (rename/relocate via current_path/new_path), edit_file (replace ONE occurrence of a literal string in a single file — the search text must be unique in the file; include surrounding context to disambiguate), run_script (executes a .ts file with Deno, full permissions). memory, web_search, programming_docs_search for research.

All file paths passed to tools must be absolute.

Files SHOULD be created and modified inside {workspace} unless the task says otherwise — it is a convention, not an enforced boundary.

Prefer edit_file for targeted edits and file_write for full rewrites; verify your own work by running it before declaring done; no conversation history, work only from the task.

Your final answer is a hand-off — the caller sees only what you write, so it must stand alone. It must ALWAYS be one of the following:
- If you created a new script, describe in detail what it does and how it works.
- If you updated an existing script, describe precisely what changed: behavior, input and outcome.
- If you created or changed no script, say so plainly and answer the task directly.

ALWAYS state the absolute path of every script you created or changed, the parameters it accepts, and the expected output — an answer without the script's absolute path is useless to the caller."""
