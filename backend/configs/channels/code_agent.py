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

# The seven tools pinned in every LLM call for this channel. Five are shared
# main-loop tools (read, search_files, file_write, manage_files, move) that the
# code_agent delegate reuses from the main loop's toolset; two remain
# delegate-exclusive (replace_all, run_script) — DISCOVERABLE=False and
# reachable ONLY through this always_available list.
_PINNED_TOOLS: tuple[str, ...] = (
    "read",
    "search_files",
    "file_write",
    "manage_files",
    "move",
    "replace_all",
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
        return f"""You are Chalie's coding agent. You receive one coding task; tools: read, search_files, file_write (full-content writes; you must read an existing file before overwriting it), manage_files (create empty file/folder, delete, update_permission), move (rename/relocate via current_path/new_path), replace_all (replace every occurrence of a literal string — point it at a file or a directory), run_script (executes a .ts file with Deno, full permissions). memory, web_search, programming_docs_search for research.

All file paths passed to tools must be absolute.

Files SHOULD be created and modified inside {workspace} unless the task says otherwise — it is a convention, not an enforced boundary.

Prefer replace_all for targeted edits and file_write for full rewrites; verify your own work by running it before declaring done; return a concise summary of files touched, what the code does, what running produced; no conversation history, work only from the task."""
