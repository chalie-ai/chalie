# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CodeAgentAbility — delegate a coding task to a focused agent with a scratch workspace.

Mirrors WebSearchAbility/PimAbility structure-for-structure: an ability that
builds its OWN ``ProcessorConfig`` subclass (``CodeAgentConfig``) inside
``run()`` and calls ``MessageProcessor.process()``. The delegate works in a
scratch workspace (the conventional code_agent location — not a containment
boundary) and can execute a written ``.ts`` file through ``run_script`` with
full permissions — replacing the deleted ``code_eval`` one-shot stdin sandbox
with a real, persistent, multi-step coding loop.

Toolset: read, search_files, file_write, manage_files, move, replace_all,
run_script.

The hand-off contract — what the delegate's final answer must contain (script
paths, parameters, expected output) — is a prompt instruction in
``CodeAgentConfig.system_prompt``, so the delegate's own text is the whole
result; ``run()`` passes it through unchanged.
"""

from __future__ import annotations

from typing import ClassVar, cast

from abilities._delegate import DelegateAbility, delegate_result
from abilities._result import ToolResult
from configs.channels.code_agent import CodeAgentConfig
from configs.enums.param_key import Keys
from services.file_mapper_service import FileMapperService


class CodeAgentAbility(DelegateAbility):
    # An action-less delegate: the dispatcher's ACTION_REQUIRED pre-gate (the
    # ``""`` key covers action-less tools) rejects a missing/empty ``instructions``
    # with ``code=missing-params`` BEFORE the policy gate and BEFORE run() — so an
    # empty instruction never spawns an expensive delegate on an empty goal.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.instructions,)}

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.instructions: {
                "type": "string",
                "description": (
                    "A plain-language coding task. E.g. 'write a script that "
                    "computes compound interest and print the result', 'create "
                    "a file with this data and a script that summarizes it'."
                ),
            },
        },
        "required": [Keys.instructions],
    }

    def get_name(self) -> str:
        return "code_agent"

    def get_summary(self) -> str:
        return (
            "Delegate a coding task to a focused agent with its own scratch "
            "workspace and a TypeScript runtime. It can read, search, write, "
            "manage, move, replace, and run scripts — any path is absolute, "
            "the workspace is just a conventional home, not a containment "
            "boundary. Use for anything that needs real code written to disk "
            "and executed, not a one-line calculation."
        )

    def get_examples(self) -> list[str]:
        return [
            "write a script that parses this CSV and computes the totals",
            "create a small TypeScript project that fetches and summarizes an API",
            "build a script to validate these JSON files against a schema",
            "write and run a script that renames these files by a pattern",
            "create a file with this data and write a script to analyze it",
            "refactor the function in this file and verify it still works",
            "write a script that scrapes this page and saves the results to a file",
            "build a small tool that transforms this data and run it",
        ]

    def get_search_tooltip(self) -> str:
        return "delegate a coding task with a file workspace"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError("code_agent.run() dispatched without a bound MessageProcessor")

        # Ensure the scratch workspace exists before the delegate runs.
        FileMapperService.get_code_agent_workspace_path().mkdir(
            parents=True, exist_ok=True
        )

        agent_mp = MessageProcessor.process(
            CodeAgentConfig(mp.config.policy_channel),
            raw_input=cast(str, self.param(params, Keys.instructions, required=True)),
        )
        return delegate_result(
            agent_mp.result(),
            hint="Break the task into smaller steps or clarify the requirements, then retry.",
        )
