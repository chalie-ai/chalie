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

After the sub-loop finishes, ``run()`` deterministically appends a hand-off
footer naming every ``.ts`` file the run mutated and the exact command to run
it. This is wiring, not a prompt instruction: it is built from the delegate's
own tool-call trail (``ToolCallService.by_turn()`` — the same post-hoc read
``MessageProcessor._touched_pattern_names`` uses), so it can never be
dropped, hallucinated, or phrased inconsistently by the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._delegate import DelegateAbility, delegate_result
from abilities._result import ToolResult
from abilities.run_script import build_run_command
from configs.channels.code_agent import CodeAgentConfig
from configs.enums.param_key import Keys
from models.tool_call import ToolCall
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


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

    # The 4 code_agent inner tools that mutate the workspace (mirrors
    # CodeAgentConfig._INNER_TOOLS's file-mutating subset) and the hand-off
    # statuses their successful calls can produce.
    _MUTATION_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {"manage_files", "file_write", "move", "replace_all"}
    )
    _STATUS_CREATED: ClassVar[str] = "created"
    _STATUS_MODIFIED: ClassVar[str] = "modified"
    _STATUS_DELETED: ClassVar[str] = "deleted"
    _STATUS_MOVED: ClassVar[str] = "moved"

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
        text = self._append_handoff_footer(agent_mp.result(), agent_mp)
        return delegate_result(
            text, hint="Break the task into smaller steps or clarify the requirements, then retry."
        )

    # ── Deterministic hand-off footer ──────────────────────────────────────

    def _append_handoff_footer(self, text: str, agent_mp: "MessageProcessor") -> str:
        """Append one line per ``.ts`` file this run mutated, or return *text*
        unchanged when nothing was mutated. Isolated in its own try/except
        (mirrors ``MessageProcessor._post_turn``'s per-handler isolation): a bug
        in footer construction must never discard the delegate's already-
        successful text.
        """
        try:
            lines = self._footer_lines(agent_mp)
        except Exception:
            logger.warning("[code_agent] hand-off footer construction failed", exc_info=True)
            return text
        if not lines:
            return text
        footer = "\n".join(lines)
        return f"{text}\n\n{footer}" if text.strip() else footer

    def _footer_lines(self, agent_mp: "MessageProcessor") -> list[str]:
        """Render one footer line per mutated ``.ts`` path, in first-mutated
        order. Paths from the new toolset are already absolute (no workspace
        containment to validate), so no ValueError-skip is needed.
        """
        mutations = self._collect_mutations(agent_mp)
        return [self._render_line(abs_path, status) for abs_path, status in mutations.items()]

    def _collect_mutations(self, agent_mp: "MessageProcessor") -> dict[str, str]:
        """Replay this run's successful mutation calls, in call order, into an
        ordered ``absolute_path -> status`` map (``created``/``modified``/
        ``deleted``/``moved``). Only ``state=DONE`` (``ToolResult.ok``) calls
        count — a refused or errored call did not mutate anything. Non-``.ts``
        paths are dropped: the hand-off contract is about runnable scripts.
        """
        mutations: dict[str, str] = {}
        for call in agent_mp.tool_call_service.by_turn():
            if call.state != ToolCall.DONE or call.tool_name not in self._MUTATION_TOOLS:
                continue
            try:
                call_params = json.loads(call.params)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(call_params, dict):
                continue

            tool = call.tool_name
            if tool == "manage_files":
                self._handle_manage_files(mutations, call_params)
            elif tool == "file_write":
                self._handle_file_write(mutations, call_params, call.result)
            elif tool == "move":
                self._handle_move(mutations, call_params)
            elif tool == "replace_all":
                for abs_path in self._replace_all_paths(call.result, call_params):
                    self._mark(mutations, abs_path, self._STATUS_MODIFIED)
        return mutations

    def _handle_manage_files(
        self,
        mutations: dict[str, str],
        call_params: dict[str, object],
    ) -> None:
        """manage_files carries an ``action`` param; the action determines the
        mutation status. ``update_permission`` is a no-op for the footer."""
        action = call_params.get(Keys.action)
        if action == "create":
            self._mark(mutations, call_params.get(Keys.path), self._STATUS_CREATED)
        elif action == "delete":
            self._mark(mutations, call_params.get(Keys.path), self._STATUS_DELETED)
        # update_permission → ignore (no .ts tracking impact)

    def _handle_file_write(
        self,
        mutations: dict[str, str],
        call_params: dict[str, object],
        rendered_result: str,
    ) -> None:
        """file_write's ToolResult body is a dict ``{"path", "bytes", "created"}``
        rendered as compact JSON (see ``abilities/_result.py``: a ``dict`` body is
        rendered as compact JSON in the wire envelope). Parse the body via the
        existing ``_envelope_body`` helper to recover the dict, then read the
        ``created`` flag: ``true`` → ``created``, ``false`` or any parse failure →
        ``modified``."""
        body = self._envelope_body(rendered_result, "file_write")
        try:
            result_dict = json.loads(body)
            if isinstance(result_dict, dict) and result_dict.get("created") is True:
                status = self._STATUS_CREATED
            else:
                status = self._STATUS_MODIFIED
        except (json.JSONDecodeError, TypeError):
            status = self._STATUS_MODIFIED
        self._mark(mutations, call_params.get(Keys.path), status)

    def _handle_move(self, mutations: dict[str, str], call_params: dict[str, object]) -> None:
        """move carries ``current_path`` and ``new_path`` (Keys.current_path /
        Keys.new_path). Keep the existing _apply_move semantics: drop any tracked
        entry for current_path; new_path gets status ``moved`` only when it ends
        with .ts."""
        self._apply_move(mutations, call_params.get(Keys.current_path), call_params.get(Keys.new_path))

    def _mark(self, mutations: dict[str, str], raw_path: object, status: str) -> None:
        """Record ``raw_path -> status``, dropping non-``.ts`` paths, unless the
        path is already ``created`` and *status* is ``modified`` — a file created
        and later modified in the same run reports once, as created (Dylan's
        explicit collapse rule)."""
        if not isinstance(raw_path, str) or not raw_path.endswith(".ts"):
            return
        if status == self._STATUS_MODIFIED and mutations.get(raw_path) == self._STATUS_CREATED:
            return
        mutations[raw_path] = status

    def _apply_move(self, mutations: dict[str, str], raw_source: object, raw_destination: object) -> None:
        """A move drops any tracked entry for the old path outright and, only
        when the destination is itself a ``.ts`` file, adds a ``moved`` entry for
        the new path — moved to a non-``.ts`` destination, the file is no longer
        a runnable script and gets no line at all."""
        if isinstance(raw_source, str):
            mutations.pop(raw_source, None)
        if isinstance(raw_destination, str) and raw_destination.endswith(".ts"):
            mutations[raw_destination] = self._STATUS_MOVED

    def _replace_all_paths(self, rendered_result: str, call_params: dict[str, object]) -> list[str]:
        """Recover the per-file paths a successful ``replace_all`` call touched.

        ``replace_all`` takes an optional ``path`` param (a FILE or DIRECTORY);
        when a directory is scanned, body paths are RELATIVE to that root, so
        absolute-ize each by joining with the call's ``path`` param when present,
        else with the code_agent workspace path. When the call targeted a single
        file, body paths are absolute as-passed — the existing
        ``Path(directory) / path`` join handles both cases (joining an absolute
        path onto a base yields the absolute path unchanged).
        """
        body = self._envelope_body(rendered_result, "replace_all")
        directory = call_params.get(Keys.path)
        if not isinstance(directory, str):
            directory = str(FileMapperService.get_code_agent_workspace_path())
        paths: list[str] = []
        for line in body.splitlines():
            path, sep, count = line.rpartition(": ")
            if sep and path and count.strip().isdigit():
                paths.append(str(Path(directory) / path))
        return paths

    def _envelope_body(self, rendered_result: str, tool_name: str) -> str:
        """Strip the ``[tool(...)]`` header and the ``[end:tool]`` tail (and
        anything a loop-guard steer appended after it) from a rendered wire
        envelope, returning only the body text in between."""
        end_marker = f"[end:{tool_name}]"
        end_idx = rendered_result.find(end_marker)
        head = rendered_result[:end_idx] if end_idx != -1 else rendered_result
        _, _, body = head.partition("\n")
        return body

    def _render_line(self, abs_path: str, status: str) -> str:
        """Render one footer line for *abs_path*. All paths reaching here are
        already absolute (no workspace containment — the workspace is just the
        conventional location)."""
        if status == self._STATUS_DELETED:
            return f"Script deleted at `{abs_path}`."
        cmd = build_run_command(abs_path)
        if status == self._STATUS_CREATED:
            return f"Script created at `{abs_path}`. To execute it, run: `{cmd}`"
        if status == self._STATUS_MOVED:
            return f"Script moved to `{abs_path}`. To execute it, run: `{cmd}`"
        return f"Script modified at `{abs_path}`. To execute it, run: `{cmd}`"
