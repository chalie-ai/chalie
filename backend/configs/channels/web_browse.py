# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""WebBrowseConfig — the delegate channel for the ``web_browse`` tool.

Paired with ``WebBrowseAbility`` (abilities/web_browse.py). Drives the rebuilt
9-verb ``browser`` tool plus ``read``, ``vision`` and ``memory`` in a clean
cross-turn context. It writes a real per-turn transcript row on its own
``delegate:web_browse`` channel so the turn uid is assigned and the delegate
renders its own act-trail across ACT iterations (do NOT set the two
skip flags True). That same uid keys the per-run browser PageSession and the
screenshot ledger; the post-turn hook closes both when the run ends.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._delegate import render_trail
from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig
from tools.browser.session import close_session, screenshot_ledger

_WEB_BROWSE_SYSTEM_PROMPT = (
    "You are a web-browsing agent with one goal, given below. You drive a real "
    "browser through the `browser` tool: open a page, read or search it, click "
    "and fill what you need by visible text, then read the result. Every call "
    "returns JSON describing the page and what changed — trust it over your "
    "assumptions, and never invent content, URLs, or results.\n\n"
    "Screenshots are saved as documents; use the `vision` tool with the "
    "returned doc_id to see one. Use `memory` to recall user preferences when "
    "the task needs them. If a page demands a login or CAPTCHA, report that "
    "plainly instead of trying to get past it.\n\n"
    "STOP RULE: the moment you can answer the goal — or know you cannot — stop "
    "calling tools and give your final answer, citing the pages you actually "
    "visited."
)

_WEB_BROWSE_TOOLS: tuple[str, ...] = ("browser", "read", "vision")


class _CloseBrowserSession(PostTurnHook):
    """End-of-run cleanup: close the delegate's browser tab + screenshot ledger.

    Stashes the ledger on itself first — ``close_session`` pops it, but the
    outer ``WebBrowseAbility`` still needs the doc_ids after ``process()``
    returns, to hand them to the caller alongside the delegate's answer."""

    def __init__(self) -> None:
        self.final_ledger: list[tuple[str, str]] = []

    def run(self, mp, result_text: str) -> None:  # noqa: ARG002 — hook signature
        uid = getattr(mp, "uid", None)
        if uid:
            self.final_ledger = screenshot_ledger(uid)
            close_session(uid)


class WebBrowseConfig(ProcessorConfig):
    """ProcessorConfig for the web_browse delegate. ``policy_channel`` is
    inherited from the caller that invoked the tool; the user-facing permission
    check happens at the outer ``web_browse`` tool."""

    uses_delegate_provider: ClassVar[bool] = True

    def __init__(self, policy_channel: "ProcessorConfig.POLICY_CHANNEL") -> None:
        super().__init__(
            channel="delegate:web_browse",
            role="web_browse",
            policy_channel=policy_channel,
            always_available=[*_WEB_BROWSE_TOOLS, "memory"],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=200,
            skip_transcript=False,  # uid + own transcript row, or the
            skip_input_row=False,   # act-trail dies and the loop runs blind
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn_hooks=(_CloseBrowserSession(),),
        )

    def final_screenshots(self) -> list[tuple[str, str]]:
        """The ``(doc_id, url)`` pairs captured by the finished run.

        Populated by the post-turn hook just before it closes the session;
        empty until the run ends (or when no screenshots were taken)."""
        return self.post_turn_hooks[0].final_ledger

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        """Goal + mechanical screenshot ledger + act-trail.

        The ledger line is rebuilt from session state on EVERY iteration, so a
        mid-run act-trail compaction can never lose a screenshot doc_id (the
        compactor handover is LLM-written and probabilistic; this is not)."""
        parts = [f"Browsing goal:\n{mp._raw_input}"]  # type: ignore[attr-defined]
        shots = screenshot_ledger(getattr(mp, "uid", None) or 0)
        if shots:
            lines = "\n".join(f"- doc_id={doc_id} ({url})" for doc_id, url in shots)
            parts.append(f"Screenshots captured this run (view with the vision tool):\n{lines}")
        trail = render_trail(mp)
        if trail:
            parts.append(trail)
        return "\n\n".join(parts)

    def get_system_prompt(self, mp) -> str:
        return _WEB_BROWSE_SYSTEM_PROMPT
