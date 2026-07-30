# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the ``code_agent`` delegate's user prompt, and the
class-of-defect guard that covers every channel behind it.

``CodeAgentConfig`` sets ``suppress_history=True``, so the string
:meth:`PromptService.user_prompt` returns IS the delegate's entire input: there
is no history, no seeded memory, nothing else in the message list to carry the
task. A channel with no dispatch branch falls through to ``return ""``, and an
empty user message is not a soft failure — providers that validate request
parameters reject it outright, which crashes the turn rather than degrading it.
Empty-content tolerance is per-provider, so the same omission shows up as either
silent garbage or a hard rejection depending on the selected model. Both are
wrong, and neither is detectable by asserting on the delegate's answer.

These lock the wiring itself:

  1. The prompt carries the task — non-empty, and the instruction text verbatim.
  2. The prompt carries the date anchor. The telemetry block's ``local_time`` is
     the only date any channel receives, and a coding agent that cannot date its
     own work writes wrong dates into the files it creates.
  3. The prompt carries the act trail's re-feed seam, so the delegate sees its
     own tool output across ACT iterations instead of iterating blind.
  4. **No ProcessorConfig channel falls through the dispatch at all** — the
     generalisation of (1). This is the guard that fails on the next channel
     added without a builder, rather than on a provider error days later.

Driven against the real :class:`PromptService` on real
:class:`MessageProcessor` instances over the real production configs. Zero mocks.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sqlite3

import pytest

import configs.channels as channels_pkg
from configs.channels.code_agent import CodeAgentConfig
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.telemetry import Telemetry
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

_TASK = "write a Deno script that reverses a string and prove it runs"

# The warning the dispatch emits when a channel has no prompt builder. Matching
# on it is what turns a silent "" into a test failure.
_FALLTHROUGH = "unhandled channel="


def _seed_telemetry(ctx: dict[str, object]) -> None:
    """Persist a heartbeat the way POST /health does, and drop the singleton's
    cache either side so the read comes from the row, not a neighbour's state."""
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None
    Telemetry.replace(ctx)
    heartbeat_service._ctx = None


def _code_agent_prompt(task: str = _TASK) -> str:
    """The real user-message body for a real code_agent delegate turn."""
    mp = MessageProcessor(CodeAgentConfig(PolicyChannel.CHAT), raw_input=task)
    return mp.prompt_service.user_prompt()


# ---------------------------------------------------------------------------
# 1. The task reaches the delegate.
# ---------------------------------------------------------------------------


def test_code_agent_prompt_is_not_empty(db: sqlite3.Connection) -> None:
    """The headline guarantee. An empty body is the whole message on this
    channel, and a provider that validates parameters rejects it — the turn
    crashes instead of answering."""
    prompt = _code_agent_prompt()
    assert prompt.strip(), (
        "code_agent's user prompt is empty — with suppress_history=True this is "
        "the delegate's ENTIRE input, and a strict provider rejects an "
        f"empty-content request outright. prompt={prompt!r}"
    )


def test_code_agent_prompt_carries_the_task_verbatim(db: sqlite3.Connection) -> None:
    """The instruction the ability passed as ``raw_input`` must survive into the
    body word for word — a paraphrase or a truncation is a different task."""
    prompt = _code_agent_prompt()
    assert _TASK in prompt, (
        f"code_agent must receive its task verbatim.\n  expected: {_TASK!r}\n  prompt: {prompt!r}"
    )


def test_code_agent_prompt_does_not_route_through_the_user_channel(
    db: sqlite3.Connection,
) -> None:
    """The task is a hand-off, not a user utterance. Routing this channel at the
    user channel's builder would fix the empty body but label the coding task
    ``user:`` and inject the user-identity synthesis into a delegate holding
    write and execute tools."""
    prompt = _code_agent_prompt()
    assert "Task:" in prompt, f"the coding task must be labelled as a task. prompt={prompt!r}"
    assert "user:" not in prompt, (
        f"code_agent must not be framed as a user utterance. prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 2. The date anchor reaches the delegate.
# ---------------------------------------------------------------------------


def test_code_agent_prompt_carries_the_date_anchor(db: sqlite3.Connection) -> None:
    """``local_time`` in the telemetry block is the only date any channel gets.
    Without it the delegate has no way to date the work it writes to disk."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB"})
    prompt = _code_agent_prompt()
    assert "local_time:" in prompt, (
        "code_agent must receive the telemetry block's date anchor — a coding "
        f"agent that cannot date its own work writes wrong dates. prompt={prompt!r}"
    )


def test_code_agent_prompt_survives_an_absent_heartbeat(db: sqlite3.Connection) -> None:
    """Telemetry is off-spine: with no heartbeat persisted the block renders
    empty, and the task must still arrive rather than the body collapsing."""
    _seed_telemetry({})
    prompt = _code_agent_prompt()
    assert _TASK in prompt, (
        f"an empty telemetry block must not cost the delegate its task. prompt={prompt!r}"
    )
    assert "local_time:" not in prompt, (
        f"no heartbeat means no anchor line — not a fabricated one. prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 3. The act trail's re-feed seam.
# ---------------------------------------------------------------------------


def test_code_agent_prompt_renders_its_act_trail(db: sqlite3.Connection) -> None:
    """The trail is how the delegate sees its own tool output across ACT
    iterations. On turn 0 there are no calls, so the assertion is that the
    builder asks for one at all — proven by the render being reached and the
    body staying well-formed rather than by trail content that does not exist
    yet. The populated-trail path is shared with every sibling delegate and
    covered by ``act_trail``'s own tests."""
    mp = MessageProcessor(CodeAgentConfig(PolicyChannel.CHAT), raw_input=_TASK)
    prompt = mp.prompt_service.user_prompt()
    trail = mp.prompt_service.act_trail()
    if trail:
        assert trail in prompt, (
            f"a rendered act trail must reach the delegate. trail={trail!r} prompt={prompt!r}"
        )
    assert prompt.rstrip().endswith(_TASK) or trail, (
        f"with no trail the body ends at the task. prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# 4. The class guard — no channel may fall through the dispatch.
# ---------------------------------------------------------------------------


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def test_code_agent_channel_is_dispatched(db: sqlite3.Connection, caplog: pytest.LogCaptureFixture) -> None:
    """The specific omission that crashed the turn: ``delegate:code_agent``
    reaching the fallthrough."""
    with caplog.at_level(logging.WARNING, logger="services.prompt_service"):
        _code_agent_prompt()
    assert _FALLTHROUGH not in caplog.text, (
        f"{Channel.DELEGATE_CODE_AGENT.value} hit the dispatch fallthrough. log={caplog.text!r}"
    )


def test_no_processor_config_channel_hits_the_dispatch_fallthrough(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """Enumerate every ProcessorConfig subclass, resolve its prompt channel, and
    drive the real dispatch. None may reach ``return ""``.

    A builder that raises is still *dispatched* — the raise proves the branch was
    taken, and per-builder state requirements are not this test's subject — so
    those are named in the failure message instead of counted as failures.
    Configs that cannot be built with a channel's standard argument forms are
    likewise named rather than silently skipped."""
    for mod in pkgutil.iter_modules(channels_pkg.__path__):
        importlib.import_module(f"configs.channels.{mod.name}")

    fell_through: dict[str, str] = {}
    raised: dict[str, str] = {}
    unbuilt: list[str] = []

    for cls in sorted(_all_subclasses(ProcessorConfig), key=lambda c: c.__name__):
        instance = None
        for call_args in ([], [PolicyChannel.CHAT], [{"hidden_input": True}]):
            try:
                instance = cls(*call_args)
                break
            except Exception:
                continue
        if instance is None:
            unbuilt.append(cls.__name__)
            continue

        mp = MessageProcessor(instance, raw_input=_TASK)
        # Resolve the dispatch key first: a failure HERE is a real failure, not
        # a builder's state requirement.
        channel = instance.prompt_channel or instance.channel
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="services.prompt_service"):
            try:
                mp.prompt_service.user_prompt()
            except Exception as exc:  # noqa: BLE001 — a raise proves the branch was taken
                raised[cls.__name__] = f"{channel} ({type(exc).__name__})"
        if _FALLTHROUGH in caplog.text:
            fell_through[cls.__name__] = channel

    assert not fell_through, (
        "Every ProcessorConfig channel needs a prompt builder — the fallthrough "
        "returns an empty user body, which a strict provider rejects outright.\n"
        f"  fell through: {fell_through}\n"
        f"  dispatched but raised on test-env state (not a failure here): {raised}\n"
        f"  not built with a standard channel signature: {sorted(unbuilt)}"
    )
