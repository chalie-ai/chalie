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
task. This channel once had no dispatch arm at all, and the fallthrough returned
an empty body — which is not a soft failure. Empty-content tolerance is
per-provider: a lenient provider answers plausible nonsense, a strict one rejects
the request and the resulting crash names the vendor instead of the omission.
Both are wrong, and neither is detectable by asserting on the delegate's answer.

These lock the wiring itself:

  1. The prompt carries the task — non-empty, and the instruction text verbatim.
  2. The prompt starts with the turn's own ``[Ddd YYYY-MM-DD HH:MM] Task:``
     stamp — the delegate's only anchor for dating the work it writes to
     disk — and carries no World State block at all: a delegate holding
     write and execute tools has no business receiving the user's device
     telemetry.
  3. The prompt carries the act trail's re-feed seam, so the delegate sees its
     own tool output across ACT iterations instead of iterating blind.
  4. An unrouted channel raises :class:`UnroutedPromptChannel` — the fallthrough
     no longer manufactures a contentless message for the provider to judge.
  5. **No ProcessorConfig channel reaches that fallthrough at all** — the
     generalisation of (1). This is the guard that fails on the next channel
     added without a builder, rather than on a provider error days later.

Driven against the real :class:`PromptService` on real
:class:`MessageProcessor` instances over the real production configs. Zero mocks.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sqlite3

import pytest

import configs.channels as channels_pkg
from configs.channels.code_agent import CodeAgentConfig
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from exceptions import UnroutedPromptChannel
from services.processor_config import ProcessorConfig
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit

_TASK = "write a Deno script that reverses a string and prove it runs"


def _seed_telemetry(ctx: dict[str, object]) -> None:
    """Persist a heartbeat snapshot the way POST /health does."""
    from services.telemetry_service import TelemetryService
    TelemetryService.write(ctx)


def _code_agent_prompt(task: str = _TASK) -> str:
    """The real user-message body for a real code_agent delegate turn."""
    mp = MessageProcessor(CodeAgentConfig(PolicyChannel.CHAT), raw_input=task)
    return mp.prompt_service.user_prompt()


def _open_code_agent_turn(task: str = _TASK) -> MessageProcessor:
    """Build a code_agent delegate turn the way ``CodeAgentAbility``'s real
    invocation does — with a real input row and a real execution row, since
    ``skip_input_row=False`` for this config (see configs/channels/code_agent.py)
    — so the prompt's stamp resolves to a real timestamp instead of falling
    through ``_input_stamp``'s chain to the missing-timestamp placeholder.
    The other tests in this file don't depend on either row and use the
    lighter ``_code_agent_prompt`` helper instead."""
    mp = MessageProcessor(CodeAgentConfig(PolicyChannel.CHAT), raw_input=task)
    mp.turn_id = mp.transcript_service.allocate_turn()
    mp.uid = mp.transcript_service.append_input(mp.raw_input)
    mp.current_transcript_id = mp.uid
    mp.turn_execution_service.open()
    return mp


# ---------------------------------------------------------------------------
# 1. The task reaches the delegate.
# ---------------------------------------------------------------------------


def test_code_agent_prompt_is_not_empty(db: sqlite3.Connection) -> None:
    """The headline guarantee. With ``suppress_history=True`` this body is the
    whole message, so a builder that returns nothing is as bad as no builder at
    all — and unlike the missing arm, it would not raise."""
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
# 2. The task line is stamped, and the delegate carries no world block.
# ---------------------------------------------------------------------------


def test_code_agent_prompt_starts_with_the_stamped_task_line(db: sqlite3.Connection) -> None:
    """The whole body is ``[stamp] Task:\\n<task>`` (+ trail): the stamp is
    the delegate's only anchor for dating the work it writes to disk, taken
    from the turn's own input row. A coding agent that cannot date its own
    work writes wrong dates into the files it creates."""
    mp = _open_code_agent_turn()
    prompt = mp.prompt_service.user_prompt()
    match = re.match(r"^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}\] Task:\n", prompt)
    assert match, (
        "code_agent's prompt must start with a [Ddd YYYY-MM-DD HH:MM] Task: "
        f"stamp line. prompt={prompt!r}"
    )
    assert prompt[match.end():].startswith(_TASK), (
        f"the task text must follow the stamp line verbatim. prompt={prompt!r}"
    )


def test_code_agent_prompt_carries_no_world_block(db: sqlite3.Connection) -> None:
    """The delegate holds write and execute tools; the user's device
    telemetry (battery, location, focus state) has no business reaching it.
    Seeding a heartbeat proves the block is omitted by design, not merely
    absent for lack of data."""
    _seed_telemetry({"timezone": "Europe/Malta", "locale": "en-GB", "local_time": "10:47"})
    prompt = _code_agent_prompt()
    assert "### Background Telemetry,Processes" not in prompt, (
        f"code_agent must never receive the World State block. prompt={prompt!r}"
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
# 4. The fallthrough is loud — and no shipped channel can reach it.
# ---------------------------------------------------------------------------


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def test_an_unrouted_channel_raises_instead_of_returning_an_empty_body(
    db: sqlite3.Connection,
) -> None:
    """The fallthrough must name the wiring error. A channel with no arm is a
    config shipped without its dispatch branch — returning ``""`` handed that
    omission to the provider, which either answered nonsense or rejected the
    request with a message pointing at itself."""
    mp = MessageProcessor(make_stub_config(channel="delegate:not_wired_yet"), raw_input=_TASK)
    with pytest.raises(UnroutedPromptChannel) as caught:
        mp.prompt_service.user_prompt()
    assert caught.value.channel == "delegate:not_wired_yet"
    assert "delegate:not_wired_yet" in str(caught.value), (
        f"the unrouted channel must be named in the crash reason. exc={caught.value!s}"
    )


def test_code_agent_channel_is_dispatched(db: sqlite3.Connection) -> None:
    """The specific omission that crashed the turn: ``delegate:code_agent``
    reaching the fallthrough."""
    try:
        _code_agent_prompt()
    except UnroutedPromptChannel as exc:  # pragma: no cover — the defect under test
        pytest.fail(f"{Channel.DELEGATE_CODE_AGENT.value} hit the dispatch fallthrough: {exc}")


def test_no_processor_config_channel_hits_the_dispatch_fallthrough(
    db: sqlite3.Connection,
) -> None:
    """Enumerate every ProcessorConfig subclass, build it, and drive the real
    dispatch. None may reach the fallthrough.

    Only :class:`UnroutedPromptChannel` counts as a failure. Any other exception
    proves the branch WAS taken and the builder then tripped on state this tier
    does not set up — not this test's subject — so those are named in the failure
    message instead. Configs that cannot be built with a channel's standard
    argument forms are likewise named rather than silently skipped."""
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
        # Resolve the dispatch key up front: it names the channel in the report
        # even when the builder itself blows up on unseeded state.
        channel = instance.prompt_channel or instance.channel
        try:
            mp.prompt_service.user_prompt()
        except UnroutedPromptChannel:
            fell_through[cls.__name__] = channel
        except Exception as exc:  # noqa: BLE001 — a raise elsewhere proves the branch was taken
            raised[cls.__name__] = f"{channel} ({type(exc).__name__})"

    assert not fell_through, (
        "Every ProcessorConfig channel needs a prompt builder — reaching the "
        "fallthrough crashes the turn, and used to send a contentless message.\n"
        f"  fell through: {fell_through}\n"
        f"  dispatched but raised on test-env state (not a failure here): {raised}\n"
        f"  not built with a standard channel signature: {sorted(unbuilt)}"
    )
