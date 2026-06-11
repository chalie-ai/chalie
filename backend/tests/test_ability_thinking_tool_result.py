# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the thinking ability's ToolResult contract (TKT-916).

The behavioural migration to the ``ToolResult`` type already shipped in TKT-882:
``ThinkingAbility.run`` returns ``ToolResult.ok(<prose>)`` for a deliberation and
``ToolResult.ok("")`` for the ``NOTHING`` sentinel. THIS file is a type-only,
already-green pinning layer: it nails down what the PARENT LOOP SEES coming back
out of the dispatch — the rendered wire envelope.

Division of labour with the sibling suite: request-mirror semantics (the thinking
pass sends the parent's verbatim user message, tool surface, and the lean
deliberation system overlay — TKT-833/TKT-838) are pinned by
``test_thinking_history_carryforward.py``. This file pins the OUTPUT side: the
deliberation prose must survive into the envelope byte-for-byte (no JSON wrapping
around reasoning text), the ``NOTHING`` sentinel must collapse to an empty body,
and the rendered result must be the row recorded onto the parent's act-trail.

Real hot path, zero internal mocks: the genuine ``ToolDispatcher(parent).dispatch``
chokepoint, the real ``AbilityRegistry`` resolution of the production
``ThinkingAbility``, the real ``PolicyManager`` gate (``thinking`` is in
``PolicyManager.INTERNAL`` so the gate short-circuits straight to run()), a real
single-pass ``MessageProcessor.process`` deliberation loop, real SQLite (``db``),
the real transcript factory (``write_input_row``), and the real ``ActTrail`` write.
The ONLY stand-in is the external-LLM boundary (``Providers._resolve``) — the
single sanctioned seam — here a recording provider whose reply text is configurable
so each test controls exactly what the deliberation "thinks".
"""

from unittest.mock import patch

import pytest

from services.provider_api import ProviderApiResponse

pytestmark = pytest.mark.unit

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _RecordingProvider:
    """Stand-in for the resolved LLM provider — the single sanctioned boundary.

    Returns a one-shot ``ProviderApiResponse`` whose ``text`` is whatever
    ``reply_text`` the test supplies (no tool calls, so the single-pass thinking
    ACT loop ends after one send). ``estimate_request_tokens`` returns 1 so the
    pre-flight over-cap check never triggers; the context-limit / content-label
    members mirror the real ProviderClient surface the request builder reads.
    """

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text

    def get_context_limit(self):
        return 200000

    def estimate_request_tokens(self, dto):
        return 1

    def send(self, dto):
        return ProviderApiResponse(text=self._reply_text, model="recorder", tool_calls=None)


def _build_parent(raw_input: str):
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires the thinking pass from: input row written and ``active_tools`` seeded
    with the channel's always_available tier (message_processor._setup), matching
    the template in ``test_thinking_history_carryforward.py``."""
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor
    from services.transcript_service import write_input_row

    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, raw_input, None)
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", raw_input)
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def _dispatch_thinking(parent, reply_text: str) -> str:
    """Drive the real turn-0 dispatch with a recorder that returns *reply_text*."""
    from abilities._dispatcher import ToolDispatcher

    recorder = _RecordingProvider(reply_text)
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        return ToolDispatcher(parent).dispatch("thinking", {})


# ── 1. deliberation prose survives into the envelope byte-for-byte ───────────────


def test_deliberation_prose_is_verbatim_in_envelope(db):
    """Multi-line prose containing quotes/braces/blank lines — the kind of text a
    JSON wrapper would mangle — must appear in the success envelope EXACTLY, with
    no escaping and no meta beyond ``status=success``. This proves the ``str`` body
    is rendered verbatim (the deliberation is Chain-of-Thought the next pass reads,
    not data)."""
    prose = 'Plan:\n- use "search" first\n- then {maybe} timer\n\nKey fact: 42'

    out = _dispatch_thinking(_build_parent("Do the complex thing."), prose)

    assert out == f"[thinking(status=success)]\n{prose}\n[end:thinking]"


# ── 2. NOTHING sentinel → empty body line ────────────────────────────────────────


def test_nothing_sentinel_yields_empty_body(db):
    """The ``NOTHING`` sentinel collapses to an empty body — ``ToolResult.ok("")``
    rendered as ``[thinking(status=success)]\\n\\n[end:thinking]`` (the body line
    is empty, the envelope still well-formed)."""
    out = _dispatch_thinking(_build_parent("Trivial request."), "NOTHING")

    assert out == "[thinking(status=success)]\n\n[end:thinking]"


# ── 3. NOTHING sentinel is case- and whitespace-insensitive ──────────────────────


def test_nothing_sentinel_is_case_insensitive(db):
    """run() does ``text.strip().upper() == "NOTHING"`` — so lowercase and
    surrounding whitespace still resolve to the empty-body envelope."""
    lowercase = _dispatch_thinking(_build_parent("Trivial A."), "nothing")
    assert lowercase == "[thinking(status=success)]\n\n[end:thinking]"

    padded = _dispatch_thinking(_build_parent("Trivial B."), "  NOTHING  \n")
    assert padded == "[thinking(status=success)]\n\n[end:thinking]"


# ── 4. the rendered envelope is recorded onto the parent's act-trail ─────────────


def test_result_recorded_in_act_trail(db):
    """The dispatch path records the rendered thinking envelope onto the parent's
    act-trail against its transcript anchor — this is how the deliberation flows
    back into the next ``get_user_prompt`` via ``_render_act_trail``. The distinctive
    token must be present in the recorded row."""
    from services.act_trail import ActTrail

    prose = "Deliberation token XYLOPHONE-DELIBERATION must reach the trail."
    parent = _build_parent("Where are we?")

    out = _dispatch_thinking(parent, prose)

    trail = ActTrail().fetch_by_transcript_id(parent.uid)
    matching = [row for row in trail if row["tool_name"] == "thinking"]
    assert matching, "thinking dispatch recorded no act-trail row"
    recorded = matching[-1]["result"]
    assert recorded == out
    assert "XYLOPHONE-DELIBERATION" in recorded
    assert recorded == f"[thinking(status=success)]\n{prose}\n[end:thinking]"


# ── 5. no legacy markers in the rendered output ──────────────────────────────────


def test_no_legacy_markers(db):
    """The migrated envelope carries none of the pre-TKT-882 markers: no skill-tag
    formatting, no raw JSON ``"status": "success"`` field, and never the banned
    ``code=error`` placeholder (mirrors the sibling tool-result suites)."""
    out = _dispatch_thinking(_build_parent("Plan it."), "A real deliberation line.")

    assert "<skill_tag" not in out
    assert '"status": "success"' not in out
    assert "code=error]" not in out
    assert "code=error," not in out
    assert "code=error)" not in out
    # Positive shape: the canonical success head, prose body verbatim, sealed close.
    assert out.startswith("[thinking(status=success)]\n")
    assert out.endswith("\n[end:thinking]")


# ── 6. frozen parameter surface + registry + INTERNAL gate ───────────────────────


def test_frozen_surface_and_registry():
    """Pin the immovable surface: the empty object schema, the policy-manager
    ``INTERNAL`` gate membership (so the dispatch bypasses the policy prompt), and
    the registry name. ``ThinkingConfig`` mirroring is pinned by the carryforward
    suite, not here."""
    from abilities._registry import AbilityRegistry
    from abilities.thinking import ThinkingAbility
    from services.policy_manager import INTERNAL

    assert ThinkingAbility().get_parameters() == {"type": "object", "properties": {}}
    assert "thinking" in INTERNAL

    registered = AbilityRegistry.get("thinking")
    assert registered.get_name() == "thinking"
    assert isinstance(registered, ThinkingAbility)
