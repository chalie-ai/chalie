# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Web-delegate-ability-specific business-logic tests migrated from the
per-ability conformance file removed in TKT-975.

Pins TKT-898's failure-half contract for web_search and web_browse: the
missing-param pre-gate fires before a delegate is spawned, an empty delegate
answer maps to ``delegate-no-answer``, and a real prose answer is returned
verbatim as success.

NOTE: a file ``test_web_delegate_act_trail.py`` already exists covering the
act-trail surface. This file is named ``test_web_delegates.py`` and covers the
distinct param-gate + delegate-result-mapping contract.
"""

import pytest

from abilities._delegate import delegate_result
from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import MP, body, head, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test db, with a seeded transcript anchor
    so the dispatcher's act-trail record has a uid to key against."""
    return MP(seed_transcript(db, content="research something"), UserConfig({}))


# ── Param gate: missing query/goal is rejected BEFORE a delegate spawns ─────────


@pytest.mark.parametrize(
    "tool,param,params",
    [
        ("web_search", "query", {}),
        ("web_search", "query", {"query": ""}),
        ("web_browse", "goal", {}),
        ("web_browse", "goal", {"goal": ""}),
    ],
)
def test_missing_param_blocked_before_delegate(db, chat_mp, tool, param, params):
    """A missing/empty required param renders ``code=missing-params`` through the
    real dispatch chokepoint. The pre-gate fires before the policy gate and before
    run(), so no ``MessageProcessor.process`` delegate is ever spawned — this test
    needs no provider and cannot hang. At HEAD (no ACTION_REQUIRED on either tool)
    the empty param flowed into run() and spawned a real delegate loop."""
    rendered = ToolDispatcher(chat_mp).dispatch(tool, dict(params))

    open_tag = head(rendered, tool)
    assert "status=error" in open_tag
    assert "code=missing-params" in open_tag
    assert param in body(rendered, tool)
    assert f"valid: {param}" in rendered


# ── Empty / non-empty delegate answer → err / ok mapping ────────────────────────


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_empty_delegate_answer_is_no_answer_error(empty):
    """An empty (or whitespace-only) delegate answer maps to an ERROR with the
    stable ``delegate-no-answer`` code and a self-correction hint — never a blank
    ``ok("")`` the outer model silently trusts."""
    tr = delegate_result(empty, hint="Narrow the query, then retry.")

    assert tr.status == "error"
    assert tr.code == "delegate-no-answer"
    assert tr.hint == "Narrow the query, then retry."


def test_real_delegate_answer_is_success_verbatim():
    """A real prose answer is returned as success with the body verbatim — the
    delegate's synthesis is never reshaped or summarised by the mapping."""
    answer = "Malta's population is about 540,000 (2024 estimate)."
    tr = delegate_result(answer, hint="Narrow the query, then retry.")

    assert tr.status == "success"
    assert tr.code is None
    assert tr.body == answer
