# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Integration feature test: SearchAbility.run() produces the new XML-record format.

Drives the real ability through the real production entry point (ToolDispatcher)
exactly as the ACT loop does, against a real converged DB.  No mocks.

The assertion: the rendered body returned by the dispatcher contains the
``<result index=`` and ``score=`` attributes and the ``summary:`` field that
the new ``render_records`` formatter emits.

Baseline-fail proof: the current search implementation returns a JSON list
(``[{"title":…, "url":…, "snippet":…, "source":…}]``), not ``<result index=``
blocks.  The test therefore fails with AssertionError on unmodified code.

Marked ``integration`` so it is excluded from ``-m unit`` CI gates (it makes
live provider network calls).
"""

import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import MP, seed_transcript

pytestmark = pytest.mark.integration


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    """Real MessageProcessor stub — same factory the existing search tests use."""
    return MP(seed_transcript(db, "chat", "search for something"), UserConfig({}))


def test_search_run_returns_xml_record_format_with_score_and_summary(db: sqlite3.Connection, chat_mp: MP) -> None:
    """SearchAbility.run() must return a body that uses the new <result index=> format.

    This drives the FULL production dispatch path (ToolDispatcher → PolicyManager
    → SearchAbility.run → render_records) and asserts on the string that the ACT
    loop receives — the same observable effect a real user's turn depends on.
    """
    rendered = ToolDispatcher(chat_mp).dispatch(
        "search",
        {"query": "python asyncio concurrency", "act_summary": "Searching for asyncio"},
    )

    # The rendered envelope must indicate success
    assert "status=success" in rendered, (
        f"Expected a success result, got:\n{rendered[:500]}"
    )

    # The body must use the new XML record format
    assert "<result index=" in rendered, (
        f"Expected <result index=…> blocks in output, got:\n{rendered[:800]}"
    )

    # Results must carry relevance scores (rank_results was applied)
    assert "score=" in rendered, (
        f"Expected score= attribute on result blocks, got:\n{rendered[:800]}"
    )

    # Results must use 'summary:' field (not the legacy 'snippet:' field)
    assert "summary:" in rendered, (
        f"Expected 'summary:' field in result blocks, got:\n{rendered[:800]}"
    )
