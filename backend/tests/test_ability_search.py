# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import MP, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    return MP(seed_transcript(db, "chat", "search for something"), UserConfig({}))


# ── Blank query: slips the pre-gate, caught by search's own .strip() guard ─────


def test_blank_query_reports_missing_params(db: sqlite3.Connection, chat_mp: MP) -> None:
    out = ToolDispatcher(chat_mp).dispatch(
        "search", {"query": "   ", "act_summary": "x"}
    )

    assert "[search(status=error, code=missing-params" in out
    assert "query" in out
    assert "[end:search]" in out
