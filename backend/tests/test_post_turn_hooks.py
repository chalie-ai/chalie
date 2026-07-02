# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test for the PostTurnHook failure-isolation contract.

Drives the real production after-turn path — ``MessageProcessor._end_turn`` —
against the real test database with the real production hook
``PersistUserSummaryHook``.  A sibling hook that raises MUST NOT stop the real
hook from doing its downstream work (writing user_summary rows to data_graph).

No mocks: the data_graph write goes through the real DataGraphService bound to
the real test DB, and the assertion reads the row straight back out of SQLite.
"""

import sqlite3
from typing import TYPE_CHECKING

import pytest

from configs.channels.user_summary import PersistUserSummaryHook
from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig
from tests.helpers import StubProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor as _MP

pytestmark = pytest.mark.unit


class _ExplodingHook(PostTurnHook):

    def __init__(self) -> None:
        self.ran = False

    def run(self, mp: "_MP", result_text: str) -> None:
        self.ran = True
        raise RuntimeError("hook deliberately exploded")


_SUMMARY_CHANNEL = "test_post_turn_hooks_channel"


def _config_with_hooks(hooks: tuple[PostTurnHook, ...]) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel=_SUMMARY_CHANNEL,
        role="user_summary",
        policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=[],
        skip_transcript=True,
        skip_input_row=False,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn_hooks=hooks,
    )


def _make_processor(hooks: tuple[PostTurnHook, ...]) -> "_MP":
    from services.message_processor import MessageProcessor

    return MessageProcessor(_config_with_hooks(hooks), raw_input="raw input")


_SUMMARY_JSON = '{"short": "Alice, a runner", "long": "Alice is a long-distance runner in Berlin."}'


def _read_summary(db: sqlite3.Connection) -> dict[str, object]:
    rows = db.execute(
        "SELECT key, value FROM data_graph "
        "WHERE kind='system' AND key IN ('user_summary', 'user_summary_long') "
        "AND active=1 AND deleted_at IS NULL"
    ).fetchall()
    return dict(rows)


def test_exploding_sibling_does_not_block_real_hook(db: sqlite3.Connection) -> None:
    """A saboteur hook raising first must not stop PersistUserSummaryHook from
    writing both user_summary rows to data_graph (failure isolation)."""
    saboteur = _ExplodingHook()
    mp = _make_processor((saboteur, PersistUserSummaryHook()))

    # Real production after-turn entry point — runs the hook loop.
    mp._end_turn(_SUMMARY_JSON)

    assert saboteur.ran, "saboteur hook should have been invoked"
    summary = _read_summary(db)
    assert summary.get("user_summary") == "Alice, a runner"
    assert summary.get("user_summary_long") == "Alice is a long-distance runner in Berlin."


def test_isolation_holds_regardless_of_hook_order(db: sqlite3.Connection) -> None:
    saboteur = _ExplodingHook()
    mp = _make_processor((PersistUserSummaryHook(), saboteur))

    mp._end_turn(_SUMMARY_JSON)

    assert saboteur.ran
    summary = _read_summary(db)
    assert summary.get("user_summary") == "Alice, a runner"
    assert summary.get("user_summary_long") == "Alice is a long-distance runner in Berlin."
