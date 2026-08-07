# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the Memory v3 consolidator service (Slice C).

Covers the consolidator-specific wiring on the real ``db`` fixture:
readiness (most-recent not-yet-consolidated turn), the consolidated-marker check
(a turn is done once a map row cites one of its transcript ids), and provenance
stamping (the write tools attribute ``sourced_from`` off the consolidator
config). The full LLM drive (``MessageProcessor.process``) is exercised by the
wider suite; these tests pin the consolidator's own logic.
"""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from abilities._result import ToolResult
from abilities.save_graph import SaveGraph
from contracts.params.save_graph_params_bag import SaveGraphParamsBag
from models.memory_graph import MemoryGraphRow
from models.memory_map import MemoryMapRow
from services.memory_consolidator_service import MemoryConsolidatorService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _seed_turn(db, channel: str, turn_id: int, lines) -> None:
    for role, content in lines:
        db.execute(
            "INSERT INTO transcript (channel, role, content, turn_id) "
            "VALUES (?, ?, ?, ?)",
            (channel, role, content, turn_id),
        )
    db.commit()


def test_most_recent_unconsolidated_turn_then_marker(db) -> None:
    _seed_turn(db, "test_c", 1, [("user", "I live in Lisbon"), ("assistant", "Nice")])
    _seed_turn(db, "test_c", 2, [("user", "I adopted a dog"), ("assistant", "Cool")])

    svc = MemoryConsolidatorService()
    assert svc._most_recent_unconsolidated_turn("test_c") == 2

    # Writing a map row sourced from turn 2 marks turn 2 consolidated.
    t2_row = db.execute(
        "SELECT id FROM transcript WHERE channel = ? AND turn_id = ?", ("test_c", 2)
    ).fetchone()
    assert t2_row is not None
    MemoryMapRow(contents="user adopted a dog", sourced_from=json.dumps([int(t2_row[0])])).save()

    assert svc._most_recent_unconsolidated_turn("test_c") == 1


def test_save_graph_stamps_provenance_from_consolidator_config(db) -> None:
    cfg = SimpleNamespace(_source_transcript_ids=[42, 43])
    mp = SimpleNamespace(config=cfg)
    ability = SaveGraph(mp=cast("MessageProcessor", mp))

    bag = SaveGraphParamsBag.from_params({"subject": "user.residence", "contents": "Lisbon"})
    assert not isinstance(bag, ToolResult)
    ability.run(bag)

    row = MemoryGraphRow.by_subject("user.residence")
    assert row is not None
    assert json.loads(row.sourced_from) == [42, 43]
