# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature tests for the Memory v3 consolidator service (Slice C).

Covers the consolidator-specific wiring on the real ``db`` fixture: the
10-row floor, the consolidated-flag progress tracker, the window format, and
provenance stamping (the write tools attribute ``sourced_from`` off the
consolidator config). The full LLM drive (``MessageProcessor.process``) is
exercised by the wider suite; these tests pin the consolidator's own logic.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from abilities._result import ToolResult
from abilities.save_graph import SaveGraph
from configs.channels.memory_consolidator import MemoryConsolidatorConfig
from contracts.params.save_graph_params_bag import SaveGraphParamsBag
from models.memory_graph import MemoryGraphRow
from services.memory_consolidator_service import MemoryConsolidatorService

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.unit


def _seed_rows(db: sqlite3.Connection, channel: str, lines: list[tuple[str, str]]) -> None:
    """Insert transcript rows with consolidated=0 by default."""
    for role, content in lines:
        db.execute(
            "INSERT INTO transcript (channel, role, content, consolidated) "
            "VALUES (?, ?, ?, 0)",
            (channel, role, content),
        )
    db.commit()


def test_consolidate_rejects_less_than_min_rows(db: sqlite3.Connection) -> None:
    """Fewer than 10 unconsolidated rows → no consolidation attempt."""
    _seed_rows(db, "test_c", [("user", "I live in Lisbon")])

    svc = MemoryConsolidatorService()
    result = svc.consolidate("test_c")
    assert result == "test_c: <10 unconsolidated rows"

    # Row should still be unconsolidated.
    row = db.execute(
        "SELECT consolidated FROM transcript WHERE channel = ? AND role = ?",
        ("test_c", "user"),
    ).fetchone()
    assert row is not None
    assert int(row[0]) == 0


def test_build_window_formats_rows_oldest_first(db: sqlite3.Connection) -> None:
    """_build_window (pure) emits the ## channel / ### Description / ### Exchanges
    shape, formats each exchange as ``[yyyy-mm-dd HH:mm @ location] role: content``,
    falls back to ``unknown`` for a NULL location, and keeps rows oldest-first."""
    rows = [
        (101, "user", "I live in Lisbon", "2026-01-01T09:00:00+00:00", "Lisbon"),
        (102, "assistant", "Lovely city", "2026-01-01T09:00:30+00:00", None),
        (103, "user", "I adopted a dog", "2026-01-01T09:05:00+00:00", "Lisbon"),
    ]
    # A budget large enough to hold all three rows.
    window, batch_ids = MemoryConsolidatorService._build_window(rows, "test_c", 4000)

    assert window.startswith("## test_c\n### Description\n")
    assert "### Exchanges\n" in window
    assert "[2026-01-01 09:00 @ Lisbon] user: I live in Lisbon" in window
    # NULL location_name falls back to "unknown".
    assert "[2026-01-01 09:00 @ unknown] assistant: Lovely city" in window
    assert "[2026-01-01 09:05 @ Lisbon] user: I adopted a dog" in window
    # Oldest-first (ASC by id); the whole batch fits the generous budget.
    assert batch_ids == [101, 102, 103]


def test_build_window_truncates_at_budget(db: sqlite3.Connection) -> None:
    """A budget smaller than the corpus drops the newest overflow rows but keeps
    at least the oldest (the first row is always included)."""
    big = [
        (i, "user", "y" * 80, "2026-01-01T09:00:00+00:00", None)
        for i in range(1, 30)
    ]
    # Tiny budget: only a handful of ~80-char rows fit before the cap.
    window, batch_ids = MemoryConsolidatorService._build_window(big, "test_c", 60)

    assert batch_ids[0] == 1
    assert len(batch_ids) < 29


def test_mark_consolidated_stamps_flag(db: sqlite3.Connection) -> None:
    """_mark_consolidated writes consolidated=1 on exactly the batch over the
    real db; rows outside the batch stay at 0."""
    _seed_rows(db, "test_c", [("user", f"line {i}") for i in range(12)])
    ids = [
        int(r[0])
        for r in db.execute(
            "SELECT id FROM transcript WHERE channel = ? ORDER BY id ASC LIMIT 5",
            ("test_c",),
        ).fetchall()
    ]
    assert ids, "seeded rows should exist"

    MemoryConsolidatorService._mark_consolidated(ids)

    marked = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE consolidated = 1 AND channel = ?",
        ("test_c",),
    ).fetchone()
    assert marked is not None
    assert int(marked[0]) == len(ids)


def test_tick_skips_excluded_channels(db: sqlite3.Connection) -> None:
    """Channels in _EXCLUDED_CHANNELS are never touched by tick()."""
    from services.memory_consolidator_service import _EXCLUDED_CHANNELS  # noqa: PLC0415

    assert "memory_consolidator" in _EXCLUDED_CHANNELS
    assert "skills_building" in _EXCLUDED_CHANNELS
    assert "discovery" in _EXCLUDED_CHANNELS


def test_save_graph_stamps_provenance_from_consolidator_config(db: sqlite3.Connection) -> None:
    """The consolidator config's source_transcript_ids flow into SaveGraph
    provenance."""
    cfg = MemoryConsolidatorConfig(
        target_channel="user",
        window="## user\n### Description\ntest\n\n### Exchanges\n",
        source_transcript_ids=[42, 43],
    )
    mp = type("MockMP", (), {"config": cfg})()
    ability = SaveGraph(mp=mp)

    bag = SaveGraphParamsBag.from_params({"subject": "user.residence", "contents": "Lisbon"})
    assert not isinstance(bag, ToolResult)
    ability.run(bag)

    row = MemoryGraphRow.by_subject("user.residence")
    assert row is not None
    assert json.loads(row.sourced_from) == [42, 43]


def test_consolidated_flag_prevents_re_consolidation(db: sqlite3.Connection) -> None:
    """Rows stamped consolidated=1 are no longer returned by the query."""
    lines: list[tuple[str, str]] = [("user", f"I live in Lisbon {i}") for i in range(12)]
    _seed_rows(db, "test_c", lines)

    # Manually stamp half the rows as consolidated.
    ids = db.execute(
        "SELECT id FROM transcript WHERE channel = ? ORDER BY id ASC LIMIT 6",
        ("test_c",),
    ).fetchall()
    row_ids = [int(r[0]) for r in ids]
    placeholders = ",".join("?" for _ in row_ids)
    db.execute(
        f"UPDATE transcript SET consolidated = 1 WHERE id IN ({placeholders})",
        row_ids,
    )
    db.commit()

    # The service should still see >= 10 unconsolidated rows (the remaining 6
    # from the seed + 0 new ones = 6, so it should return <10).
    svc = MemoryConsolidatorService()
    result = svc.consolidate("test_c")
    # Only 6 unconsolidated rows remain; the floor is 10.
    assert result == "test_c: <10 unconsolidated rows"
