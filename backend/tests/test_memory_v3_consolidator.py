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
    shape, formats each exchange as ``[yyyy-mm-dd HH:mm @ location] role: content``
    when location_name is present, omits the ``@ ...`` segment when it is absent or
    NULL (no ``unknown`` placeholder), and keeps rows oldest-first."""
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
    # NULL location_name: no "@" segment at all — never "unknown".
    assert "[2026-01-01 09:00] assistant: Lovely city" in window
    assert "[2026-01-01 09:05 @ Lisbon] user: I adopted a dog" in window
    # The literal string "unknown" must never appear in the window.
    assert "unknown" not in window
    # Oldest-first (ASC by id); the whole batch fits the generous budget.
    assert batch_ids == [101, 102, 103]


def test_build_window_uses_descriptor_header(db: sqlite3.Connection) -> None:
    """The window header uses the channel descriptor (name + description), not the
    raw channel key and not the system-prompt preamble. The preamble text must not
    leak into the window body."""
    from configs.channels.memory_consolidator import (  # noqa: PLC0415
        _DEFAULT_PREAMBLE,
        _PREAMBLES,
    )

    rows: list[tuple[int, str, str, str, str | None]] = [
        (1, "user", "hello", "2026-01-01T09:00:00+00:00", None),
        (2, "user", "world", "2026-01-01T09:01:00+00:00", None),
    ]
    window, _ = MemoryConsolidatorService._build_window(rows, "user", 4000)

    # Header must use the descriptor name and description, not the raw key.
    assert window.startswith("## User conversation\n### Description\n")
    assert "Direct exchanges between the user and the assistant" in window
    assert "### Exchanges\n" in window

    # The preamble text must NOT appear anywhere in the window body.
    for preamble_fragment in _PREAMBLES.values():
        assert preamble_fragment not in window
    assert _DEFAULT_PREAMBLE not in window


def test_build_window_truncates_at_budget(db: sqlite3.Connection) -> None:
    """A budget smaller than the corpus drops the newest overflow rows but keeps
    at least the oldest (the first row is always included)."""
    big: list[tuple[int, str, str, str, str | None]] = [
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


def _seed_turn(db: sqlite3.Connection, channel: str, user_content: str,
               assistant_content: str, settled: int = 1) -> int:
    """One turn: user row + assistant row, with the assistant row carrying
    ``settled`` (1 = settled, 0 = in-flight). Returns the turn_id."""
    turn_id = int(db.execute(
        "SELECT COALESCE(MAX(turn_id), 0) + 1 FROM transcript WHERE channel = ?",
        (channel,),
    ).fetchone()[0])
    db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, consolidated) "
        "VALUES (?, ?, ?, ?, 0, 0)",
        (channel, "user", user_content, turn_id),
    )
    db.execute(
        "INSERT INTO transcript (channel, role, content, turn_id, settled, consolidated) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (channel, "assistant", assistant_content, turn_id, settled),
    )
    db.commit()
    return turn_id


def test_settled_turn_gate_excludes_inflight_turns(db: sqlite3.Connection) -> None:
    """Rows of in-flight turns (no settled assistant sibling) are excluded from
    the consolidator window. A channel with many settled turns plus one
    in-flight turn consolidates only the settled turn rows; the in-flight
    rows stay at consolidated=0 after a successful pass."""
    # Seed 10 settled turns (20 rows) — enough to pass the _MIN_ROWS gate.
    for i in range(10):
        _seed_turn(db, "gate_c", f"settled user {i}", f"settled assistant {i}", settled=1)
    # Seed 1 in-flight turn (2 rows) — no settled assistant sibling.
    inflight_turn_id = _seed_turn(
        db, "gate_c", "inflight user", "inflight assistant", settled=0
    )

    svc = MemoryConsolidatorService()
    result = svc.consolidate("gate_c")
    assert result == "gate_c consolidated (20 rows)"

    # The in-flight turn's rows must still be consolidated=0.
    inflight_rows = db.execute(
        "SELECT id, consolidated FROM transcript "
        "WHERE channel = ? AND turn_id = ?",
        ("gate_c", inflight_turn_id),
    ).fetchall()
    assert len(inflight_rows) == 2
    for row_id, consolidated in inflight_rows:
        assert int(consolidated) == 0, f"row {row_id} should stay unconsolidated"

    # The settled turn rows must now be consolidated=1.
    settled_count = db.execute(
        "SELECT COUNT(*) FROM transcript "
        "WHERE channel = ? AND turn_id != ? AND consolidated = 1",
        ("gate_c", inflight_turn_id),
    ).fetchone()
    assert int(settled_count[0]) == 20


def test_consolidator_recall_depth_exceeds_chat_default(
    db: sqlite3.Connection,
) -> None:
    """The consolidator's recall uses k=10 per lane while the chat default
    stays at k=3. Seed more than 3 matching graph facts: recall through a
    consolidator-config context returns more than 3; the same query through
    the default path returns at most 3."""
    from abilities.recall import Recall  # noqa: PLC0415
    from contracts.params.recall_params_bag import RecallParamsBag  # noqa: PLC0415
    from contracts.search_config import config_for_table  # noqa: PLC0415
    from services.memory_recall_service import MemoryRecallService  # noqa: PLC0415
    from services.search_expander_service import SearchExpanderService  # noqa: PLC0415

    # Seed 5 matching graph facts. Dot separators keep "pet" a standalone FTS5
    # token — "user.pet3" tokenizes to "pet3", which the query "pet" never
    # matches.
    ses_config = config_for_table("memory_graph")
    assert ses_config is not None
    for i in range(5):
        row = MemoryGraphRow(subject=f"user.pet.{i}", contents=f"The user has a pet #{i}")
        row.save()
        assert isinstance(row.id, int)
        SearchExpanderService()._process_row("memory_graph", row.id, ses_config)

    # Default k: all 5 facts match, only 3 come back.
    default_result = MemoryRecallService().recall("pet", k_graph=3, k_map=0)
    assert len(default_result["graph"]) == 3, f"default graph hits: {default_result['graph']}"

    # Consolidator path: mp stub carrying MemoryConsolidatorConfig (recall_k=10).
    cfg = MemoryConsolidatorConfig(
        target_channel="user",
        window="## test\n### Description\ntest\n\n### Exchanges\n",
        source_transcript_ids=[],
    )
    mp_stub = type("MockMP", (), {"config": cfg})()
    ability = Recall(mp=mp_stub)
    bag = RecallParamsBag.from_params({"query": "pet"})
    assert isinstance(bag, RecallParamsBag)
    tr = ability.run(bag)
    assert isinstance(tr, ToolResult) and tr.status == "success"
    assert isinstance(tr.body, str)
    # The body lists facts as "- subject: contents" lines; all 5 fit in k=10.
    fact_lines = [line for line in tr.body.splitlines() if line.startswith("- user.pet")]
    assert len(fact_lines) == 5, f"expected all 5 facts at k=10, got:\n{tr.body!r}"
