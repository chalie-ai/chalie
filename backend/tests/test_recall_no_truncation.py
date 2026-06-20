"""Feature tests: memory recall surfaces the FULL episode gist — NO truncation
(TKT-821).

Three production hot-paths hard-clipped a recalled episode gist that the model
actually reads, violating the NO-TRUNCATION contract:

1. **Semantic recall** — ``services.memory_retrieval.recall_episodes`` projected
   ``gist[:200]`` into the structured recall body.
2. **Location recall** — ``services.memory_retrieval._search_episodes_by_location``
   projected ``(gist or "")[:200]``.
3. **Turn-0 flashback** — ``services.turn_zero_flashback._render_episode`` clipped
   the gist to 160 chars + "…".

Each test drives the REAL entry point production uses — ``MemoryAbility.run`` for
the two recall lanes (the same call the ACT-loop dispatcher makes) and
``MessageProcessor._seed_turn_zero`` for the flashback — seeds an episode whose
gist is LONGER than the old cap via the production ``EpisodicService.store_episode``
write path, and asserts the model-visible text is the WHOLE, unclipped gist.

Zero mocks: real ``EpisodicService`` / ``DatabaseService`` against the ``db``
fixture, the real ``ToolDispatcher`` envelope renderer, the real ``UserConfig``
MessageProcessor + embedding model for the flashback gate. RED against HEAD before
the truncation is removed: each gist exceeds its path's old cap, so the truncated
content cannot equal the full gist.
"""

import json
import sqlite3

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.memory import MemoryAbility

pytestmark = pytest.mark.unit


#: The conftest builds the vec tables at 256 dims (tests/conftest.py); production
#: always passes an embedding, so a faithful episode is seeded with one. The
#: runtime recall query embeds at 768 and the vec lane no-ops on the width
#: mismatch (logged, non-fatal) — the FTS lane carries the episode. Same
#: precedent as tests/test_turn0_flashback_continuation_gate.py.
_VEC_DIM = 256


def _unit(index: int, dim: int = _VEC_DIM) -> list[float]:
    """A unit basis vector matching the fixture's vec-table width."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _parse_body(rendered: str) -> dict[str, object]:
    """JSON-parse the recall body between the open tag and ``[end:memory]``."""
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:memory]")
    return dict(json.loads(rendered[head:tail]))


def _render_recall(params: dict[str, object]) -> str:
    """Run a real recall through the ability entry point and render the
    dispatcher envelope — the exact string the model reads. Unbound mp mirrors
    the data-graph/location/REST recall lanes that do not need a processor."""
    result = MemoryAbility().run(params)
    assert result is not None, "MemoryAbility.run() returned None"
    return ToolDispatcher._render("memory", result, None)


def _store_episode(gist: str, *, emb_index: int = 7, **fields: object) -> str:
    """Seed one episode via the PRODUCTION write path (EpisodicService), with a
    256-dim embedding exactly as every production caller passes."""
    from services.database_service import get_shared_db_service
    from services.episodic_service import EpisodicService

    data: dict[str, object] = {"gist": gist, "salience": 8, "channel": "user", **fields}
    return EpisodicService(get_shared_db_service()).store_episode(
        data, embedding=_unit(emb_index)
    )


def test_semantic_recall_returns_the_full_untruncated_gist(db: sqlite3.Connection) -> None:
    """An explicit semantic ``recall`` projects the WHOLE episode gist into the
    structured body — not the first 200 chars."""
    long_gist = (
        "The user spent the whole afternoon at the Valletta harbour debugging "
        "the ferry booking integration with the Gozo Channel Line scheduling "
        "system, chasing a timezone bug that mis-stamped every Saturday "
        "departure by an hour, then rewrote the parser and confirmed the "
        "family trip reservation for six passengers was finally stored against "
        "the correct sailing."
    )
    assert len(long_gist) > 200, "gist must exceed the old 200-char cap"

    _store_episode(long_gist)

    body = _parse_body(
        _render_recall(
            {"action": "recall", "query": "harbour ferry booking Gozo Channel Line"}
        )
    )
    from typing import cast
    results = cast(list[dict[str, object]], body["results"])
    episodes = [r for r in results if r.get("kind") == "episode"]
    assert episodes, f"semantic recall surfaced no episode: {body!r}"

    match = next((r for r in episodes if r["content"] == long_gist), None)
    assert match is not None, (
        "recalled episode content is not the full gist — still truncated. got: "
        f"{[r['content'] for r in episodes]!r}"
    )
    assert "…" not in cast(str, match["content"]), "gist was clipped with an ellipsis"
    assert len(cast(str, match["content"])) == len(long_gist)


def test_location_recall_returns_the_full_untruncated_gist(db: sqlite3.Connection) -> None:
    """A location-only ``recall`` projects the WHOLE gist of every episode at
    that place — not the first 200 chars."""
    long_gist = (
        "Over a long lunch in Valletta the user walked through the entire "
        "harbour redevelopment proposal, sketching where the new ferry "
        "terminal, the pedestrian promenade and the underground parking would "
        "sit, and promised to send the annotated map to the planning committee "
        "before the end of the month so the public consultation could open on "
        "schedule."
    )
    assert len(long_gist) > 200, "gist must exceed the old 200-char cap"

    _store_episode(long_gist, location_name="Valletta", emb_index=9)

    body = _parse_body(_render_recall({"action": "recall", "location": "Valletta"}))
    from typing import cast
    rows = cast(list[dict[str, object]], body["results"])
    assert rows, f"location recall surfaced no episode: {body!r}"

    match = next((r for r in rows if r["content"] == long_gist), None)
    assert match is not None, (
        "location-recalled episode content is not the full gist — still "
        f"truncated. got: {[r['content'] for r in rows]!r}"
    )
    assert "…" not in cast(str, match["content"]), "gist was clipped with an ellipsis"
    assert match.get("location") == "Valletta"
    assert len(cast(str, match["content"])) == len(long_gist)


def test_turn_zero_flashback_renders_the_full_untruncated_gist(db: sqlite3.Connection) -> None:
    """The turn-0 flashback bundle the model reads on session start carries the
    WHOLE episode gist — not a 160-char one-liner clipped with "…"."""
    from typing import cast
    from configs.channels import UserConfig
    from services.message_processor import MessageProcessor
    from services.transcript_service import write_input_row

    long_gist = (
        "Remind me about the Gozo ferry booking — the user booked it for the "
        "family trip on Saturday morning, paid the deposit online through the "
        "Gozo Channel Line portal, and asked to be reminded the evening before "
        "so there is time to print the boarding passes and pack the car for "
        "the early sailing."
    )
    assert len(long_gist) > 160, "gist must exceed the old 160-char one-liner cap"

    _store_episode(long_gist)

    # Drive the real session-start flashback exactly where it fires in the ACT
    # loop: a fresh UserConfig MessageProcessor with an input row written.
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "remind me about the Gozo ferry booking", {})
    mp.config = UserConfig()
    mp.uid = write_input_row("user", "user", "remind me about the Gozo ferry booking")
    mp.active_tools = list(mp.config.always_available or [])
    mp._seed_turn_zero()

    injected = db.execute(
        "SELECT result FROM tool_calls WHERE tool_name = 'memory' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert injected is not None, "the session-start seed recorded no memory call"
    block = cast(str, cast(tuple[object, ...], injected)[0])

    assert long_gist in block, (
        f"turn-0 flashback clipped the gist — full text absent: {block!r}"
    )
    assert "…" not in block, "flashback clipped the gist with an ellipsis"
