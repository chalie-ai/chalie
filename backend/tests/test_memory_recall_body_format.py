"""Feature test: memory recall returns a STRUCTURED JSON body, and the transcript
back-reference resolves episodes from it (TKT-886).

Two contracts are pinned end-to-end against the real production hot path:

1. **Structured recall body.** ``recall`` renders
   ``[memory(status=success, …)]\\n{"results": [{id, content, score, kind,
   created_at}, …]}\\n[end:memory]`` — machine-parseable rows, not the old
   ``[id:X,relevance:Y] text`` prose. An EXPLICIT recall additionally carries a
   ``fallback`` field naming the document/schedule tools; the silent turn-0 seed
   carries none.

2. **The episode back-reference still resolves.** ``transcript_service.
   _fetch_referenced_episodes`` reads the rendered recall envelope back out of the
   ``tool_calls`` table and must resolve the underlying episode row. The format is
   load-bearing: this test stores a REAL episode, records a REAL recall envelope
   carrying that episode's id, and proves the real consumer fetches it back.

Zero mocks — real DataGraphService / EpisodicService / DatabaseService against the
real ``db`` fixture, the real ``ToolDispatcher`` envelope renderer, and the real
``ActTrail`` write.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.memory import MemoryAbility

pytestmark = pytest.mark.unit


def _parse_body(rendered: str) -> object:
    """JSON-parse the body between the open tag and ``[end:memory]``."""
    head = rendered.index("]\n") + 2
    tail = rendered.index("\n[end:memory]")
    return json.loads(rendered[head:tail])


def _render_recall(params: dict) -> str:
    """Run a real recall and render the dispatcher envelope (no bound mp — the
    data-graph recall lane does not depend on a processor)."""
    result = MemoryAbility().run(params)
    assert result is not None, "MemoryAbility.run() returned None"
    return ToolDispatcher._render("memory", result, None)


class TestMemoryRecallStructuredBody:
    def test_recall_hit_is_a_structured_json_row(self, db):
        """A real data-graph hit projects to a JSON row with the full contract:
        id / content / score / kind — replacing the old prose markers."""
        from services.data_graph_service import get_data_graph_service

        get_data_graph_service().store(
            kind="user_specific", key="residence", value="Valletta", source="test:seed",
        )

        out = _render_recall({"action": "recall", "query": "residence city"})

        assert "[memory(status=success" in out
        assert "[end:memory]" in out
        assert "results=0" not in out, f"expected a hit, got results=0: {out!r}"

        body = _parse_body(out)
        assert isinstance(body, dict), f"recall body must be a JSON object: {body!r}"
        rows = body["results"]
        assert isinstance(rows, list) and rows, f"expected non-empty results list: {body!r}"

        match = next((r for r in rows if r.get("id") == "residence"), None)
        assert match is not None, f"stored key 'residence' missing from rows: {rows!r}"
        # Every contracted field is present on the row.
        assert set(match) >= {"id", "content", "score", "kind", "created_at"}, (
            f"row missing contracted fields: {match!r}"
        )
        assert "Valletta" in match["content"]
        assert match["score"] in ("high", "medium", "low")
        assert match["kind"] == "user_specific"

    def test_explicit_recall_carries_fallback_field_seed_does_not(self, db):
        """The explicit recall body carries the fallback guardrail naming the
        document/schedule tools; the silent seed body omits it entirely."""
        explicit = _parse_body(
            _render_recall({"action": "recall", "query": "xyzzy_nonexistent_key_abc"})
        )
        assert explicit["results"] == []
        assert "fallback" in explicit, f"explicit recall lost the guardrail: {explicit!r}"
        assert "`document` (action: search)" in explicit["fallback"]
        assert "`schedule` (action: search)" in explicit["fallback"]

        seed = _parse_body(
            _render_recall(
                {"action": "recall", "query": "xyzzy_nonexistent_key_abc", "_auto": True}
            )
        )
        assert seed["results"] == []
        assert "fallback" not in seed, f"silent seed leaked the guardrail: {seed!r}"


class TestEpisodeBackReferenceFromStructuredBody:
    def test_recall_envelope_resolves_its_episode_downstream(self, db):
        """The transcript back-reference fetches the real episode from a recorded
        recall envelope whose JSON body carries that episode's id.

        Producer → consumer end-to-end: a REAL episode is stored, a REAL recall
        envelope carrying an ``episode``-kind row is recorded on the act-trail, and
        the REAL ``_fetch_referenced_episodes`` resolves it back out of the JSON —
        proving the structured body keeps the load-bearing back-reference alive.
        """
        from services.act_trail import ActTrail
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        from services.memory_retrieval import _recall_payload
        from services.transcript_service import _fetch_referenced_episodes

        shared_db = get_shared_db_service()
        episode_id = EpisodicService(shared_db).store_episode(
            {"gist": "We talked about the trip to Gozo at home", "salience": 7, "channel": "chat"}
        )

        # A transcript anchor the trail row references.
        cur = db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            ("chat", "user", "what did we talk about at home"),
        )
        db.commit()
        transcript_id = cur.lastrowid

        # Build the recall body exactly as production does — a real episode-kind
        # hit through the real projection — and render the real envelope.
        hit = {
            "id": episode_id,
            "text": "We talked about the trip to Gozo at home",
            "relevance": "high",
            "confidence": 0.9,
            "kind": "episode",
            "created_at": "2026-06-10T00:00:00+00:00",
        }
        body = {"results": _recall_payload([hit])}
        from abilities._result import ToolResult
        rendered = ToolDispatcher._render(
            "memory", ToolResult.ok(body, query="home", results=1, degraded=False), None
        )

        # Record it on the real trail under tool_name='memory' — the exact row the
        # consumer reads back.
        ActTrail().record(
            tool_name="memory", params={"action": "recall"}, result=rendered,
            transcript_id=transcript_id,
        )

        entries = [{"id": transcript_id}]
        episodes = _fetch_referenced_episodes(entries, shared_db)

        assert episodes, "back-reference resolved no episodes from the JSON body"
        assert any(str(ep.get("id")) == episode_id for ep in episodes), (
            f"episode {episode_id} not resolved from recall envelope: "
            f"{[ep.get('id') for ep in episodes]!r}"
        )
