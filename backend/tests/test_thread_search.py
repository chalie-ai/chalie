"""Feature tests for keyword thread search (§5.2).

Search is NOT a separate endpoint — it is the thread feed (GET /api/threads) with a
``q`` filter. Same controller, same getter, same response shape; the only difference
is that ``?q=`` keyword-filters the page to matching threads and caps it at 5.

Zero mocks. Drives the real production writers against the real test DB, then hits the
real GET /api/threads endpoint and asserts the contract:

1. ``?q=`` returns the SAME ``{threads: [...]}`` metadata shape as the bare feed —
   the matching turns with their preview/gist, not a special projection.
2. Only threads with a matching USER-role row come back (assistant text never matches).
3. The result set is capped at 5, most-recent (turn_id DESC) first.
4. A turn that never grew past settle0 (a non-thread) is still findable.
5. The gist rides search results exactly as it rides the feed.
6. No ``q`` → the unfiltered feed is unchanged (regression guard).
"""
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from services.thread_gist_service import get_thread_gist_service
from services.transcript_service import Transcript


def _threads(client: FlaskClient, q: str = "") -> list[dict[str, object]]:
    url = f'/api/threads?q={q}' if q else '/api/threads'
    payload = cast("dict[str, object]", client.get(url).get_json())
    return cast("list[dict[str, object]]", payload["threads"])


@pytest.mark.unit
class TestThreadSearch:

    def test_query_filters_feed_to_matching_threads_same_shape(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """?q= returns the matching thread with the same metadata shape as the feed."""
        client = cast("FlaskClient", authed_client[0])
        hit = Transcript.turn_id_of_row(
            Transcript.write_input_row('user', 'user', "Researching a Mac Mini for my homelab"),
        )
        Transcript.turn_id_of_row(
            Transcript.write_input_row('user', 'user', "Drafting an HN post about Rust"),
        )

        results = _threads(client, "homelab")
        assert [r["turn_id"] for r in results] == [hit]
        assert results[0]["preview"] == "Researching a Mac Mini for my homelab"  # same feed shape
        assert "last_activity_at" in results[0] and "row_count" in results[0]

    def test_only_user_rows_match(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """A keyword only present in the assistant reply surfaces no thread."""
        client = cast("FlaskClient", authed_client[0])
        turn_id = Transcript.turn_id_of_row(Transcript.write_input_row('user', 'user', "tell me a fact"))
        Transcript.write_assistant_row('user', "Octopuses have three hearts", turn_id=turn_id)

        assert _threads(client, "Octopuses") == []
        assert [r["turn_id"] for r in _threads(client, "fact")] == [turn_id]

    def test_caps_at_five_most_recent_first(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """Seven matching threads → only the five newest (turn_id DESC) come back."""
        client = cast("FlaskClient", authed_client[0])
        turn_ids = [
            Transcript.turn_id_of_row(
                Transcript.write_input_row('user', 'user', f"widget number {n} please"),
            )
            for n in range(7)
        ]

        results = _threads(client, "widget")
        assert [r["turn_id"] for r in results] == list(reversed(turn_ids))[:5]

    def test_non_thread_turn_is_findable(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """A turn that only ever had its opening exchange (never grew past settle0) is
        still searchable — its turn_id opens the thread view."""
        client = cast("FlaskClient", authed_client[0])
        turn_id = Transcript.turn_id_of_row(
            Transcript.write_input_row('user', 'user', "quick question about kubernetes"),
        )
        Transcript.write_assistant_row('user', "sure, ask away", turn_id=turn_id)

        assert Transcript.settle0('user', turn_id) is not None, "single exchange has settled"
        assert [r["turn_id"] for r in _threads(client, "kubernetes")] == [turn_id]
        assert client.get(f'/api/thread/{turn_id}').get_json()["turn_id"] == turn_id

    def test_gist_rides_search_results(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """The gist attaches to search hits exactly as it does to the feed."""
        client = cast("FlaskClient", authed_client[0])
        turn_id = Transcript.turn_id_of_row(
            Transcript.write_input_row('user', 'user', "Drafting an HN post"),
        )
        get_thread_gist_service().upsert('user', turn_id, "HN Post Feedback")

        mine = next(r for r in _threads(client, "Drafting") if r["turn_id"] == turn_id)
        assert mine["gist"] == "HN Post Feedback"

    def test_no_query_is_the_unfiltered_feed(
        self, authed_client: tuple[object, sqlite3.Connection, object],
    ) -> None:
        """Without q the feed is unchanged — every seeded thread is present."""
        client = cast("FlaskClient", authed_client[0])
        seeded = {
            Transcript.turn_id_of_row(Transcript.write_input_row('user', 'user', f"topic {n}"))
            for n in range(3)
        }

        feed_ids = {r["turn_id"] for r in _threads(client)}
        assert seeded <= feed_ids
