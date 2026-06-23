"""
Integration: moments pin → list → search → forget round-trip.

Exercises the real Flask app (authed_client), the real SQLite database, the
real transcript_service, and the real data_graph-backed moments API end to end
— zero mocks beyond auth bypass and the isolated store provided by the fixture.

The driving contract is the one the chat UI actually sends: the remember button
has NO assistant transcript row id (the WS event only carries a per-request
exchange UUID, which the transcript table does not store), so it pins by
``message_text``. These tests assert that payload succeeds and persists, which
is exactly the case that returned HTTP 400 ``transcript_id is required`` before
the fix.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from services.transcript_service import Transcript
from services.markup import extract_plaintext, sanitize

pytestmark = pytest.mark.integration


# Sanitised assistant content as it is stored + rendered. Contains an allowlisted
# <b> so we also confirm matching tolerates inline formatting tags.
_ASSISTANT_HTML = sanitize(
    "The first star <b>formed before its host galaxy</b>, according to recent "
    "telescope observations about cosmology."
)


def _seed_assistant_turn(content: str = _ASSISTANT_HTML) -> int:
    return Transcript.write_assistant_row("user", content)


def _ui_message_text(content: str = _ASSISTANT_HTML) -> str:
    return extract_plaintext(content)


def test_pin_by_message_text_persists_moment(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    client, _db, _store = authed_client
    tid = _seed_assistant_turn()

    resp = client.post(
        "/moments",
        json={"message_text": _ui_message_text()},
        content_type="application/json",
    )

    assert resp.status_code == 201, resp.get_json()
    item = resp.get_json()["item"]
    assert item["transcript_id"] == tid
    assert item["key"] == f"moment_{tid}"
    assert item["value"] == _ASSISTANT_HTML
    assert item["message_text"] == _ASSISTANT_HTML


def test_pin_missing_text_and_id_is_rejected(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    client, _db, _store = authed_client

    resp = client.post("/moments", json={}, content_type="application/json")

    assert resp.status_code == 400
    assert "message_text" in resp.get_json()["error"]


def test_pin_unmatched_text_returns_404(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    """Text that matches no assistant turn yields 404, not a silent success."""
    client, _db, _store = authed_client
    _seed_assistant_turn()

    resp = client.post(
        "/moments",
        json={"message_text": "this string was never said by the assistant"},
        content_type="application/json",
    )

    assert resp.status_code == 404


def test_full_roundtrip_pin_list_search_forget(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    """The complete lifecycle the Recall UI depends on."""
    client, _db, _store = authed_client
    tid = _seed_assistant_turn()
    msg_text = _ui_message_text()

    # 1. Pin via the UI payload.
    create = client.post(
        "/moments",
        json={"message_text": msg_text},
        content_type="application/json",
    )
    assert create.status_code == 201, create.get_json()
    created = create.get_json()["item"]
    assert created["transcript_id"] == tid

    # 2. List shows it.
    listed = client.get("/moments")
    assert listed.status_code == 200
    keys = [it["key"] for it in listed.get_json()["items"]]
    assert f"moment_{tid}" in keys

    # 3. Search returns the pinned text (FTS over the stored value).
    search = client.get("/moments/search?q=cosmology")
    assert search.status_code == 200
    values = [it["value"] for it in search.get_json()["items"]]
    assert _ASSISTANT_HTML in values

    # 4. Forget (the Undo path) removes it — keyed by transcript_id, the value
    #    the create response actually hands the client.
    forget = client.post(f"/moments/{created['transcript_id']}/forget")
    assert forget.status_code == 200
    assert forget.get_json()["ok"] is True

    # 5. Gone from list and search.
    listed_after = client.get("/moments")
    keys_after = [it["key"] for it in listed_after.get_json()["items"]]
    assert f"moment_{tid}" not in keys_after

    search_after = client.get("/moments/search?q=cosmology")
    values_after = [it["value"] for it in search_after.get_json()["items"]]
    assert _ASSISTANT_HTML not in values_after


def test_explicit_transcript_id_path_still_works(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    """Programmatic callers may still pin by integer transcript_id."""
    client, _db, _store = authed_client
    tid = _seed_assistant_turn()

    resp = client.post(
        "/moments",
        json={"transcript_id": tid},
        content_type="application/json",
    )

    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()["item"]["transcript_id"] == tid


def test_duplicate_pin_flags_duplicate(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    """Pinning the same turn twice returns 200 with duplicate=True."""
    client, _db, _store = authed_client
    _seed_assistant_turn()
    payload = {"message_text": _ui_message_text()}

    first = client.post("/moments", json=payload, content_type="application/json")
    second = client.post("/moments", json=payload, content_type="application/json")

    assert first.status_code == 201
    assert first.get_json()["duplicate"] is False
    assert second.status_code == 200
    assert second.get_json()["duplicate"] is True
