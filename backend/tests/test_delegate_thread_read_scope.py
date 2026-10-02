"""Feature test: which threads a client may read by channel, and which it may not.

A delegate turn has no routing ``type`` of its own, so the single-turn read
(``GET /api/threads/<id>``) also accepts a ``channel`` — but only for the
delegates whose child turn writes a transcript worth watching. Everything else
must stay refused loudly: reading an arbitrary channel would expose internal
transcripts (the vision and thread-gist passes, the user spine addressed
sideways, a typo), and neither a send nor a stop may be steerable by channel: a
delegate turn shares its numeric turn id with a user turn, so a stop that quietly
ignored the channel would cancel the user's own running turn.

Drives the real Flask routes (``authed_client``: real blueprints, real SQLite,
real ``TurnSerializerService`` and ``TurnExecutionService``) — no collaborator is
substituted. The send tests use an empty message as the control, which the
handler rejects before any turn (and so any LLM call) can start, proving the
request got past scope resolution. One more test keeps the readable-channel
allowlist in step with the delegate abilities the registry actually holds.
"""

import sqlite3
import threading
import time

import pytest
from flask.testing import FlaskClient

from abilities._delegate import DelegateAbility
from abilities._registry import AbilityRegistry
from configs.channels.user import UserConfig
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution

pytestmark = pytest.mark.unit

_MULTIPART = "multipart/form-data"
_DELEGATE_CHANNEL_PREFIX = "delegate:"


def _join_turn_threads(timeout_s: float = 10.0) -> None:
    """Join any turn a send started on a daemon thread, so a regression that lets
    a steered send through fails THIS test instead of leaking a live turn into the
    next one."""
    deadline = time.monotonic() + timeout_s
    for thread in threading.enumerate():
        if thread.name.startswith("turn-"):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


@pytest.mark.parametrize(
    "query",
    ["channel=user", "channel=delegate:vision", "channel=delegate:thread_gist", "channel=bogus", "channel="],
    ids=["user-spine", "vision-delegate", "gist-delegate", "unknown", "empty"],
)
def test_read_by_channel_refuses_everything_but_a_watchable_delegate(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], query: str,
) -> None:
    """A channel that is not one of the watchable delegates — including the user
    channel itself, the delegates that write no transcript, a typo and an empty
    value — is a 400, never a read."""
    client, _db, _store = authed_client

    response = client.get(f"/api/threads/1?{query}")

    assert response.status_code == 400


@pytest.mark.parametrize(
    "channel",
    ["delegate:web_search", "delegate:web_browse", "delegate:pim", "delegate:code_agent"],
)
def test_watchable_delegate_channel_reads_as_a_quiet_empty_block_when_the_turn_is_unknown(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], channel: str,
) -> None:
    """The four delegates with a transcript worth watching are readable by
    channel. A turn that does not exist (never ran, or already purged) is an
    empty block addressed by that channel with no type — not a 404 the side panel
    would have to special-case."""
    client, _db, _store = authed_client

    response = client.get(f"/api/threads/424242?channel={channel}")

    assert response.status_code == 200
    block = (response.get_json() or {})["result"]
    assert block["channel"] == channel and block["type"] is None
    assert block["messages"] == []


def test_read_naming_both_type_and_channel_is_refused(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """Addressing a turn by type AND channel is ambiguous, so it is a 400 — while
    each form alone is accepted."""
    client, _db, _store = authed_client

    assert client.get("/api/threads/1?type=user&channel=delegate:web_search").status_code == 400
    assert client.get("/api/threads/1?type=user").status_code == 200
    assert client.get("/api/threads/1?channel=delegate:web_search").status_code == 200


@pytest.mark.parametrize("where", ["form", "query"])
def test_send_cannot_be_addressed_by_channel(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], where: str,
) -> None:
    """A send is addressed by type. Naming a delegate channel — in the form body
    or the query string — is a 400 rather than landing a message on a delegate
    channel only its caller drives. The same send without a channel gets past
    scope resolution (it stops at the empty-message check, before any turn)."""
    client, _db, _store = authed_client
    url = "/api/threads/-1"

    try:
        control = client.post(url, data={"text": ""}, content_type=_MULTIPART)
        steered = (
            client.post(url, data={"text": "hi", "channel": "delegate:web_search"}, content_type=_MULTIPART)
            if where == "form"
            else client.post(f"{url}?channel=delegate:web_search", data={"text": "hi"}, content_type=_MULTIPART)
        )
        assert control.status_code == 422
        assert steered.status_code == 400
    finally:
        _join_turn_threads()


def test_stop_cannot_be_addressed_by_channel(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A stop is addressed by type. A delegate turn shares its numeric id with a
    user turn, so a stop that ignored ``channel=`` would cancel the USER's running
    turn 7 while the caller believed it was stopping a delegate. It is a 400 and
    the user turn keeps running; the same stop without a channel does reach it."""
    client, _db, _store = authed_client
    channel = UserConfig().channel
    assert MessageProcessor(UserConfig(), 7).turn_execution_service.open() is not None

    steered = client.delete("/api/threads/7?channel=delegate:web_search")

    assert steered.status_code == 400
    running = TurnExecution.latest(channel, 7)
    assert running is not None and running.state == TurnExecution.WORKING
    assert running.ended_at is None and not running.cancel_requested

    plain = client.delete("/api/threads/7")

    assert plain.status_code == 200 and (plain.get_json() or {})["result"]["cancelled"] is True
    stopped = TurnExecution.latest(channel, 7)
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED


def test_readable_delegate_channels_are_exactly_the_registered_delegate_abilities() -> None:
    """Each delegate ability's child turn runs on ``delegate:<its name>``, and the
    channels a client may read are exactly those. A delegate added without its
    channel could not be followed from its pill; a channel left behind after its
    ability went would invite reads of a turn nothing writes."""
    registered = {
        f"{_DELEGATE_CHANNEL_PREFIX}{ability.NAME}"
        for ability in AbilityRegistry.all()
        if isinstance(ability, DelegateAbility)
    }

    assert registered == WATCHABLE_DELEGATE_CHANNELS
