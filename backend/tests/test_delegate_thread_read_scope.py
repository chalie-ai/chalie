"""Feature test: which threads a client may read or stop by channel, and which it may not.

A delegate turn has no routing ``type`` of its own, so the single-turn read
(``GET /api/threads/<id>``) and the stop (``DELETE /api/threads/<id>``) also
accept a ``channel`` — but only for the delegates whose turns write a
transcript worth watching: a subagent's own channel (``delegate:web_search:12``)
or its kind's bare one. Everything else must stay refused loudly: reading an
arbitrary channel would expose internal transcripts (the vision and thread-gist
passes, the user spine addressed sideways, a typo), a send may never land on a
channel only a delegate's caller drives, and the feed and the per-turn actions
are addressed by type alone. Turn ids are counted per channel: every subagent
runs as turn 1 of its own channel, as the chat's first turn is turn 1 of the
user channel, so a stop by channel must reach that subagent's turn and never
the user's turn or another subagent's turn of the same id.

Drives the real Flask routes (``authed_client``: real blueprints, real SQLite,
real ``TurnSerializerService`` and ``TurnExecutionService``) and observes the
stop's frames through the REAL socket registry — no collaborator is substituted.
Running turns are opened through the real ``TurnExecutionService`` of an inert
processor, so no model is ever called. The send tests use an empty message as the
control, which the handler rejects before any turn (and so any LLM call) can
start, proving the request got past scope resolution. One more test keeps the
watchable-channel allowlist in step with the delegate abilities the registry
actually holds.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient

from abilities._delegate import DelegateAbility
from abilities._registry import AbilityRegistry
from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution
from tests.helpers import drain_background_turns
from tests._delegates import listening

pytestmark = pytest.mark.unit

_MULTIPART = "multipart/form-data"
_DELEGATE_CHANNEL_PREFIX = "delegate:"
_DELEGATE_CHANNEL = "delegate:web_search:12"
_OTHER_DELEGATE_CHANNEL = "delegate:web_search:13"
_TURN = 1


def _open_user_and_delegate_turns() -> None:
    """Open a running user turn and two running ``web_search`` subagent turns,
    each on its own channel, all three turn 1 — the turns a channel tells
    apart."""
    search = WebSearchConfig(PolicyChannel.CHAT)
    for config in (UserConfig(), search.on_channel(_DELEGATE_CHANNEL), search.on_channel(_OTHER_DELEGATE_CHANNEL)):
        assert MessageProcessor(config, _TURN).turn_execution_service.open() is not None


def _assert_still_running(channel: str) -> None:
    running = TurnExecution.latest(channel, _TURN)
    assert running is not None and running.state == TurnExecution.WORKING
    assert running.ended_at is None and not running.cancel_requested


@pytest.mark.parametrize("method", ["GET", "DELETE"], ids=["read", "stop"])
@pytest.mark.parametrize(
    "query",
    [
        "channel=user", "channel=delegate:vision", "channel=delegate:vision:12", "channel=delegate:thread_gist",
        "channel=delegate:web_search:", "channel=delegate:web_search:twelve", "channel=delegate:web_search:012",
        "channel=delegate:web_search:12:1", "channel=bogus", "channel=",
    ],
    ids=[
        "user-spine", "vision-delegate", "vision-subagent", "gist-delegate", "no-agent-id", "word-agent-id",
        "padded-agent-id", "nested-agent-id", "unknown", "empty",
    ],
)
def test_read_or_stop_by_channel_refuses_everything_but_a_watchable_delegate(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], method: str, query: str,
) -> None:
    """A channel that is not one of the watchable delegates — including the user
    channel itself, the delegates that write no transcript, a subagent's channel
    spelled with no agent number, a malformed one or one nested past it, a typo
    and an empty value — is a 400, never a read and never a stop: every running
    turn of that id keeps running."""
    client, _db, _store = authed_client
    _open_user_and_delegate_turns()

    response = client.open(f"/api/threads/{_TURN}?{query}", method=method)

    assert response.status_code == 400
    _assert_still_running(UserConfig().channel)
    _assert_still_running(_DELEGATE_CHANNEL)
    _assert_still_running(_OTHER_DELEGATE_CHANNEL)


@pytest.mark.parametrize(
    "channel",
    [
        "delegate:web_search:12", "delegate:web_browse:12", "delegate:pim:12", "delegate:code_agent:12",
        "delegate:subtask:12", "delegate:web_search", "delegate:web_browse", "delegate:pim",
        "delegate:code_agent", "delegate:subtask",
    ],
)
def test_watchable_delegate_channel_reads_as_a_quiet_empty_block_when_the_turn_is_unknown(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], channel: str,
) -> None:
    """Every subagent of the five delegate kinds with a transcript worth
    watching is readable by its own channel, and each kind's bare channel —
    where its subagents ran before each had its own — stays readable. A turn
    that does not exist (never ran, or already purged) is an empty block
    addressed by that channel with no type — not a 404 the side panel would
    have to special-case."""
    client, _db, _store = authed_client

    response = client.get(f"/api/threads/424242?channel={channel}")

    assert response.status_code == 200
    block = (response.get_json() or {})["result"]
    assert block["channel"] == channel and block["type"] is None
    assert block["messages"] == []


def test_read_or_stop_naming_both_type_and_channel_is_refused(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """Addressing a turn by type AND channel is ambiguous, so a read or a stop
    naming both is a 400 that stops neither running turn — while each form alone
    is accepted for a read."""
    client, _db, _store = authed_client
    _open_user_and_delegate_turns()
    both = f"/api/threads/{_TURN}?type=user&channel={_DELEGATE_CHANNEL}"

    assert client.get(both).status_code == 400
    assert client.delete(both).status_code == 400
    _assert_still_running(UserConfig().channel)
    _assert_still_running(_DELEGATE_CHANNEL)
    _assert_still_running(_OTHER_DELEGATE_CHANNEL)
    assert client.get(f"/api/threads/{_TURN}?type=user").status_code == 200
    assert client.get(f"/api/threads/{_TURN}?channel={_DELEGATE_CHANNEL}").status_code == 200


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
            client.post(url, data={"text": "hi", "channel": _DELEGATE_CHANNEL}, content_type=_MULTIPART)
            if where == "form"
            else client.post(f"{url}?channel={_DELEGATE_CHANNEL}", data={"text": "hi"}, content_type=_MULTIPART)
        )
        assert control.status_code == 422
        assert steered.status_code == 400
    finally:
        drain_background_turns()


@pytest.mark.parametrize(
    ("path", "joiner"),
    [("/api/threads/all", "?"), (f"/api/threads/batch?id[]={_TURN}", "&"), ("/api/threads/settings/-1", "?")],
    ids=["feed", "batch", "settings"],
)
def test_feed_and_per_turn_actions_cannot_be_addressed_by_channel(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], path: str, joiner: str,
) -> None:
    """Only the single-turn read and the stop name a channel. The feed and the
    per-turn actions are addressed by type, so naming a delegate channel there is
    a 400 — while the same request without it succeeds."""
    client, _db, _store = authed_client

    assert client.get(path).status_code == 200
    assert client.get(f"{path}{joiner}channel={_DELEGATE_CHANNEL}").status_code == 400


def test_stop_by_channel_cancels_the_delegate_turn_never_the_users_same_numbered_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A subagent's turn, the user's turn and another subagent's turn are each
    turn 1 of their own channel. Stopping the subagent by its channel cancels
    THAT turn — stamped cancelled and closed — and sends exactly one frame,
    addressed by the channel with no ``type`` (a ``type`` would land it in the
    user's thread), while the user's running turn and the other subagent's are
    left untouched."""
    client, _db, _store = authed_client
    _open_user_and_delegate_turns()

    with listening() as listener:
        response = client.delete(f"/api/threads/{_TURN}?channel={_DELEGATE_CHANNEL}")

    assert response.status_code == 200
    assert (response.get_json() or {})["result"]["cancelled"] is True
    stopped = TurnExecution.latest(_DELEGATE_CHANNEL, _TURN)
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED
    assert stopped.cancel_requested and stopped.ended_at is not None
    _assert_still_running(UserConfig().channel)
    _assert_still_running(_OTHER_DELEGATE_CHANNEL)
    assert len(listener.frames) == 1
    frame = listener.frames[0]
    assert "type" not in frame
    assert (frame["channel"], frame["turn_id"], frame["state"]) == (_DELEGATE_CHANNEL, _TURN, TurnExecution.CANCELLED)


def test_stop_by_channel_with_no_running_delegate_turn_is_a_quiet_ack(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A delegate that already finished (or never ran) is a harmless
    ``no_active_turn`` ack with no frame — and never falls through to the user's
    running turn of the same number."""
    client, _db, _store = authed_client
    assert MessageProcessor(UserConfig(), _TURN).turn_execution_service.open() is not None

    with listening() as listener:
        response = client.delete(f"/api/threads/{_TURN}?channel={_DELEGATE_CHANNEL}")

    assert response.status_code == 200
    result = (response.get_json() or {})["result"]
    assert result["reason"] == "no_active_turn" and not result["cancelled"]
    assert listener.frames == []
    _assert_still_running(UserConfig().channel)


def test_readable_delegate_channels_are_exactly_the_registered_delegate_abilities() -> None:
    """Each delegate ability's subagents run on channels under
    ``delegate:<its name>``, and the kinds a client may read are exactly those.
    A delegate added without its channel could not be followed from its pill; a
    channel left behind after its ability went would invite reads of a turn
    nothing writes."""
    registered = {
        f"{_DELEGATE_CHANNEL_PREFIX}{ability.NAME}"
        for ability in AbilityRegistry.all()
        if isinstance(ability, DelegateAbility)
    }

    assert registered == WATCHABLE_DELEGATE_CHANNELS
