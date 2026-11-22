"""Feature tests: each thread runs on the model and thinking level stored on it.

A thread's model and level live on its ``threads`` row. A new thread copies
them from its channel's latest thread; a channel's first thread has none and
runs on the default at ``auto``. Every turn reads its own thread's row; a model
that is unset or no longer configured falls back to the default, and the
default is then stored on the thread, so the fallback sticks. A change through
``/api/threads/settings/<turn_id>`` is a plain update of the row and applies
from the thread's next message; ``-1`` names the channel's latest thread.

These tests drive the real send endpoint, the real settings action, the real
provider delete and selection endpoints and the real ``MessageProcessor``
against the real migrated SQLite database. The only substitution is the LLM
network boundary (``services.provider_service.build_client``): it records which
provider row each chat call was built from and the thinking level the request
carried, then answers through the scripted double the turn tests share.
"""

import sqlite3
from collections.abc import Iterator
from typing import cast

import pytest
from flask.testing import FlaskClient

import services.provider_service as provider_service
from configs.channels.discovery import DiscoveryConfig
from configs.channels.user import UserConfig
from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.thread import Thread
from services.provider_api import ProviderApiRequest
from tests.test_turn_join import AuthedClient, _drain, _label_thread, _post, _Provider, _result

pytestmark = pytest.mark.unit

_CHANNEL = "user"

Call = tuple[int, ThinkingLevel]


class _Recorder:
    """Builds a client per provider config, as the real factory does, and
    records ``(provider id, thinking level)`` for every chat call one makes.
    Helper calls (thread labels, skill suggestions) are not chat calls and are
    not recorded: they are not the thread's turn."""

    def __init__(self) -> None:
        self.double = _Provider()
        self.calls: list[Call] = []

    def build(self, config: dict[str, object], *_a: object, **_k: object) -> "_Client":
        return _Client(self, cast("int", config["id"]))

    def since(self, mark: int) -> list[Call]:
        """The chat calls made after the first ``mark``; fails on none, so
        an assertion over them can never pass vacuously."""
        calls = self.calls[mark:]
        assert calls, "no chat call was made — nothing to check the model against"
        return calls


class _Client:
    def __init__(self, recorder: _Recorder, provider_id: int) -> None:
        self._recorder = recorder
        self._provider_id = provider_id

    def get_context_limit(self) -> int:
        return self._recorder.double.get_context_limit()

    def abort(self) -> None:
        self._recorder.double.abort()

    def send(self, dto: ProviderApiRequest) -> ProviderResponse:
        if dto.type is ProviderType.CHAT:
            self._recorder.calls.append((self._provider_id, dto.thinking_mode))
        return self._recorder.double.send(dto)


@pytest.fixture
def recorder(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    rec = _Recorder()
    monkeypatch.setattr(provider_service, "build_client", rec.build)
    yield rec
    rec.double.release.set()
    _drain()


def _start(client: FlaskClient, text: str) -> int:
    """Start a new thread, run it to completion and label it, so a later
    reply into it is the only caller of the provider; return its turn id."""
    turn_id = cast("int", _result(_post(client, -1, text))["turn_id"])
    _drain()
    _label_thread(turn_id)
    return turn_id


def _reply(client: FlaskClient, turn_id: int, text: str) -> None:
    _result(_post(client, turn_id, text))
    _drain()


def _stored(turn_id: int, channel: str = _CHANNEL) -> tuple[int | None, str]:
    thread = Thread.of(channel, turn_id)
    assert thread is not None, f"turn {turn_id} has no thread"
    return thread.provider_id, thread.thinking_level


def _settings(client: FlaskClient, turn_id: int) -> dict[str, object]:
    return _result(client.get(f"/api/threads/settings/{turn_id}"))


def _change(client: FlaskClient, turn_id: int, provider_id: int, level: str) -> dict[str, object]:
    return _result(
        client.post(f"/api/threads/settings/{turn_id}", json={"provider_id": provider_id, "thinking_level": level})
    )


def _select(client: FlaskClient, provider_id: int) -> None:
    resp = client.post("/api/providers/selected", json={"provider_id": provider_id})
    assert resp.status_code == 200, resp.get_data(as_text=True)


# ── a new thread ─────────────────────────────────────────────────────────────


def test_a_channel_first_thread_runs_on_the_default_and_keeps_it(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    """With no thread to copy, the first thread runs on the default at
    ``auto`` and the default is stored on it — so switching the default
    afterwards moves neither it nor the next new thread, which copies it."""
    client, _db, _store = authed_client
    default, other = models
    assert _settings(client, -1) == {"provider_id": default, "thinking_level": "auto", "turn_id": None}

    mark = len(recorder.calls)
    first = _start(client, "first question")

    assert _stored(first) == (default, "auto")
    assert {provider for provider, _level in recorder.since(mark)} == {default}

    _select(client, other)
    assert _settings(client, -1) == {"provider_id": default, "thinking_level": "auto", "turn_id": first}
    mark = len(recorder.calls)
    second = _start(client, "second question")

    assert _stored(second) == (default, "auto")
    assert {provider for provider, _level in recorder.since(mark)} == {default}


def test_a_new_thread_copies_the_settings_of_the_channel_latest_thread(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    """A new thread copies the thread with the highest turn id on its channel:
    a change to an older thread is not copied, a change to the latest is."""
    client, _db, _store = authed_client
    default, other = models
    older = _start(client, "older thread")
    latest = _start(client, "latest thread")
    _change(client, older, other, "high")

    copied_latest = _start(client, "copies the latest thread")
    assert _stored(copied_latest) == (default, "auto")

    _change(client, -1, other, "high")
    mark = len(recorder.calls)
    copied_change = _start(client, "copies the change")

    assert _stored(latest) == (default, "auto"), "-1 changed a thread other than the latest"
    assert _stored(copied_latest) == (other, "high")
    assert _stored(copied_change) == (other, "high")
    assert set(recorder.since(mark)) == {(other, ThinkingLevel.HIGH)}


def test_a_new_thread_on_another_channel_does_not_copy_the_chat_settings(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    """Copying stays within a channel: a background loop's first thread runs
    on the default at ``auto`` whatever the chat's latest thread runs on."""
    client, _db, _store = authed_client
    default, other = models
    _start(client, "a chat thread")
    _change(client, -1, other, "high")

    mark = len(recorder.calls)
    loop = MessageProcessor(DiscoveryConfig(), raw_input="look around")
    loop.begin()
    loop.result()
    _drain()

    assert _stored(loop.turn_id, DiscoveryConfig().channel) == (default, "auto")
    assert {provider for provider, _level in recorder.since(mark)} == {default}


# ── an existing thread ───────────────────────────────────────────────────────


def test_a_settings_change_applies_from_the_thread_next_message(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    """A change is a plain update of the thread's row; its next message runs
    on it."""
    client, _db, _store = authed_client
    _default, other = models
    turn_id = _start(client, "question")

    changed = _change(client, turn_id, other, "high")

    assert changed == {"provider_id": other, "thinking_level": "high", "turn_id": turn_id}
    assert _stored(turn_id) == (other, "high")
    assert _settings(client, turn_id) == changed
    mark = len(recorder.calls)
    _reply(client, turn_id, "follow-up")
    assert set(recorder.since(mark)) == {(other, ThinkingLevel.HIGH)}


@pytest.mark.parametrize("level", ["medium", "high"])
def test_a_medium_or_high_thread_forces_that_level(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int], level: str,
) -> None:
    client, _db, _store = authed_client
    default, _other = models
    turn_id = _start(client, "question")
    _change(client, turn_id, default, level)

    mark = len(recorder.calls)
    _reply(client, turn_id, "follow-up")

    assert set(recorder.since(mark)) == {(default, ThinkingLevel(level))}


def test_an_auto_thread_leaves_the_level_to_the_deliberation_gate(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    """``auto`` forces nothing: the turn runs at whatever level the
    deliberation gate picked for it."""
    client, _db, _store = authed_client
    default, _other = models
    turn_id = _start(client, "question")
    _change(client, turn_id, default, "high")
    _change(client, turn_id, default, "auto")

    mark = len(recorder.calls)
    reply = MessageProcessor.process(UserConfig(), raw_input="follow-up", turn_id=turn_id)
    reply.result()
    _drain()

    assert reply.thinking_override is None
    assert set(recorder.since(mark)) == {(default, ThinkingLevel(reply.thinking_level))}


# ── a model that is no longer configured ─────────────────────────────────────


@pytest.mark.parametrize("gone", ["deleted", "never-configured"])
def test_a_thread_whose_model_is_gone_runs_on_the_default_and_keeps_it(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int], gone: str,
) -> None:
    """A thread whose model was deleted, or was set to an id no provider has,
    keeps its level and runs on the default; the default is then stored on
    the thread, so it stays on it even when the default changes later."""
    client, _db, _store = authed_client
    default, other = models
    turn_id = _start(client, "question")
    if gone == "deleted":
        _change(client, turn_id, other, "high")
        assert client.delete(f"/api/providers/{other}").status_code == 204
        missing = other
    else:
        missing = other + 1000
        _change(client, turn_id, missing, "high")

    assert _stored(turn_id) == (missing, "high")
    assert _settings(client, turn_id) == {"provider_id": default, "thinking_level": "high", "turn_id": turn_id}
    mark = len(recorder.calls)
    _reply(client, turn_id, "follow-up")

    assert set(recorder.since(mark)) == {(default, ThinkingLevel.HIGH)}
    assert _stored(turn_id) == (default, "high"), "the fallback was not stored on the thread"


# ── refusals ─────────────────────────────────────────────────────────────────


def test_settings_of_a_thread_that_does_not_exist_are_not_found(
    authed_client: AuthedClient, models: tuple[int, int],
) -> None:
    """A missing thread, and ``-1`` on a channel with no thread yet, cannot be
    changed; the refusal creates no thread."""
    client, _db, _store = authed_client
    default, _other = models
    change = {"provider_id": default, "thinking_level": "high"}

    assert client.get("/api/threads/settings/42").status_code == 404
    assert client.post("/api/threads/settings/42", json=change).status_code == 404
    assert client.post("/api/threads/settings/-1", json=change).status_code == 404
    assert Thread.all().get() == [], "a refused change must create no thread"


def test_a_settings_change_to_an_unknown_level_is_refused_and_changes_nothing(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int],
) -> None:
    client, _db, _store = authed_client
    _default, other = models
    turn_id = _start(client, "question")
    before = _stored(turn_id)

    resp = client.post(f"/api/threads/settings/{turn_id}", json={"provider_id": other, "thinking_level": "low"})

    assert resp.status_code == 422, resp.get_data(as_text=True)
    assert _stored(turn_id) == before


# ── the memory step ──────────────────────────────────────────────────────────


def test_the_memory_step_runs_on_the_settings_of_the_chat_latest_thread(
    authed_client: AuthedClient, recorder: _Recorder, models: tuple[int, int], real_memory_step: None,
) -> None:
    """The memory step that follows a settled thread opens a thread of its own
    on the same channel, so like any new thread it copies the channel's latest
    thread: it runs on the model and level picked for the main chat box."""
    client, db, _store = authed_client
    _default, other = models
    _start(client, "question")
    _change(client, -1, other, "high")

    mark = len(recorder.calls)
    asked = _start(client, "plan my week")
    memory_turns = {
        int(r["turn_id"])
        for r in db.execute(
            "SELECT turn_id FROM transcript WHERE channel = ? AND role = 'memory' AND turn_id > ?", (_CHANNEL, asked),
        )
    }

    assert memory_turns, "no memory step followed the thread — nothing to check"
    assert set(recorder.since(mark)) == {(other, ThinkingLevel.HIGH)}
    assert {_stored(memory_turn) for memory_turn in memory_turns} == {(other, "high")}
