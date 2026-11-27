# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared harness for the feature tests of the subagent tools that run a turn of
their own (``web_search``, ``web_browse``, ``code_agent``, ``pim``,
``subtask``).

A subagent's turn runs on its own thread beside its caller's — at once for a
caller that is replied to later, one after the other for a caller that waits —
so one ordered script cannot say whose provider call comes next.
:class:`ChannelProvider` answers each call from the script of the channel that
made it instead, so the order the threads reach the provider in cannot change
what any of them is told.

Everything here drives real production code against the real, fully-migrated
SQLite database. The one substitution is the LLM network boundary,
:data:`BUILD_CLIENT`, the factory ``ProviderService`` builds its transport
client with. Turns that must happen in a set order are pinned by holding a
provider call on a ``threading.Event`` (:class:`Held`), never by sleeping.
"""

from __future__ import annotations

import json
import logging
import socketserver
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NamedTuple, NoReturn, cast
from unittest.mock import patch

from flask.testing import FlaskClient

import services.vault_service as _vault_mod
from configs.channels.code_agent import CodeAgentConfig
from configs.channels.pim import PimConfig
from configs.channels.web_browse import WebBrowseConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.channels import Channel, delegate_agent_id
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.agent import Agent
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.processor_config import ProcessorConfig
from services.tool_call_service import ToolCallService
from services.vault_service import _vault_state, get_vault_service
from services.websocket import Websocket
from tests.helpers import NO_PLAN, is_planning_request

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py).
BUILD_CLIENT = "services.provider_service.build_client"

USER = Channel.USER.value
WAIT_S = 15.0

#: How many ended subagents the ``earlier_agents`` fixture records before a
#: test, so the first agent the test spawns is number ``EARLIER_AGENTS + 1``
#: while its turn is number 1.
EARLIER_AGENTS = 100

# Each subagent's own config, keyed by the route its turns are answered from —
# the subagent's tool name.
_DELEGATE_CONFIGS: dict[str, Callable[[PolicyChannel], ProcessorConfig]] = {
    "web_search": WebSearchConfig,
    "web_browse": WebBrowseConfig,
    "code_agent": CodeAgentConfig,
    "pim": PimConfig,
}
ROUTES = ("chat", "subtask", "background", *_DELEGATE_CONFIGS)


def delegate_channel(tool_name: str) -> str:
    """The bare channel of the *tool_name* subagent kind, which each of its
    subagents' own channels is spelled under."""
    return _DELEGATE_CONFIGS[tool_name](PolicyChannel.CHAT).channel


def tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


def calls(*tools: dict[str, object]) -> ProviderResponse:
    """A model step that makes *tools* and says nothing."""
    return ProviderResponse(text="", model="m", tool_calls=list(tools))


def text(answer: str) -> ProviderResponse:
    """A model step that answers with *answer* and calls nothing."""
    return ProviderResponse(text=answer, model="m", tool_calls=None)


class Held:
    """A script entry that holds its turn's provider call until the test
    releases it, then answers with *outcome* — or raises it, for a failing
    call — after running *then* on the held turn's own thread. ``entered`` is
    set the moment the call arrives, so the test can act on the turn while it
    is mid-call; ``opened`` records whether the wait ended because the gate
    opened rather than because it timed out, so a test fails on a gate that
    never opened instead of on its downstream symptom. ``thread`` is the held
    turn's own thread — the one that, once the turn ends, sends its late reply
    — so a test can wait for exactly that turn to finish."""

    _GATE_TIMEOUT_S = 10.0

    def __init__(self, outcome: ProviderResponse | Exception, then: Callable[[], None] | None = None) -> None:
        self._outcome = outcome
        self._then = then
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.opened = False
        self.thread: threading.Thread | None = None

    def __call__(self) -> ProviderResponse:
        self.thread = threading.current_thread()
        self.entered.set()
        self.opened = self.gate.wait(timeout=self._GATE_TIMEOUT_S)
        if self._then is not None:
            self._then()
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def raises(exc: Exception) -> Callable[[], ProviderResponse]:
    """A script entry whose provider call fails with *exc*."""
    def entry() -> ProviderResponse:
        raise exc
    return entry


class ProcessDeath(BaseException):
    """The process dying mid-call. Raised from a provider call it is no
    ``Exception``, so nothing in the turn catches it: the turn's thread ends
    with its execution row still open and nothing reported — what a killed
    process leaves behind for the next boot to sweep."""


def dies() -> NoReturn:
    """A script entry whose provider call is the last thing its process does."""
    raise ProcessDeath


@contextmanager
def process_deaths_expected() -> Iterator[None]:
    """Let turn threads die of :class:`ProcessDeath` inside the block without
    it counting as an unhandled thread exception; any other exception a thread
    dies of still reaches the hook that was there before."""
    previous = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not ProcessDeath:
            previous(args)

    threading.excepthook = hook
    try:
        yield
    finally:
        threading.excepthook = previous


Entry = ProviderResponse | Callable[[], ProviderResponse]


class Request(NamedTuple):
    """One provider call as the provider saw it."""

    route: str
    text: str
    tools: frozenset[str]
    system: str = ""
    definitions: tuple[dict[str, object], ...] = ()

    def definition(self, tool_name: str) -> dict[str, object] | None:
        """The definition of *tool_name* as this call sent it, or ``None``
        when it was not offered."""
        return next((d for d in self.definitions if d.get("name") == tool_name), None)


def _as_read(request_text: str) -> str:
    """The messages of one recorded request as the model reads them: each
    message's text — a plain string, or the text of each of its parts — joined
    by newlines, with the JSON escaping of the record undone."""
    out: list[str] = []
    for message in json.loads(request_text):
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        else:
            out.extend(str(part.get("text", "")) for part in content or [] if isinstance(part, dict))
    return "\n".join(out)


def _shape(dto: ProviderRequest) -> tuple[str, frozenset[str]]:
    """A request's messages as one JSON text, and the names of the tools it offers."""
    return json.dumps(dto.messages, default=str), frozenset(str(t["name"]) for t in dto.tools or [])


class ChannelProvider:
    """Answers each provider call from the script of the channel that made it.

    ``ProviderRequest`` names no channel, so the route is read off what the
    request carries, most specific first: a subagent's turn is sent under that
    subagent's own system prompt (its first line, read from the subagent's real
    config, so a reworded prompt moves the route with it); a request that
    offers no ``find_tools`` is background work a finished turn spawns (the
    memory step among it); a subtask's conversation opens with its "Task from
    Chalie" row; anything else is a chat-shaped turn — the person's chat, a
    thread reply, or discovery.

    Each script is consumed in order under a lock; past its end a chat-shaped
    or subagent route gets an empty answer and background work gets "nothing
    new", which is how the memory step ends a pass with nothing to save. A test
    asserts its scripts were used up (:meth:`unused`), so a response consumed by
    the wrong turn fails it. A callable entry runs outside the lock, so a held
    call never blocks another route."""

    _TERMINAL = ProviderResponse(text="", model="overflow", tool_calls=None)
    _NOTHING_NEW = ProviderResponse(text="nothing new", model="overflow", tool_calls=None)

    def __init__(self, **scripts: list[Entry]) -> None:
        unknown = set(scripts) - set(ROUTES)
        if unknown:
            raise ValueError(f"no such route: {sorted(unknown)}; routes are {ROUTES}")
        self._scripts = {route: list(script) for route, script in scripts.items()}
        self._held = [e for script in self._scripts.values() for e in script if isinstance(e, Held)]
        self._heads = {
            route: config(PolicyChannel.CHAT).system_prompt.splitlines()[0]
            for route, config in _DELEGATE_CONFIGS.items()
        }
        self._lock = threading.Lock()
        self.requests: list[Request] = []

    def get_context_limit(self) -> int:
        return 200000

    def abort(self) -> None:
        """A stopped turn's watcher aborts its call in flight; a held call
        still waits for its own gate, which the test opens."""

    def route(self, dto: ProviderRequest) -> str:
        """The route *dto* is answered from."""
        return self._route(dto, *_shape(dto))

    def _route(self, dto: ProviderRequest, text: str, names: frozenset[str]) -> str:
        for route, head in self._heads.items():
            if dto.system.startswith(head):
                return route
        if "find_tools" not in names:
            return "background"
        if "Task from Chalie" in text:
            return "subtask"
        return "chat"

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        text, names = _shape(dto)
        route = self._route(dto, text, names)
        with self._lock:
            self.requests.append(Request(route, text, names, dto.system, tuple(dto.tools or ())))
            script = self._scripts.get(route)
            if not script:
                return self._NOTHING_NEW if route == "background" else self._TERMINAL
            entry = script.pop(0)
        return entry() if callable(entry) else entry

    def texts(self, route: str) -> list[str]:
        """The messages of every call routed to *route*, in arrival order."""
        with self._lock:
            return [r.text for r in self.requests if r.route == route]

    def systems(self, route: str) -> list[str]:
        """The system prompt of every call routed to *route*, in arrival order."""
        with self._lock:
            return [r.system for r in self.requests if r.route == route]

    def read(self, route: str) -> list[str]:
        """What the model read on every call routed to *route*, in arrival
        order: the text of each message, unescaped, one message after another."""
        return [_as_read(t) for t in self.texts(route)]

    def offered(self, route: str) -> list[frozenset[str]]:
        """The tools offered on every call routed to *route*, in arrival order."""
        with self._lock:
            return [r.tools for r in self.requests if r.route == route]

    def definitions(self, route: str, tool_name: str, said: str) -> list[dict[str, object] | None]:
        """The definition of *tool_name* as it was sent on every call routed to
        *route* whose messages contain *said* — ``None`` on a call it was not
        offered on — in arrival order. *said* tells apart turns that share a
        route, such as a scheduled task and the person's chat."""
        with self._lock:
            return [r.definition(tool_name) for r in self.requests if r.route == route and said in r.text]

    def last(self, route: str) -> Request:
        """The latest call routed to *route*. Read from inside a script entry,
        it is the call that entry answers, as long as no other turn calls on
        that route meanwhile — true while every other turn waits on this one."""
        with self._lock:
            return next(r for r in reversed(self.requests) if r.route == route)

    def unused(self) -> dict[str, int]:
        """Every route whose script still holds entries, with how many."""
        with self._lock:
            return {route: len(script) for route, script in self._scripts.items() if script}

    def release_all(self) -> None:
        """Open every held entry's gate, consumed or not, so no turn waits on a
        test that has already failed."""
        for held in self._held:
            held.gate.set()


def as_sent(definition: dict[str, object] | None) -> tuple[str, object, frozenset[str]]:
    """A tool definition as a request carried it: its description, the choices
    its ``action`` parameter offers, and the names of all its parameters."""
    assert definition is not None, "the tool was not sent on this request"
    schema = cast("dict[str, object]", definition["input_schema"])
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    action = properties.get("action", {})
    return str(definition["description"]), action.get("enum"), frozenset(properties)


def turn_threads() -> list[threading.Thread]:
    """The live threads turns run on, and the background work a finished turn
    spawns (post-turn work, the memory step)."""
    return [
        t for t in threading.enumerate()
        if t.name in ("skill-suggest", "thread-gist") or t.name.startswith(("turn-", "memory-step-"))
    ]


def drain(timeout_s: float = WAIT_S) -> None:
    """Join every turn and the background work turns spawn — including a turn
    a joined one opened on its way out, such as a subagent's reply — and fail
    loudly if any is still running when the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = turn_threads()
        if not pending:
            return
        for t in pending:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
    leftover = [t.name for t in turn_threads()]
    assert not leftover, f"still running after {timeout_s:.0f}s: {leftover}"


@contextmanager
def patched(provider: ChannelProvider) -> Iterator[None]:
    """Serve every provider call from *provider* for the block, and before
    leaving it open every held call and wait for every turn to end, so none
    outlives this test's provider and database."""
    with patch(BUILD_CLIENT, return_value=provider):
        try:
            yield
        finally:
            provider.release_all()
            drain()


def wait(event: threading.Event, what: str) -> None:
    assert event.wait(timeout=WAIT_S), f"timed out after {WAIT_S:.0f}s waiting for {what}"


def begin(config: ProcessorConfig, raw_input: str, turn_id: int = -1) -> MessageProcessor:
    """Start a real turn on *config* on its own thread and return its processor."""
    mp = MessageProcessor(config, turn_id=turn_id, raw_input=raw_input)  # inert (I2)
    mp.begin()
    return mp


class WarningWatch(logging.Handler):
    """A real logging handler that notes every WARNING-or-worse record it is
    given and, when given *on_warning*, runs it on the thread that warned —
    before the warning call returns, so a test can act at exactly that point of
    the work that warned."""

    def __init__(self, on_warning: Callable[[], None] | None = None) -> None:
        super().__init__(level=logging.WARNING)
        self._on_warning = on_warning
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self._on_warning is not None:
            self._on_warning()


@contextmanager
def watching_warnings(
    on_warning: Callable[[], None] | None = None, source_name: str = ToolCallService.__module__,
) -> Iterator[WarningWatch]:
    """Watch the warnings of one service — by default the one that owns a
    call's frames — and only them. Unrelated services warn on their own (the
    on-device model loaders do whenever their files are absent, as on a fresh
    checkout), and such a warning must neither trigger *on_warning* nor count
    as a report of the failure under test."""
    watch = WarningWatch(on_warning)
    source = logging.getLogger(source_name)
    source.addHandler(watch)
    try:
        yield watch
    finally:
        source.removeHandler(watch)


class RecordingClient:
    """A real socket-registry subscriber: implements the registry's
    ``send(str)`` Protocol and captures every broadcast frame, handing each to
    *on_frame* on the thread that broadcast it."""

    def __init__(self, on_frame: Callable[[dict[str, object]], None] | None = None) -> None:
        self.frames: list[dict[str, object]] = []
        self._on_frame = on_frame

    def close(self) -> None:
        """Nothing to release: the captured frames stay readable after the registry lets go."""

    def send(self, data: str) -> None:
        frame = json.loads(data)
        self.frames.append(frame)
        if self._on_frame is not None:
            self._on_frame(frame)


@contextmanager
def listening(on_frame: Callable[[dict[str, object]], None] | None = None) -> Iterator[RecordingClient]:
    """A client connected to the real socket registry for the duration of the
    block, recording every frame broadcast meanwhile."""
    listener = RecordingClient(on_frame)
    Websocket._connect(listener)
    try:
        yield listener
    finally:
        Websocket._disconnect(listener)


class _ImapPeer(socketserver.StreamRequestHandler):
    """The least an IMAP server says for a mailbox to connect: a greeting, its
    capabilities, a login accepted and a logout. Any other command is refused,
    so a connect that starts asking for more fails instead of passing on an
    answer this peer never meant."""

    def handle(self) -> None:
        self.wfile.write(b"* OK [CAPABILITY IMAP4rev1] ready\r\n")
        for raw in self.rfile:
            tag, _, rest = raw.decode().strip().partition(" ")
            command = rest.split(" ", 1)[0].upper()
            if command == "CAPABILITY":
                self.wfile.write(f"* CAPABILITY IMAP4rev1\r\n{tag} OK done\r\n".encode())
            elif command == "LOGIN":
                self.wfile.write(f"{tag} OK done\r\n".encode())
            elif command == "LOGOUT":
                self.wfile.write(f"* BYE\r\n{tag} OK done\r\n".encode())
                return
            else:
                self.wfile.write(f"{tag} BAD unexpected\r\n".encode())


class _ImapServer(socketserver.ThreadingTCPServer):
    daemon_threads = True


def _forget_vault() -> None:
    """Drop the process's unlocked vault key and its service, so no other
    test inherits the vault one test unlocked."""
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@contextmanager
def connected_mail(client: FlaskClient) -> Iterator[None]:
    """A mailbox connected the way a person connects one — through the
    capability setup route, which probes the mail server and connects — for
    the duration of the block. The mail server is an IMAP peer on loopback;
    the vault the credentials are stored in is this test's own, unlocked for
    the block and dropped after it. On the way out the mailbox is disconnected
    through the product's own route, which removes the stored credentials."""
    server = _ImapServer(("127.0.0.1", 0), _ImapPeer)
    serving = threading.Thread(target=server.serve_forever, name="imap-peer", daemon=True)
    serving.start()
    _forget_vault()
    vault = get_vault_service()
    vault.initialize("test-vault-password")
    vault.unlock("test-vault-password")
    try:
        host, port = cast("tuple[str, int]", server.server_address)
        setup = client.post(
            "/api/capabilities/setup/mail",
            json={"email": "person@example.test", "password": "app-password",
                  "imap_host": host, "imap_port": port, "imap_tls": "0"},
        )
        assert setup.status_code == 204, setup.data
        for first_sync in [t for t in threading.enumerate() if t.name == "cap-init-mail"]:
            first_sync.join(timeout=WAIT_S)
            assert not first_sync.is_alive(), "the mailbox's first sync did not end"
        yield
    finally:
        client.post("/api/capabilities/disconnect/mail")
        server.shutdown()
        server.server_close()
        serving.join(timeout=WAIT_S)
        _forget_vault()


def run_turn(
    provider: ChannelProvider,
    config: ProcessorConfig,
    raw_input: str,
    on_frame: Callable[[dict[str, object]], None] | None = None,
    turn_id: int = -1,
) -> tuple[MessageProcessor, list[dict[str, object]]]:
    """Drive a real turn on *config* to its end against *provider*, then open
    every held call — so a subagent held there finishes only after its caller's
    turn has ended — and wait for every turn that follows, a subagent's reply
    among them. *turn_id* sends into an existing turn (a thread reply) or names
    the turn a caller that owns its ids runs as (a schedule's). Returns the
    processor and every frame the socket registry saw meanwhile (each also
    handed to *on_frame*, on the thread that broadcast it)."""
    with listening(on_frame) as listener, patched(provider):
        mp = begin(config, raw_input, turn_id)
        mp.result()
        provider.release_all()
        drain()
    return mp, listener.frames


def calls_of(db: sqlite3.Connection, tool_name: str) -> list[sqlite3.Row]:
    """Every call of *tool_name*, oldest first."""
    return db.execute(
        "SELECT id, state, result, ended_at, created_at FROM tool_calls WHERE tool_name = ? ORDER BY id",
        (tool_name,),
    ).fetchall()


class Child(NamedTuple):
    """The turn a subagent runs: the channel of its own, and its turn there."""

    channel: str
    turn_id: int


def child(db: sqlite3.Connection, kind: str, call_id: int) -> Child:
    """The turn of the subagent the call *call_id* started — read off the
    turn's first input row, which carries the call's id, on the own channel of
    a subagent of *kind* (that kind's bare channel)."""
    rows = db.execute(
        "SELECT channel, turn_id FROM transcript WHERE tool_call_id = ? ORDER BY id", (str(call_id),),
    ).fetchall()
    found = next(
        (r for r in rows if delegate_agent_id(r["channel"]) is not None and r["channel"].rpartition(":")[0] == kind),
        None,
    )
    assert found is not None, f"no {kind} subagent's turn carries call {call_id}"
    return Child(cast("str", found["channel"]), cast("int", found["turn_id"]))


def kind_rows(db: sqlite3.Connection, kind: str) -> int:
    """How many transcript rows the subagents of *kind* (that kind's bare
    channel) wrote, each on its own channel, all told."""
    return cast("int", db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel >= ? AND channel < ?", (f"{kind}:", f"{kind};"),
    ).fetchone()[0])


def agent_number(channel: str) -> int:
    """The number the model knows the subagent running on *channel* by — its
    row's ``agent_id``; the row must exist, with *channel* as its own. Read on
    the calling thread's own connection, so a provider entry may call it."""
    agent_id = delegate_agent_id(channel)
    assert agent_id is not None, f"{channel} is no subagent's own channel"
    agent = Agent.by_id(agent_id)
    assert agent is not None and agent.channel == channel, f"no agents row runs on {channel}"
    return agent.agent_id


def replies(db: sqlite3.Connection, call_id: int, channel: str = USER) -> list[sqlite3.Row]:
    """Every reply row sent back for the call *call_id*: an input row on the
    caller's *channel* linked to that call."""
    return db.execute(
        "SELECT content, turn_id, joined FROM transcript "
        "WHERE channel = ? AND role = 'user' AND tool_call_id = ? ORDER BY id",
        (channel, str(call_id)),
    ).fetchall()


def linked_rows(db: sqlite3.Connection, channel: str) -> list[sqlite3.Row]:
    """Every row on *channel* that carries a tool call's id, oldest first."""
    return db.execute(
        "SELECT role, content, turn_id, tool_call_id FROM transcript "
        "WHERE channel = ? AND tool_call_id IS NOT NULL ORDER BY id",
        (channel,),
    ).fetchall()


def assistant_texts(db: sqlite3.Connection, channel: str, turn_id: int) -> list[str]:
    """Every assistant row of one turn, oldest first."""
    return [
        cast("str", r["content"]) for r in db.execute(
            "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant' ORDER BY id",
            (channel, turn_id),
        ).fetchall()
    ]


def execution(channel: str, turn_id: int) -> TurnExecution:
    """The latest execution of one turn, which must exist."""
    found = TurnExecution.latest(channel, turn_id)
    assert found is not None, f"no turn execution for {channel} turn {turn_id}"
    return found


def block(client: FlaskClient, path: str) -> dict[str, object]:
    """GET *path* and return the thread block under ``result``."""
    response = client.get(path)
    assert response.status_code == 200, response.data
    return cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])


def messages(thread: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", thread["messages"])


def chips(thread: dict[str, object]) -> list[dict[str, object]]:
    return [
        chip for message in messages(thread)
        for chip in cast("list[dict[str, object]]", message.get("tool_calls") or [])
    ]


def chip(thread: dict[str, object], tool_name: str) -> dict[str, object]:
    return next(c for c in chips(thread) if c["tool_name"] == tool_name)
