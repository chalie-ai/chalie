"""AskUser — the one service every ask that waits on a user's reply goes through.

A caller says what to ask (one classmethod per kind: a permission ask, a
browser hand-off, a question with answers to pick from, or a sign-in code)
and what to do with the answer (a callback); it never knows which screens exist
or whether anyone is there. An ask is made whether or not a UI is open, has no time limit, and
lives in memory only: a restart loses every open ask, and the runs waiting on
them end with the process.

Opening an ask stores it under one lock and broadcasts ``ask_opened``; the
first reply, or a cancel, takes it, broadcasts ``ask_closed`` and runs the
callback on the thread that answered. Nothing here blocks: waiting is the
caller's business — it waits in its own thread and cancels when its run stops.

An ask's ``data`` holds only what every UI may see. A reply's value is never
logged, framed, listed or put in an exception message: log lines name the ask
id and kind only.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from exceptions import AskReplyInvalid
from models.ws_message import WsMessage
from services.time_utils import utc_now
from services.websocket import Websocket

logger = logging.getLogger(__name__)

AskKind = Literal["permission", "handoff", "question", "code"]


@dataclass(frozen=True)
class AskReply:
    """How an ask ended. ``answered`` is False when it was cancelled — dismissed,
    or the caller's run stopped — and ``value`` is then ``None``; otherwise
    ``value`` is the kind's answer."""

    answered: bool
    value: object | None


AskCallback = Callable[[AskReply], None]


class Origin(BaseModel):
    """The interactive turn an ask belongs to — plain data about who is asking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    turn_id: int
    forked: bool


class PermissionAsk(BaseModel):
    """May a run take a gated action? Answered with a bool: allowed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str
    summary: str
    origin: Origin | None


class HandoffAsk(BaseModel):
    """A run hands its browser tab to the user until they finish. Answered with ``None``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tab_id: str
    reason: str
    origin: Origin | None


class QuestionAnswer(BaseModel):
    """One answer a question offers: a short label and an optional line about it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    description: str = ""


class QuestionAsk(BaseModel):
    """A run asks the user one question. Answered with a :class:`QuestionReply`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answers: tuple[QuestionAnswer, ...]
    multi_select: bool
    origin: Origin | None


class QuestionReply(BaseModel):
    """The answer to a question: the labels picked, or the user's own text (stripped)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: tuple[str, ...] = ()
    other: str | None = None

    @field_validator("other")
    @classmethod
    def _strip_other(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()

    @model_validator(mode="after")
    def _one_of(self) -> Self:
        if bool(self.labels) == (self.other is not None):
            raise ValueError("give either 'labels' or 'other'")
        return self


class CodeAsk(BaseModel):
    """A sign-in needs the user's 2FA code. Answered with a :class:`CodeReply`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: Literal["unifi"]
    origin: Origin | None


class CodeReply(BaseModel):
    """The answer to a code ask: the code the user typed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str


@dataclass(frozen=True)
class OpenAsk:
    """One ask still waiting for a reply."""

    ask_id: str
    kind: AskKind
    data: PermissionAsk | HandoffAsk | QuestionAsk | CodeAsk
    asked_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ask_id": self.ask_id,
            "kind": self.kind,
            "data": self.data.model_dump(mode="json"),
            "asked_at": self.asked_at,
        }


def _accepts(ask: OpenAsk, value: object) -> bool:
    """Whether *value* answers *ask*."""
    if isinstance(ask.data, PermissionAsk):
        return isinstance(value, bool)
    if isinstance(ask.data, HandoffAsk):
        return value is None
    if isinstance(ask.data, CodeAsk):
        if not isinstance(value, CodeReply):
            return False
        return len(value.code) == 6 and value.code.isascii() and value.code.isdigit()
    if not isinstance(value, QuestionReply):
        return False
    if value.other is not None:
        return value.other != ""
    offered = {answer.label for answer in ask.data.answers}
    picked = set(value.labels)
    fits_mode = len(value.labels) >= 1 if ask.data.multi_select else len(value.labels) == 1
    return fits_mode and len(picked) == len(value.labels) and picked <= offered


class AskUser:
    """Open asks, keyed by ask id in the order they opened. Process-global."""

    _open: ClassVar[dict[str, tuple[OpenAsk, AskCallback]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def permission(cls, params: PermissionAsk, callback: AskCallback) -> str:
        """Ask whether a run may take a gated action; the reply's value is a bool: allowed."""
        return cls._ask("permission", params, callback)

    @classmethod
    def handoff(cls, params: HandoffAsk, callback: AskCallback) -> str:
        """Hand a browser tab to the user; the reply (value ``None``) says they are done."""
        return cls._ask("handoff", params, callback)

    @classmethod
    def question(cls, params: QuestionAsk, callback: AskCallback) -> str:
        """Ask the user a question; the reply's value is a :class:`QuestionReply`."""
        return cls._ask("question", params, callback)

    @classmethod
    def code(cls, params: CodeAsk, callback: AskCallback) -> str:
        """Ask for a sign-in code; the reply's value is a :class:`CodeReply`."""
        return cls._ask("code", params, callback)

    @classmethod
    def reply(cls, ask_id: str, value: object) -> bool:
        """Answer ask *ask_id*; the first reply wins. False when it is not open.

        A value wrong for the ask's kind raises :class:`AskReplyInvalid` and
        leaves the ask open."""
        with cls._lock:
            entry = cls._open.get(ask_id)
            if entry is None:
                return False
            if not _accepts(entry[0], value):
                raise AskReplyInvalid(f"A {entry[0].kind} ask cannot take that value")
            del cls._open[ask_id]
        cls._close(entry, AskReply(answered=True, value=value), "answered")
        return True

    @classmethod
    def cancel(cls, ask_id: str) -> bool:
        """Close ask *ask_id* unanswered. False when it is not open."""
        with cls._lock:
            entry = cls._open.pop(ask_id, None)
        if entry is None:
            return False
        cls._close(entry, AskReply(answered=False, value=None), "cancelled")
        return True

    @classmethod
    def open_asks(cls) -> list[OpenAsk]:
        """Every open ask, oldest first."""
        with cls._lock:
            return [ask for ask, _ in cls._open.values()]

    @classmethod
    def open_handoff_on(cls, tab_id: str) -> bool:
        """Whether a hand-off ask is open on tab *tab_id*."""
        return any(
            isinstance(ask.data, HandoffAsk) and ask.data.tab_id == tab_id
            for ask in cls.open_asks()
        )

    @classmethod
    def _ask(cls, kind: AskKind, params: PermissionAsk | HandoffAsk | QuestionAsk | CodeAsk, callback: AskCallback) -> str:
        ask = OpenAsk(
            ask_id=uuid.uuid4().hex,
            kind=kind,
            data=params,
            asked_at=utc_now().isoformat(timespec="seconds"),
        )
        with cls._lock:
            cls._open[ask.ask_id] = (ask, callback)
        try:
            Websocket.broadcast(WsMessage(type="ask_opened", **ask.to_dict()))
        except BaseException:
            # No UI heard of it and the caller never gets its id to cancel it.
            with cls._lock:
                cls._open.pop(ask.ask_id, None)
            raise
        logger.info("[AskUser] opened ask_id=%s kind=%s", ask.ask_id, kind)
        return ask.ask_id

    @staticmethod
    def _close(entry: tuple[OpenAsk, AskCallback], reply: AskReply, how: str) -> None:
        ask, callback = entry
        logger.info("[AskUser] %s ask_id=%s kind=%s", how, ask.ask_id, ask.kind)
        try:
            Websocket.broadcast(WsMessage(type="ask_closed", ask_id=ask.ask_id))
        finally:
            # The caller is waiting on this whatever happened to the frame.
            try:
                callback(reply)
            except Exception:
                logger.exception("[AskUser] callback failed ask_id=%s kind=%s caller=%r", ask.ask_id, ask.kind, callback)
                raise
