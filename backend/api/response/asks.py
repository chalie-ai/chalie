"""Response DTOs for the asks endpoints — the asks still waiting on the user."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .policies import PermissionOriginResponse
from .response import Response


class PermissionAskDataResponse(Response):
    """A permission ask: may a run take the gated action ``action_id``?"""

    action_id: str
    summary: str = Field(description="The model's one-line account of the call.")
    origin: PermissionOriginResponse | None = Field(description="The interactive turn the ask belongs to; null when it belongs to none.")


class HandoffAskDataResponse(Response):
    """A hand-off ask: a browsing run waits while the user takes over its tab."""

    tab_id: str
    reason: str
    origin: PermissionOriginResponse | None = Field(description="The interactive turn the ask belongs to; null when it belongs to none.")


class QuestionAnswerResponse(Response):
    """One answer a question offers."""

    label: str
    description: str = Field(description="One line about the answer; empty when the model gave none.")


class QuestionAskDataResponse(Response):
    """A question ask: a run waits for the user to pick an answer or type their own."""

    question: str
    answers: list[QuestionAnswerResponse] = Field(description="2 to 4 answers to pick from, in the model's order.")
    multi_select: bool = Field(description="Whether more than one answer may be picked.")
    origin: PermissionOriginResponse | None = Field(description="The interactive turn the ask belongs to; null when it belongs to none.")


class CodeAskDataResponse(Response):
    """A code ask: a sign-in to ``service`` waits for the user's 2FA code."""

    service: Literal["unifi"]
    origin: PermissionOriginResponse | None = Field(description="The interactive turn the ask belongs to; null when it belongs to none.")


class OpenAskResponse(Response):
    """One ask still waiting for a reply: the same fields as its ``ask_opened`` frame."""

    ask_id: str
    kind: Literal["permission", "handoff", "question", "code"]
    data: PermissionAskDataResponse | HandoffAskDataResponse | QuestionAskDataResponse | CodeAskDataResponse
    asked_at: str = Field(description="When the ask opened, ISO-8601 UTC with its offset (e.g. '2026-10-07T06:32:11+00:00').")
