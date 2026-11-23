# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""QuestionAbility — ask the user one question and wait for their answer.

The question goes through :class:`AskUser` as a ``question`` ask, so every
screen the user has open shows it and the first answer wins. The run waits
with no time limit; a stopped run closes the ask, and so does a dismissal.
Neither the question nor the answer is ever logged.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.question_params_bag import QuestionParamsBag
from services.ask_user import AskReply, AskUser, Origin, QuestionAsk, QuestionReply

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

_PARK_POLL = 0.25  # seconds between stop checks while the question waits


class QuestionAbility(Ability[QuestionParamsBag]):
    NAME: ClassVar[str] = "question"
    PARAMS: ClassVar[type[ParamBag] | None] = QuestionParamsBag
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.CONVERSATION
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.question, Keys.answers)}
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("ask user", "multiple choice", "clarifying question", "pick an option")

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.question: {"type": "string", "description": "The question, as the user will read it."},
            Keys.answers: {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "description": "2 to 4 answers for the user to pick from. Do not add an 'Other' answer: the user can always type their own.",
                "items": {
                    "type": "object",
                    "properties": {
                        Keys.label: {"type": "string", "description": "The answer, in a few words. It comes back as-is when picked."},
                        Keys.description: {"type": "string", "description": "Optional one line that explains the answer."},
                    },
                    "required": [Keys.label],
                },
            },
            Keys.multi_select: {"type": "boolean", "description": "true lets the user pick more than one answer. Defaults to false."},
        },
        "required": [Keys.question, Keys.answers],
    }

    def get_summary(self) -> str:
        return (
            "Ask the user one question and wait for their answer. Give 2 to 4 answers to pick from; "
            "the user can always type their own answer instead. Returns {\"answer\": \"<label>\"}, "
            "or {\"answers\": [\"<label>\", ...]} when multi_select is true, "
            "or {\"other\": \"<their text>\"} when they typed their own."
        )

    def get_examples(self) -> list[str]:
        return [
            "ask me which one I prefer",
            "give me a few options to choose from",
            "check with me before you pick",
            "let me choose between the options",
            "ask me a multiple choice question",
            "which of these should we go with?",
        ]

    def get_search_tooltip(self) -> str:
        return "Ask the user a question with answers to pick from"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: QuestionParamsBag) -> ToolResult:
        mp = self._bound_mp()
        done = threading.Event()
        outcome: dict[str, AskReply] = {}

        def on_reply(reply: AskReply) -> None:
            outcome["reply"] = reply
            done.set()

        origin = mp.origin
        ask = QuestionAsk(
            question=params.question,
            answers=params.answers,
            multi_select=params.multi_select,
            origin=None if origin is None else Origin.model_validate(origin),
        )
        ask_id = AskUser.question(ask, on_reply)
        try:
            while not done.wait(_PARK_POLL):
                if mp.turn_execution_service.should_stop():
                    AskUser.cancel(ask_id)
        finally:
            # An exception out of the wait still closes the ask; once it is
            # answered or cancelled this is a no-op.
            AskUser.cancel(ask_id)
        return self._result(outcome["reply"], params.multi_select)

    @staticmethod
    def _result(reply: AskReply, multi_select: bool) -> ToolResult:
        if not reply.answered:
            return ToolResult.err("Failed - the question closed before the user answered.", code="question-unanswered")
        answer = reply.value
        if not isinstance(answer, QuestionReply):
            raise TypeError(f"a question ask was answered with a {type(answer).__name__}")
        if answer.other is not None:
            return ToolResult.ok({"other": answer.other})
        if multi_select:
            return ToolResult.ok({"answers": list(answer.labels)})
        return ToolResult.ok({"answer": answer.labels[0]})

    def _bound_mp(self) -> MessageProcessor:
        if self.mp is None:
            raise RuntimeError("the question ability requires a bound MessageProcessor")
        return self.mp
