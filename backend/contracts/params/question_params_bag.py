"""QuestionParamsBag — the typed input contract of the ``question`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from services.ask_user import QuestionAnswer


@dataclass(frozen=True, slots=True)
class QuestionParamsBag(ParamBag):
    """``question`` — the question, required. ``answers`` — 2 to 4 answers, each a
    label (required, unique) and an optional description, required.
    ``multi_select`` — whether more than one answer may be picked (default: False).
    Labels and descriptions are stripped; an absent description is ``""``."""

    question: str
    answers: tuple[QuestionAnswer, ...]
    multi_select: bool

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        question = cls.require_str(params, Keys.question)
        if isinstance(question, ToolResult):
            return question
        answers = cls._answers(params)
        if isinstance(answers, ToolResult):
            return answers
        multi_select = cls.bool_default(params, Keys.multi_select, default=False)
        if isinstance(multi_select, ToolResult):
            return multi_select
        return cls(question=question, answers=answers, multi_select=multi_select)

    @classmethod
    def _answers(cls, params: dict[str, object]) -> tuple[QuestionAnswer, ...] | ToolResult:
        raw = params.get(Keys.answers)
        if raw is None:
            return cls._missing(params, Keys.answers)
        if not isinstance(raw, list) or not 2 <= len(raw) <= 4:
            return ToolResult.err(
                "'answers' must list 2 to 4 answers.",
                code="invalid-param",
                hint="pass 'answers' as a list of 2 to 4 {\"label\": ..., \"description\": ...} objects.",
            )
        answers: list[QuestionAnswer] = []
        for i, item in enumerate(raw):
            label = item.get(Keys.label) if isinstance(item, dict) else None
            if not isinstance(label, str) or not label.strip():
                return ToolResult.err(
                    f"'answers' item at index {i} needs a 'label'.",
                    code="invalid-param",
                    hint="give each answer a non-empty 'label' string.",
                )
            label = label.strip()
            description = item.get(Keys.description)
            if description is not None and not isinstance(description, str):
                return ToolResult.err(
                    f"'answers' item at index {i}: 'description' must be text.",
                    code="invalid-param",
                    hint="pass 'description' as a string, or leave it out.",
                )
            if any(answer.label == label for answer in answers):
                return ToolResult.err(
                    f"'answers' has the label '{label}' twice.",
                    code="invalid-param",
                    hint="give each answer a different label.",
                )
            answers.append(QuestionAnswer(label=label, description=(description or "").strip()))
        return tuple(answers)
