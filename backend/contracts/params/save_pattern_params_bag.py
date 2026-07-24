"""SavePatternParamsBag — the typed input contract of the ``save_pattern``
ability.

The validation ORDER and rules are frozen (name presence → snake_case regex →
frequency set → summary presence → min-2 evidence); the bag preserves that
ladder exactly, one :class:`~exceptions.exception.ToolParamError` per rung.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
VALID_FREQUENCIES = frozenset({"daily", "weekly", "weekday", "weekend", "ad-hoc"})

# One-line recovery hint carrying a minimal, fully-valid example so a weak model
# can self-correct without re-reading the schema (parity with save_graph).
EXAMPLE_HINT = (
    "use a snake_case name, e.g. name=morning_run frequency=weekday "
    "summary=user runs on weekdays evidence_transcript_ids=[12, 18]"
)

_REQUIRED = ("name", "frequency", "summary", "evidence_transcript_ids")


@dataclass(frozen=True, slots=True)
class SavePatternParamsBag(ParamBag):
    """``name_`` — stripped snake_case pattern id; ``frequency`` — one of
    :data:`VALID_FREQUENCIES`; ``summary`` — stripped one-liner; ``evidence``
    — the (min-2) evidence transcript id list, passed to the store verbatim;
    ``time_anchor`` — optional anchor, ``""`` when absent or falsy."""

    name_: str
    frequency: str
    summary: str
    evidence: list[object]
    time_anchor: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        # The dispatcher pre-gate is truthiness-based, so a whitespace-only name
        # slips past it as a non-empty string and must be rejected here.
        name = (str(params.get(Keys.name_) or "")).strip()
        if not name:
            raise ToolParamError(
                "Missing required parameter(s): name.",
                code="missing-params",
                valid=_REQUIRED,
            )
        if not NAME_PATTERN.fullmatch(name):
            raise ToolParamError(
                f"Invalid pattern name {name!r}; must be snake_case.",
                code="invalid-param",
                hint=EXAMPLE_HINT,
            )

        frequency = params.get(Keys.frequency, "")
        if frequency not in VALID_FREQUENCIES:
            raise ToolParamError(
                f"Unknown frequency {frequency!r}; not a recognised cadence.",
                code="invalid-param",
                valid=tuple(sorted(VALID_FREQUENCIES)),
                hint=EXAMPLE_HINT,
            )

        summary = (str(params.get(Keys.summary) or "")).strip()
        if not summary:
            raise ToolParamError(
                "Missing required parameter(s): summary.",
                code="missing-params",
                valid=_REQUIRED,
            )

        evidence = params.get(Keys.evidence_transcript_ids)
        n_evidence = len(evidence) if isinstance(evidence, list) else 0
        if not isinstance(evidence, list) or n_evidence < 2:
            raise ToolParamError(
                f"At least 2 evidence transcript ids are required; got {n_evidence}.",
                code="invalid-param",
                hint="provide at least 2 evidence_transcript_ids from the transcript window",
            )

        anchor_raw = params.get(Keys.time_anchor, "")
        return cls(
            name_=name,
            frequency=str(frequency),
            summary=summary,
            evidence=list(evidence),
            time_anchor=str(anchor_raw) if anchor_raw else "",
        )
