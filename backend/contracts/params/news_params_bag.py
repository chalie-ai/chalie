"""NewsParamsBag — the typed input contract of the ``news`` ability.

Action-less tool, so a single flat bag: ``query`` is the search text
(pre-gated present by ACTION_REQUIRED; the bag rejects the whitespace-only
residue) and ``category`` optionally narrows to one of :data:`CATEGORIES`.

``CATEGORIES`` is single-sourced here — the ability's INPUT_SCHEMA enum and
the bag's membership check both read it, so the model-facing schema can never
drift from what validation accepts. A present-but-invalid category is rejected
loudly (``invalid-param``), never silently degraded into an uncategorised
search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

CATEGORIES = ("tech", "business", "sports", "science", "entertainment", "us", "uk")


@dataclass(frozen=True, slots=True)
class NewsParamsBag(ParamBag):
    """``query`` — what to search for, required; ``category`` — optional
    :data:`CATEGORIES` member, ``None`` when absent (uncategorised search)."""

    query: str
    category: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        raw = params.get(Keys.category)
        if raw is None:
            category = None
        elif isinstance(raw, str) and raw in CATEGORIES:
            category = raw
        else:
            raise ToolParamError(
                f"'{raw}' is not a valid value for '{Keys.category}'.",
                code="invalid-param",
                hint=f"choose one of: {', '.join(CATEGORIES)}.",
                valid=CATEGORIES,
            )
        return cls(query=cls.require_str(params, Keys.query), category=category)
