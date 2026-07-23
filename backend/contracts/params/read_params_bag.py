"""ReadParamsBag — the typed input contract of the ``read`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ReadParamsBag(ParamBag):
    """``source`` — the read target (URL or filesystem path), required; the
    ``url`` / ``path`` / ``link`` … aliases a model naturally emits are healed
    to it upstream at the dispatch seam via the shared ``VARIANTS[Keys.source]``
    ladder, before this bag is built. ``max_chars`` — cap on the returned text:
    default 20000, clamped to 100–100000."""

    source: str
    max_chars: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            source=cls.require_str(params, Keys.source),
            max_chars=cls.clamp_int(params, Keys.max_chars, default=20000, lo=100, hi=100_000),
        )
