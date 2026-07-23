"""WeatherParamsBag — the typed input contract of the ``weather`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class WeatherParamsBag(ParamBag):
    """``location`` — city or place name (e.g. 'Malta', 'London'); optional,
    defaults to empty string. When absent the ability falls back to device
    telemetry coordinates. The empty-string default preserves the original
    behaviour where a missing key was treated as "no location param"."""

    location: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            location=cls.str_default(params, Keys.location, default="").strip(),
        )
