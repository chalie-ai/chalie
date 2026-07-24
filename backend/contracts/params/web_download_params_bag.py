"""WebDownloadParamsBag — the typed input contract of the ``web_download``
ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class WebDownloadParamsBag(ParamBag):
    """``url`` — URL to download from, required. ``timeout`` — download timeout
    in minutes (default: 15, clamp: 1-120)."""

    url: str
    timeout: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        url = cls.require_str(params, Keys.url)
        if isinstance(url, ToolResult):
            return url
        timeout = cls.clamp_int(params, Keys.timeout, default=15, lo=1, hi=120)
        if isinstance(timeout, ToolResult):
            return timeout
        return cls(url=url, timeout=timeout)
