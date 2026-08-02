"""WebFetchParamsBag — the typed input contract of the ``web_fetch`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class WebFetchParamsBag(ParamBag):
    """``url`` — URL to fetch from, required. ``convert_to_markdown`` — convert
    HTML to markdown (default: True). ``timeout`` — fetch timeout in minutes
    (default: 15, clamp: 1-120)."""

    url: str
    convert_to_markdown: bool
    timeout: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        url = cls.require_str(params, Keys.url)
        if isinstance(url, ToolResult):
            return url
        convert_to_markdown = cls.bool_default(params, Keys.convert_to_markdown, default=True)
        if isinstance(convert_to_markdown, ToolResult):
            return convert_to_markdown
        timeout = cls.clamp_int(params, Keys.timeout, default=15, lo=1, hi=120)
        if isinstance(timeout, ToolResult):
            return timeout
        return cls(url=url, convert_to_markdown=convert_to_markdown, timeout=timeout)
