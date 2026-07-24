"""DelegateParamsBag — the shared typed input contract of every single-input
delegate tool (``pim``, ``code_agent``, ``web_search``, ``web_browse``).

A delegate is a subagent-as-tool: its whole wire contract is one required
plain-language ``instructions`` string handed to the inner processor as its
user prompt. The parameter name is standardized across the delegate family so
the model learns ONE calling shape; what differs per tool is the schema
DESCRIPTION, not the parameter name. A delegate needing extra inputs
(``vision``'s ``image`` document id) declares its own bag instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class DelegateParamsBag(ParamBag):
    """``instructions`` — the plain-language task for the delegate, required
    (pre-gated present by ACTION_REQUIRED; the bag rejects the whitespace-only
    residue)."""

    instructions: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(instructions=cls.require_str(params, Keys.instructions))
