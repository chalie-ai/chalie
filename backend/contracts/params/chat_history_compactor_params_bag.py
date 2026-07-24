"""ChatHistoryCompactorParamsBag — the typed input contract of the
``chat_history_compactor`` ability.

This ability never reads parameters from the call: it is dispatched
programmatically by :meth:`DispatchService._dispatch_compaction` when the
context window fills, with its input assembled internally via
``_fit_compaction_input``. The bag is intentionally empty — its existence
is the contract: the dispatch seam no longer hands the ability a raw
``dict[str, object]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ChatHistoryCompactorParamsBag(ParamBag):
    """Empty bag — ``chat_history_compactor`` accepts no parameters."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()
