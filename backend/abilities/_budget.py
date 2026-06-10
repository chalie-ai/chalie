"""``BudgetCappedAbility`` — shared per-turn call-budget scaffolding.

``save_graph`` and ``save_pattern`` both cap how many times they may fire within
a single processor turn, storing the running count on the calling processor
(``self.mp``) so the budget survives across dispatches in one ACT loop. They had
duplicated the same three moves: read the counter with a ``getattr`` default,
short-circuit with a ``budget_exceeded`` result when the cap is hit, and bump the
counter after a successful store.

This mixin owns those three moves. A concrete ability sets two ``ClassVar``s —
the counter attribute name on the processor and the cap — and calls
:meth:`budget_exceeded` at the top of ``run()`` and :meth:`bump_budget` after a
successful write. The per-ability dedupe / decay tracking sets stay in each
ability (their semantics differ) — only the numeric budget lives here.

Listing ``ABC`` in the bases makes ``Ability.__init_subclass__`` skip this mixin
in the metadata probe; concrete subclasses are probed normally.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult


class BudgetCappedAbility(Ability, ABC):
    """Mixin giving an ability a per-turn call budget kept on ``self.mp``."""

    #: Attribute on the processor holding this ability's running call count.
    BUDGET_COUNTER_ATTR: ClassVar[str] = ""

    #: Maximum number of successful calls allowed per processor turn.
    BUDGET_CAP: ClassVar[int] = 0

    def _budget_count(self) -> int:
        """Current call count read off the processor (0 when absent / no mp)."""
        proc = self.mp
        if proc is None:
            return 0
        return getattr(proc, self.BUDGET_COUNTER_ATTR, 0)

    def budget_exceeded(self) -> ToolResult | None:
        """Return the ``budget_exceeded`` result when the cap is hit, else None."""
        if self._budget_count() >= self.BUDGET_CAP:
            return ToolResult.ok({"budget_exceeded": True, "tool": self.get_name()})
        return None

    def bump_budget(self) -> None:
        """Increment the processor's call counter for this ability by one."""
        proc = self.mp
        if proc is not None:
            setattr(proc, self.BUDGET_COUNTER_ATTR, self._budget_count() + 1)
