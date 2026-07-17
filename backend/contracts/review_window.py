"""ReviewWindowAbilityContract — the full contract every review-window tool satisfies.

Composes :class:`contracts.ability.AbilityContract` (the review tools inherit
``Ability``) and adds the methods that
:class:`abilities._review_window.ReviewWindowAbility` introduces:

- ``_fetch`` / ``_row`` — **required**. Each concrete review tool supplies them
  (enforced by the ABC's ``@abstractmethod``).
- ``run`` — **shared**. The ABC owns the entire windowed-query flow, so concrete
  review tools inherit ``run`` (declared on :class:`AbilityContract`) and fill
  in only the two hooks.
"""

from __future__ import annotations

from typing import Protocol

from contracts.ability import AbilityContract


class ReviewWindowAbilityContract(AbilityContract, Protocol):
    """A review tool that returns DB rows within ±N minutes of an anchor."""

    def _fetch(
        self, lo: str, hi: str, params: "dict[str, object]"
    ) -> "list[dict[str, object]]": ...

    def _row(
        self, rec: "dict[str, object]", ordinal: int
    ) -> "dict[str, object]": ...
