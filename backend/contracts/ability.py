"""AbilityContract — the full contract every tool ability satisfies.

Declares the complete method surface of the ability family:

- ``get_name`` / ``get_summary`` / ``get_examples`` / ``get_search_tooltip`` /
  ``get_parameters`` / ``run`` — **required**. Each concrete ability supplies
  them (enforced by :class:`abilities._ability.Ability`'s ``@abstractmethod`` at
  instantiation).
- ``get_follow_up`` / ``get_input_schema`` / ``param`` / ``classify_action`` /
  ``enrich_rich_payload`` — **shared**. Supplied by the ``Ability`` ABC base,
  which every ability inherits, so subclasses get them for free.

The ABC contributes the shared helpers and leaves the getters + ``run``
abstract; concrete abilities inherit the helpers and MUST supply the required
ones (enforced by the ABC's ``@abstractmethod``, mirroring PHP interface
enforcement — Python Protocols are static-only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from abilities._result import ToolResult


class AbilityContract(Protocol):
    """Anything that describes and runs one tool."""

    def get_name(self) -> str: ...

    def get_summary(self) -> str: ...

    def get_examples(self) -> list[str]: ...

    def get_search_tooltip(self) -> str: ...

    def get_parameters(self) -> dict[str, object]: ...

    def run(self, params: dict[str, object]) -> ToolResult: ...

    def get_follow_up(self, tr: ToolResult) -> str: ...

    def get_input_schema(self) -> dict[str, object]: ...

    def param(
        self,
        params: dict[str, object],
        name: str,
        *,
        default: object = ...,
        choices: "tuple[object, ...] | None" = None,
        clamp: "tuple[object, ...] | None" = None,
        required: bool = False,
    ) -> object: ...

    def classify_action(self, params: dict[str, object]) -> "str | None": ...

    @classmethod
    def enrich_rich_payload(cls, payload: dict[str, object], created_at: "datetime | str | None") -> dict[str, object]: ...
