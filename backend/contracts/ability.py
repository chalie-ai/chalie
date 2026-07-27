"""AbilityContract — the full contract every tool ability satisfies.

Declares the complete surface of the ability family:

- ``NAME`` — **required** ClassVar: the tool's registry name, the single
  authority every caller reads (the registry raises on a missing, empty, or
  duplicate NAME at load time).
- ``get_summary`` / ``get_examples`` / ``get_search_tooltip`` /
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

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from abilities._result import ToolResult
    from contracts.params.param_bag import ParamBag

# Mirrors abilities._ability.B: what run() receives — the ability's typed
# ParamBag once migrated, the raw params dict until then. The PEP 696 default
# keeps every bare ``AbilityContract`` annotation valid during the migration.
# Contravariant because it only ever appears as run()'s input.
B_contra = TypeVar("B_contra", contravariant=True, default=Any)


class AbilityContract(Protocol[B_contra]):
    """Anything that describes and runs one tool."""

    # The ability's typed input contract; None = unmigrated (run() takes the raw
    # params dict). See abilities._ability.Ability.PARAMS.
    PARAMS: ClassVar[type[ParamBag] | None]

    # The tool's registry name — the single authority every caller reads.
    # See abilities._ability.Ability.NAME.
    NAME: ClassVar[str]

    def get_summary(self) -> str: ...

    def get_examples(self) -> list[str]: ...

    def get_search_tooltip(self) -> str: ...

    def get_parameters(self) -> dict[str, object]: ...

    def run(self, params: B_contra) -> ToolResult: ...

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
    def enrich_rich_payload(cls, payload: dict[str, object]) -> dict[str, object]: ...
