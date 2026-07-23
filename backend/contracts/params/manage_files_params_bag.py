"""ManageFilesParamsBag — the typed input contract of the ``manage_files`` ability.

The multi-action shape: :class:`ManageFilesParamsBag` is a plain router — the
ability's boundary type, never built by the seam — whose ``from_params`` reads
``action`` once and fans out to the per-action subclass. Each leaf is a frozen
dataclass declaring exactly the fields its action reads, with language-level
required-ness; a leaf receives the full params dict and simply ignores
``action`` — routing is the parent's job alone. ``run()`` narrows the router
type back to a leaf via ``isinstance``, so a handler handed the wrong leaf
fails loudly at the attribute, never silently.

The router also owns the cross-action ``permission_code`` type guard — the
original handler checked it before dispatching to any action, so the bag does
the same, raising ``invalid-param`` for a non-string value regardless of which
action is being requested. The leaves then extract and strip the value for the
two actions that consume it; the delete leaf ignores it entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

_ACTIONS = ("create", "delete", "update_permission")


class ManageFilesParamsBag(ParamBag):
    """Router and boundary type: ``ManageFilesAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> ManageFilesParamsBag:
        # Cross-action type guard — the original handler enforced this before
        # dispatching to any action, so the bag does the same here.
        perm_raw = params.get(Keys.permission_code)
        if perm_raw is not None and not isinstance(perm_raw, str):
            raise ToolParamError(
                f"permission_code must be a string, got {type(perm_raw).__name__}.",
                code="invalid-param",
                hint="pass the octal code as a string, e.g. '0644'.",
            )

        match params.get(Keys.action):
            case "create":
                return ManageFilesCreateParams.from_params(params)
            case "delete":
                return ManageFilesDeleteParams.from_params(params)
            case "update_permission":
                return ManageFilesUpdatePermissionParams.from_params(params)
            case unknown:
                raise ToolParamError(
                    f"Unknown manage_files action: {unknown!r}.",
                    code="unknown-action",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class ManageFilesCreateParams(ManageFilesParamsBag):
    """``path`` is required; ``permission_code`` is optional and defaults to
    ``""`` (the handler then picks 0644/0755 based on ``is_folder``)."""

    path: str
    permission_code: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            path=cls.require_str(params, Keys.path),
            permission_code=cls.str_default(params, Keys.permission_code, default="").strip(),
        )


@dataclass(frozen=True, slots=True)
class ManageFilesDeleteParams(ManageFilesParamsBag):
    """Only ``path`` — the delete action never reads ``permission_code``."""

    path: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(path=cls.require_str(params, Keys.path))


@dataclass(frozen=True, slots=True)
class ManageFilesUpdatePermissionParams(ManageFilesParamsBag):
    """Both ``path`` and ``permission_code`` are required — the dispatcher's
    ``ACTION_REQUIRED`` pre-gate already enforced this; the bag re-asserts the
    path non-blank contract via ``require_str``. ``permission_code`` keeps the
    handler's historical shape: stripped, ``""`` when absent (the handler then
    rejects the blank as ``invalid-param``)."""

    path: str
    permission_code: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            path=cls.require_str(params, Keys.path),
            permission_code=cls.str_default(params, Keys.permission_code, default="").strip(),
        )
