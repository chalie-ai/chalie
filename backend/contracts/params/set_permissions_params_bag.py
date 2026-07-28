"""SetPermissionsParamsBag — the typed input contract of the
``set_permissions`` ability.

A single-shot bag: the ability does exactly one thing (change file permissions),
so there is no router, no ``action`` field, no match statement. The bag declares
two required fields — ``path`` and ``permission_code`` — as frozen dataclass
attributes, with the contract enforced at construction time by ``from_params``.

``permission_code`` keeps the type-vs-absence distinction that ``require_str``
alone would collapse: an absent key is a ``missing-params`` error, but a
present non-string value (``0644`` sent as an int — the mistake this guard
exists for, since a leading-zero octal is only meaningful as a string) is an
``invalid-param`` error naming the type actually received, so the model is told
to re-send it quoted rather than to supply a param it already sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class SetPermissionsParamsBag(ParamBag):
    """``path`` — file or directory, required. ``permission_code`` — octal
    mode as a string (e.g. ``'0644'``), required."""

    path: str
    permission_code: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        # Type-vs-absence guard for permission_code — preserve the distinction
        # that require_str alone would collapse to a single missing-params error.
        perm_raw = params.get(Keys.permission_code)
        if perm_raw is not None and not isinstance(perm_raw, str):
            return ToolResult.err(
                f"permission_code must be a string, got {type(perm_raw).__name__}.",
                code="invalid-param",
                hint="pass the octal code as a string, e.g. '0644'.",
            )

        path = cls.require_str(params, Keys.path)
        if isinstance(path, ToolResult):
            return path

        permission_code = cls.require_str(params, Keys.permission_code)
        if isinstance(permission_code, ToolResult):
            return permission_code

        return cls(path=path, permission_code=permission_code)
