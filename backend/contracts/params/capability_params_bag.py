"""CapabilityParamsBag — the shared input bag of the ``CapabilityAbility``
family (``calendar``, ``email``, ``home``, ``ubiquiti``).

A capability tool's own typed input is just ``action``; every other parameter
belongs to the connected capability's handler, whose per-action schema is owned
by the capabilities layer and consumed as a dict. Enumerating those fields here
would duplicate that schema only to rebuild the dict at the handler boundary —
so the bag carries them as ``extra``, an opaque passthrough map. Framework
params (underscore-prefixed) and ``action`` itself are filtered out at
construction, exactly the set the base's dispatch used to strip.

``action`` is coerced with ``str()`` rather than gated by ``opt_str``: a
non-string action flows on to the unknown-action error, whose ``valid=`` ladder
names every real action — a better self-correction signal than a bare
``invalid-param``. ``None`` means the model omitted it; the base substitutes
the subclass's ``DEFAULT_ACTION``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class CapabilityParamsBag(ParamBag):
    """``action`` — the lowered capability action, ``None`` when omitted;
    ``extra`` — every remaining non-framework param, passed through to the
    capability handler untyped (the handler's schema lives in the
    capabilities layer)."""

    action: str | None
    extra: dict[str, object]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        raw = params.get(Keys.action)
        return cls(
            action=None if raw is None else str(raw).lower(),
            extra={
                k: v
                for k, v in params.items()
                if not k.startswith("_") and k != Keys.action
            },
        )
