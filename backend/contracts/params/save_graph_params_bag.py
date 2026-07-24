"""SaveGraphParamsBag — the typed input contract of the ``save_graph`` ability.

Owns the bounds of its own fields: :data:`ALLOWED_KINDS` (the storable subset
of the data-graph kinds) lives here and the ability imports it for its summary
and schema, so the validator and the model-facing enum can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.constants.data_graph import VALID_KINDS
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError

# Subset: exclude system (internal-write only). VALID_KINDS already omits
# document (ingest-only) and behavioral_pattern (save_pattern's tool).
ALLOWED_KINDS = sorted(VALID_KINDS - {"system"})

# One-line invalid-kind recovery hint carrying a minimal, fully-valid example so
# a weak model can self-correct without re-reading the schema.
INVALID_KIND_HINT = "pick one of the allowed kinds, e.g. kind=user_specific key=residence value=Lisbon"


@dataclass(frozen=True, slots=True)
class SaveGraphParamsBag(ParamBag):
    """``kind`` — one of :data:`ALLOWED_KINDS`; ``key`` / ``value_`` — the
    stripped, non-blank fact key and value."""

    kind: str
    key: str
    value_: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        kind = str(params.get(Keys.kind, ""))
        if kind not in ALLOWED_KINDS:
            raise ToolParamError(
                f"Unknown kind {kind!r}; not a storable fact kind.",
                code="invalid-param",
                valid=tuple(ALLOWED_KINDS),
                hint=INVALID_KIND_HINT,
            )

        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only key/value slips past it and must be rejected here.
        # One combined error names every missing field at once — the model
        # self-corrects in one step instead of discovering them serially.
        key_raw = params.get(Keys.key, "")
        value_raw = params.get(Keys.value_, "")
        key = key_raw.strip() if isinstance(key_raw, str) else ""
        value = value_raw.strip() if isinstance(value_raw, str) else ""
        if not key or not value:
            missing = ", ".join(
                name for name, val in (("key", key), ("value", value)) if not val
            )
            raise ToolParamError(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("kind", "key", "value"),
            )

        return cls(kind=kind, key=key, value_=value)
