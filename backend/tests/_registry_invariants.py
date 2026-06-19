# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Registry-invariant checks for the parameter-key healer - TEST-ONLY surface.

Lives here so production code never imports a validator it never calls. Imports
the real registry from ``abilities._params`` and reflects over the real
``AbilityRegistry`` supplied by the caller. Re-implements NO production logic,
adds NO mocks, and creates NO alternative code path. The ``_`` prefix keeps pytest
from collecting it as a test module while ``test_param_key_resilience`` imports it.
"""

from __future__ import annotations

from abilities._params import VARIANTS, KeyNormalizer, Keys

# The two keys the framework injects into / strips from every call (see
# Ability.get_input_schema and ToolDispatcher.dispatch/_execute). No parameter and
# no variant may ever collide with these, or a model key could hijack a framework
# slot. Compared on the squeezed form (see ``KeyNormalizer.squeeze``).
FRAMEWORK_KEYS = ("act_summary", "async")


class RegistryOverlapError(Exception):
    """Raised when a tool's variant ladders overlap with another parameter or a
    framework key - the invariant that keeps key healing unambiguous."""


class RegistryInvariant:
    """Asserts structural invariants the key-healing design rests on.

    Dependencies are injected so a test can drive the checks against a probe
    registry. Stateless across calls.
    """

    def __init__(
        self,
        variants: "dict[str, frozenset[str]]" = VARIANTS,
        framework_keys: "tuple[str, ...]" = FRAMEWORK_KEYS,
        normalizer: "KeyNormalizer | None" = None,
    ) -> None:
        self._variants = variants
        self._framework_keys = framework_keys
        self._normalizer = normalizer or KeyNormalizer()

    def canonical_keys(self) -> "frozenset[str]":
        """Schema-completeness check: a registered ability's property keys must be
        a subset of these, so wire keys and the variant registry cannot drift from
        the schema.
        """
        return frozenset(
            v for k, v in vars(Keys).items()
            if not k.startswith("_") and isinstance(v, str)
        )

    def check_no_overlaps(self, abilities: "list") -> None:
        """Raises :class:`RegistryOverlapError` listing ALL violations found (not
        just the first), so one run surfaces the entire registry state.
        """
        squeeze = self._normalizer.squeeze
        framework_sq = {squeeze(k) for k in self._framework_keys}
        problems: "list[str]" = []
        all_declared: "set[str]" = set()

        for ability in abilities:
            name = ability.get_name()
            try:
                properties = (ability.get_parameters() or {}).get("properties") or {}
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: get_parameters() raised {exc!r}")
                continue
            declared = list(properties.keys())
            all_declared.update(declared)
            by_squeeze = {squeeze(k): k for k in declared}

            for param in declared:
                if squeeze(param) in framework_sq:
                    problems.append(f"{name}: parameter {param!r} collides with a framework key")

            seen: "dict[str, str]" = {}
            for canonical in declared:
                for variant in self._variants.get(canonical, ()):
                    sq = squeeze(variant)
                    if sq in framework_sq:
                        problems.append(
                            f"{name}: variant {variant!r} of {canonical!r} collides with a framework key"
                        )
                    other = by_squeeze.get(sq)
                    if other is not None and other != canonical:
                        problems.append(
                            f"{name}: variant {variant!r} of {canonical!r} collides with declared "
                            f"parameter {other!r}"
                        )
                    if sq in seen and seen[sq] != canonical:
                        problems.append(
                            f"{name}: variant {variant!r} is claimed by both {seen[sq]!r} and "
                            f"{canonical!r}"
                        )
                    seen[sq] = canonical

        unknown = [k for k in self._variants if k not in all_declared]
        if unknown:
            problems.append(
                f"VARIANTS keys are not declared parameters of any ability: {sorted(unknown)}"
            )

        if problems:
            raise RegistryOverlapError(
                "Parameter-key registry overlap(s):\n  " + "\n  ".join(problems)
            )
