# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dispatch-seam key healing — make every tool resilient to weak-model argument keys.

Two cooperating layers, applied to every tool call at the dispatch seam BEFORE
the ACTION_REQUIRED pre-gate, the policy gate, or ``run()`` ever sees the params:

1. **Key sanitisation (generic, all tools).** An incoming argument key is
   lower-cased and stripped of every character outside ``[a-z0-9_-]``. This alone
   heals the single commonest model defect — a stray escaped quote or a capital,
   e.g. ``source"`` → ``source`` and ``MAX_CHARS`` → ``max_chars`` — with zero
   per-tool knowledge. (This heals a common model defect: a model stores
   ``{"source\"": "https://…"}`` and ``read`` bounces on
   ``missing-params`` because the corrupt KEY never matched.) This layer is
   :class:`~services.key_normalizer.KeyNormalizer`.

2. **Variant resolution (registry-driven, per-tool).** A sanitised key that is
   not one of the tool's declared parameters is matched against the variant
   ladders in :data:`~configs.enums.param_key.VARIANTS`; when it belongs to
   exactly one of *that tool's own* declared parameters it is rewritten to that
   parameter's canonical key. This is what lets ``read({"url": …})`` resolve to
   ``source`` while ``url`` stays canonical for ``web_download`` — resolution is
   scoped to the tool's declared keys, never a global rename. This layer is
   :class:`KeyHealer`.

Resolution is two-pass and **declared-canonical-first**, so a parameter that is
itself a synonym of another (calendar's ``summary`` vs ``title``; ``document``'s
``id`` vs the ``list`` tool's ``id`` alias) always wins as itself and is never
rewritten away. An unrecognised key passes through **verbatim** (never the lossy
lower-cased form), so MCP camelCase keys (``serverName``) and any out-of-band key
survive intact for the tool or framework to handle.

The canonical parameter names and their variant ladders both live in
:mod:`configs.enums.param_key` (:class:`~configs.enums.param_key.Keys` and
:data:`~configs.enums.param_key.VARIANTS`); this module holds only the behaviour
that consumes them.
"""

from __future__ import annotations

import logging
from typing import cast

from configs.enums.param_key import VARIANTS
from services.key_normalizer import KeyNormalizer

logger = logging.getLogger(__name__)


class KeyHealer:
    """Heal a tool call's argument KEYS against the tool's declared schema.

    One responsibility: turn the keys a weak model emitted into the tool's
    canonical parameter keys, so a stray quote, a capital, a separator, or a known
    synonym never bounces an otherwise-valid call.

    Dependencies are injected (DIP): the
    :data:`~configs.enums.param_key.VARIANTS` registry and the
    :class:`~services.key_normalizer.KeyNormalizer` both default to the shared
    module values but can be supplied — for a test with a probe registry, or an
    alternative normalisation policy — without touching this class.
    """

    def __init__(
        self,
        variants: "dict[str, frozenset[str]]" = VARIANTS,
        normalizer: "KeyNormalizer | None" = None,
    ) -> None:
        self._variants = variants
        self._normalizer = normalizer or KeyNormalizer()

    def heal(self, params: dict[str, object], schema: dict[str, object]) -> dict[str, object]:
        """Return *params* with its keys healed against a tool's declared *schema*.

        For each incoming key, in order:
          1. **exact** — if its squeezed form matches a declared parameter, emit
             that parameter's canonical key (heals quotes/case/separators, e.g.
             ``source"`` → ``source``, ``MAX_CHARS`` → ``max_chars``).
          2. **variant** — else if its squeezed form is a variant of exactly one
             declared parameter (via :data:`~configs.enums.param_key.VARIANTS`),
             emit that parameter's key (e.g. ``url`` → ``source`` for ``read``).
          3. **passthrough** — else emit the ORIGINAL key verbatim, so MCP
             camelCase and any unknown key survive untouched.

        First write wins if two incoming keys resolve to the same canonical,
        mirroring the historical alias-ladder's "first key present wins". *schema*
        is the bare ``get_parameters()`` body (no framework fields); an
        empty/parameterless schema or empty params is returned unchanged.
        """
        properties = cast("dict[str, object]", schema.get("properties")) if schema else None
        if not properties or not params:
            return dict(params)

        squeeze = self._normalizer.squeeze
        declared = list(properties.keys())
        by_squeeze = {squeeze(k): k for k in declared}     # squeezed form → canonical key
        variant_map = self._variant_map(declared, by_squeeze)  # squeezed variant → canonical key

        healed: dict[str, object] = {}
        source_key: dict[str, str] = {}                    # canonical → the raw key that filled it
        for key, value in params.items():
            sq = squeeze(key)
            canonical = by_squeeze.get(sq) or variant_map.get(sq) or key
            if canonical not in healed:
                healed[canonical] = value
                source_key[canonical] = key
            elif key != source_key[canonical]:
                # Two DISTINCT incoming keys resolved to one canonical (e.g. a model
                # sent both ``source`` and its ``url`` alias). First write wins —
                # mirroring the historical alias ladder — but the dropped value is
                # never silent: a weak model double-emitting a key is exactly the
                # confusion worth surfacing.
                logger.warning(
                    "[KeyHealer] %r and %r both map to %r — keeping %r, dropping %r",
                    source_key[canonical], key, canonical, source_key[canonical], key,
                )
        return healed

    def _variant_map(
        self, declared: "list[str]", by_squeeze: "dict[str, str]"
    ) -> "dict[str, str]":
        """Build ``squeezed variant → canonical key`` for one tool's declared params.

        A variant that squeezes onto a declared parameter is dropped (the real
        parameter always wins as itself), and a variant claimed by two declared
        parameters is dropped as ambiguous. ``RegistryInvariant`` forbids both for
        the real registry, so this is defensive: a future bad edit degrades to "no
        aliasing for that key", never silent mis-routing.
        """
        squeeze = self._normalizer.squeeze
        out: "dict[str, str | None]" = {}
        for canonical in declared:
            for variant in self._variants.get(canonical, ()):
                sq = squeeze(variant)
                if sq in by_squeeze:
                    continue                       # a real parameter owns this form
                if sq in out and out[sq] != canonical:
                    out[sq] = None                 # ambiguous across two params → drop
                elif sq not in out:
                    out[sq] = canonical
        return {sq: c for sq, c in out.items() if c is not None}
