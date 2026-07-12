# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Character-level key sanitisation — the generic layer of dispatch-seam key healing.

The first of the two cooperating layers that make every tool resilient to the
argument keys a weak model emits (the second, registry-driven layer is
:class:`~services.key_healer.KeyHealer`). Lower-casing and stripping every
character outside ``[a-z0-9_-]`` heals the single commonest model defect — a stray
escaped quote or a capital, e.g. ``source"`` → ``source`` and ``MAX_CHARS`` →
``max_chars`` — with zero per-tool knowledge.
"""

from __future__ import annotations

import re


class KeyNormalizer:
    """Reduce a raw argument key to the canonical form keys are matched on.

    One responsibility: the character-level sanitisation that gives every tool
    junk/case/separator resilience for free, independent of any registry.
    Stateless; injected into :class:`~services.key_healer.KeyHealer` so the healer
    never hard-codes its own matching rule and either layer can be swapped or
    tested in isolation.
    """

    _DROP = re.compile(r"[^a-z0-9_-]")

    def normalize(self, key: str) -> str:
        """Lower-case *key* and drop every character outside ``[a-z0-9_-]``.

        The generic sanitisation layer: ``source"`` → ``source``, ``"URL"`` →
        ``url``, ``max chars`` → ``maxchars`` (space dropped). Separators ``-`` /
        ``_`` are preserved so ``max_chars`` keeps its shape for the exact match;
        :meth:`squeeze` removes them for the loose match.
        """
        return self._DROP.sub("", str(key).lower())

    def squeeze(self, key: str) -> str:
        """:meth:`normalize` then drop ``-`` / ``_`` — the form keys are matched on.

        Collapses every spelling of one key to a single token, so ``MAX_CHARS``,
        ``max-chars`` and ``maxchars`` all compare equal (``maxchars``). Used for
        both the exact (declared-param) match and the variant match.
        """
        return self.normalize(key).replace("-", "").replace("_", "")
