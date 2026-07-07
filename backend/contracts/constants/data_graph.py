"""Canonical ``data_graph`` kind constants — the one home for the kind
vocabulary and the memory-tool storable set.

Each per-kind vertical model still declares its own ``KIND`` (its identity);
this module is the shared vocabulary the memory-tool layer
(``abilities/save_graph.py``, ``services/memory_retrieval.py``) and the one-off
key-canonicalisation migration import. Model-free by design: ``contracts`` is a
low-level package that never imports from ``models`` (avoids an import cycle),
so the pattern kind is the plain literal, not ``BehavioralPattern.KIND``."""

from __future__ import annotations

KIND_USER_SPECIFIC = "user_specific"
KIND_SYSTEM = "system"
KIND_MISC = "misc"
KIND_DOCUMENT = "document"
KIND_BEHAVIORAL_PATTERN = "behavioral_pattern"
KIND_PLACE = "place"
KIND_CONTACT = "contact"
KIND_DISCOVERY = "discovery"

# The kinds a user or agent may write through the memory tools (``save_graph``,
# ``memory`` store). ``document`` is fragment-based (ingest-only) and
# ``behavioral_pattern`` is extracted by the pattern worker — neither is
# writable as a free ``(key, value)`` pair, so both are EXCLUDED from the
# storable set (they remain in the vocabulary above for reference).
VALID_KINDS = frozenset({
    KIND_USER_SPECIFIC, KIND_SYSTEM, KIND_MISC,
    KIND_PLACE, KIND_CONTACT, KIND_DISCOVERY,
})
