"""The ``Searchable`` config-trait — a model's declared search-index footprint.

A :class:`SearchConfig` names the sidecar tables (and the base-table source
columns) a searchable model's rows populate: the FTS posting, the key/value
vector lanes, and the doc2query-variant tables. It is *declared* on the model
class (``DataGraphRow.__search__`` and its overrides), never inferred: the
presence of a config IS the enablement signal — a model with ``__search__ =
None`` earns no FTS/vec posting and is excluded from recall. There are no flags
or policy tables; the single authority is the declaration.

Model-free by design (like ``contracts/constants/data_graph.py``): this module
imports nothing from ``models`` or ``services``, so both the model layer (which
reads ``self.__search__`` on the save path) and the write-side engine
(``services.search_expander_service``, which resolves a raw ``kind`` string to
its config) can depend on it without an import cycle. The kind→config registry
is populated by ``DataGraphRow.__init_subclass__`` as each concrete vertical is
imported.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VecLane:
    """One vector sidecar: the table holding the packed embeddings and the
    base-table column whose text the embedding is generated from."""

    table: str
    source: str


@dataclass(frozen=True)
class SearchConfig:
    """The declared search-index footprint of a searchable model.

    ``fts_columns`` is the FTS5 content-column order (excluding the implicit
    ``rowid``) — it is the order the external-content ``'delete'`` command must
    supply. ``vec_lanes`` are the key/value embedding sidecars. ``variant_table``
    / ``variant_vec_table`` hold the doc2query expansions (kept out of teardown:
    an AFTER-DELETE trigger on the base table cascades them). ``queries_column``
    is the base-table column the generated variant list is persisted into.
    """

    fts_table: str
    fts_columns: tuple[str, ...]
    vec_lanes: tuple[VecLane, ...]
    variant_table: str
    variant_vec_table: str
    queries_column: str = "search_queries"


#: The footprint every searchable ``data_graph`` kind shares — the default
#: declared on :class:`~models.data_graph.DataGraphRow` and inherited by every
#: searchable vertical (facts, misc, places, discoveries, documents, contacts,
#: system memories). Non-searchable kinds override ``__search__`` to ``None``.
DATA_GRAPH_SEARCH = SearchConfig(
    fts_table="data_graph_fts",
    fts_columns=("key", "value", "kind", "search_queries"),
    vec_lanes=(
        VecLane("data_graph_key_vec", "key"),
        VecLane("data_graph_value_vec", "value"),
    ),
    variant_table="expanded_semantic",
    variant_vec_table="expanded_semantic_vec",
)


# ── kind → config registry ────────────────────────────────────────────────────
#
# Populated by DataGraphRow.__init_subclass__ at model-import time. The save path
# reads self.__search__ directly (self IS the L3 instance); this registry serves
# the write-side engine's raw-kind resolution (self-heal / per-row processing),
# which only has a `kind` string from a SQL row and no model instance.

_REGISTRY: dict[str, SearchConfig | None] = {}


def register_kind(kind: str, config: SearchConfig | None) -> None:
    """Record a concrete kind's declared search config (may be ``None``)."""
    _REGISTRY[kind] = config


def config_for_kind(kind: str) -> SearchConfig | None:
    """The declared config for ``kind``, or ``None`` when the kind is
    non-searchable OR not yet registered (its model not imported). Callers that
    need to distinguish those two cases must ensure the model is imported."""
    return _REGISTRY.get(kind)


def is_searchable(kind: str) -> bool:
    """Whether rows of ``kind`` earn a search-index posting — the single
    authority the write-side engine consults, mirroring the save path's
    ``self.__search__ is not None``. An unregistered kind reads as
    non-searchable (safe default: never index an unknown kind)."""
    return _REGISTRY.get(kind) is not None
