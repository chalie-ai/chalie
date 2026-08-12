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

    ``base_table`` is the rowid-keyed content table the sidecars index and the
    write-side engine reads from; every posting is addressed by that table's
    ``rowid``. ``fts_columns`` is the FTS5 content-column order (excluding the
    implicit ``rowid``) — the order the external-content ``'delete'`` command
    must supply. ``vec_lanes`` are the embedding sidecars (empty when a model
    populates its own vectors synchronously). ``text_columns`` name the
    base-table columns whose text seeds doc2query / the embedding, joined
    ``"a: b"`` (the first column is always included; each later column is
    appended only when non-empty). ``variant_table`` / ``variant_vec_table``
    hold the doc2query expansions (``None`` when the model has no semantic
    variant lane; kept out of teardown otherwise: an AFTER-DELETE trigger on the
    base table cascades them). ``kind_column`` — when set — names a
    discriminator column the engine gates on via the kind→config registry (only
    ``data_graph``, whose one table holds many searchable and non-searchable
    kinds). ``queries_column`` is the base-table column the generated variant
    list is persisted into. ``heal_where`` is the self-heal liveness predicate
    (the rows still eligible for a missing-index backfill).
    """

    base_table: str
    fts_table: str
    fts_columns: tuple[str, ...]
    vec_lanes: tuple[VecLane, ...]
    text_columns: tuple[str, ...]
    variant_table: str | None = None
    variant_vec_table: str | None = None
    kind_column: str | None = None
    queries_column: str = "search_queries"
    heal_where: str = "deleted_at IS NULL"


#: The footprint every searchable ``data_graph`` kind shares — the default
#: declared on :class:`~models.data_graph.DataGraphRow` and inherited by every
#: searchable vertical (facts, misc, places, discoveries, documents, contacts,
#: system memories). Non-searchable kinds override ``__search__`` to ``None``.
DATA_GRAPH_SEARCH = SearchConfig(
    base_table="data_graph",
    fts_table="data_graph_fts",
    fts_columns=("key", "value", "kind", "search_queries"),
    vec_lanes=(
        VecLane("data_graph_key_vec", "key"),
        VecLane("data_graph_value_vec", "value"),
    ),
    text_columns=("key", "value"),
    variant_table="expanded_semantic",
    variant_vec_table="expanded_semantic_vec",
    kind_column="kind",
    heal_where="deleted_at IS NULL AND active = 1",
)


#: Episode search footprint. Episodes carry their vectors synchronously
#: (``EpisodicService`` writes ``episodes_vec`` on the store path), so no vec
#: lane is declared here — the async engine only owns the FTS posting and the
#: doc2query keyword variants (persisted into the ``search_queries`` FTS column;
#: an unfiltered ``episodes_fts MATCH`` spans every column, so variants widen
#: recall with no change to the retrieval query). No semantic-variant lane:
#: episodic recall never reads ``expanded_semantic``, so it would be dead.
EPISODE_SEARCH = SearchConfig(
    base_table="episodes",
    fts_table="episodes_fts",
    fts_columns=("gist", "search_queries"),
    vec_lanes=(),
    text_columns=("gist",),
    heal_where="deleted_at IS NULL",
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


def is_searchable(kind: str) -> bool:
    """Whether rows of ``kind`` earn a search-index posting — the single
    authority the write-side engine consults, mirroring the save path's
    ``self.__search__ is not None``. An unregistered kind reads as
    non-searchable (safe default: never index an unknown kind)."""
    return _REGISTRY.get(kind) is not None


# ── base-table → config registry ───────────────────────────────────────────────
#
# The write-side engine dequeues ``{table, rowid}`` items and self-heals by
# scanning base tables, so it resolves config by the base-table name (not the
# per-row kind). Populated at module import below — one entry per searchable
# base table. A table with a ``kind_column`` still gates individual rows through
# the kind registry above; a table without one indexes every live row.

_TABLE_REGISTRY: dict[str, SearchConfig] = {}


def register_table(config: SearchConfig) -> None:
    """Record a searchable base table's config, keyed by ``config.base_table``."""
    _TABLE_REGISTRY[config.base_table] = config


def config_for_table(table: str) -> SearchConfig | None:
    """The config for a searchable base table, or ``None`` when the table has no
    declared search footprint (the safe default: never index an unknown table)."""
    return _TABLE_REGISTRY.get(table)


def searchable_tables() -> tuple[str, ...]:
    """Every registered searchable base table — the self-heal scan set."""
    return tuple(_TABLE_REGISTRY)


register_table(DATA_GRAPH_SEARCH)
register_table(EPISODE_SEARCH)
