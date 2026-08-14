"""The ``Searchable`` config-trait — a model's declared search-index footprint.

A :class:`SearchConfig` names the sidecar tables (and the base-table source
columns) a searchable model's rows populate: the FTS posting and the vector
lanes. It is *declared* on the model class (``DataGraphRow.__search__`` and its
overrides), never inferred: the presence of a config IS the enablement signal —
a model with ``__search__ = None`` earns no FTS/vec posting and is excluded from
recall. There are no flags or policy tables; the single authority is the
declaration.

Model-free by design: this module
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
    base-table columns whose text seeds the embedding, joined
    ``"a: b"`` (the first column is always included; each later column is
    appended only when non-empty). ``kind_column`` — when set — names a
    discriminator column the engine gates on via the kind→config registry (only
    ``data_graph``, whose one table holds many searchable and non-searchable
    kinds). ``indexed_column`` is the base-table column stamped with the
    ``datetime('now')`` timestamp the async engine writes when it posts the row
    into the FTS table (NULL = never indexed). ``heal_where`` is the self-heal
    liveness predicate (the rows still eligible for a missing-index backfill).
    """

    base_table: str
    fts_table: str
    fts_columns: tuple[str, ...]
    vec_lanes: tuple[VecLane, ...]
    text_columns: tuple[str, ...]
    kind_column: str | None = None
    indexed_column: str = "indexed_at"
    heal_where: str = "deleted_at IS NULL"


#: The footprint every searchable ``data_graph`` kind shares — the default
#: declared on :class:`~models.data_graph.DataGraphRow` and inherited by every
#: searchable vertical (facts, misc, places, discoveries, documents, contacts,
#: system memories). Non-searchable kinds override ``__search__`` to ``None``.
DATA_GRAPH_SEARCH = SearchConfig(
    base_table="data_graph",
    fts_table="data_graph_fts",
    fts_columns=("key", "value", "kind"),
    vec_lanes=(
        VecLane("data_graph_key_vec", "key"),
        VecLane("data_graph_value_vec", "value"),
    ),
    text_columns=("key", "value"),
    kind_column="kind",
    heal_where="deleted_at IS NULL AND active = 1",
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


#: Memory graph search footprint. Graph rows are keyed by ``subject``
#: (unique); FTS5 indexes the subject column so recall can do exact-key
#: text lookups over living facts. No vector lane — graph rows are
#: surfaced purely by subject-match.
GRAPH_SEARCH = SearchConfig(
    base_table="memory_graph",
    fts_table="memory_graph_fts",
    fts_columns=("subject",),
    vec_lanes=(),
    text_columns=("subject", "contents"),
    heal_where="1=1",
)

#: Memory map search footprint. Map rows carry episodic lineage; only their
#: ``cues`` — the situational tag set — are embedded. Neither the episode text
#: nor its ``source`` is indexed: recall matches the situation the user is in
#: against the situations a memory is about, not two narratives. No FTS.
#: A row with empty cues is unreachable by design.
MAP_SEARCH = SearchConfig(
    base_table="memory_map",
    fts_table="",
    fts_columns=(),
    vec_lanes=(VecLane("memory_map_cues_vec", "cues"),),
    text_columns=("cues",),
    heal_where="1=1",
)


register_table(DATA_GRAPH_SEARCH)
register_table(GRAPH_SEARCH)
register_table(MAP_SEARCH)
