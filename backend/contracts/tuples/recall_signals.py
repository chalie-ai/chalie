"""The per-lane recall signals for one data-graph candidate — a query-scoped
scoring side-car produced by ``DataGraphRow.gather_candidates`` and fused into a
composite score by ``services/memory_recall_service.py``. Not persisted: every
field is computed at recall time against the query embedding / FTS, so it rides
alongside the (persistent) row rather than living on it."""

from __future__ import annotations

from typing import NamedTuple


class RecallSignals(NamedTuple):
    """Raw per-lane recall signals for one candidate row: the key/value-vector
    cosines and the FTS bonus. All default to 0.0 so a lane that produced no
    signal (or a supersession predecessor with none) is simply zero across the
    board."""

    key_cos: float = 0.0
    value_cos: float = 0.0
    fts_bonus: float = 0.0
