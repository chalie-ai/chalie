"""Provider — one ``providers`` row: a configured LLM backend (platform,
model, host, sealed API key) selectable as the main/vision/delegate driver.

Active-record row-model (Rule 5 / §4.1). The primary key is the plain
autoincrement ``id``, so every read-by-id and targeted write goes through the
base verbs (:meth:`~models.model.Model.get`, ``filter("id", ...).update(...)``
/ ``.delete()``) straight off this class — no bespoke id-centric CRUD needed,
unlike :class:`~models.setting.Setting` (keyed by ``key``) or
:class:`~models.thread_gist.ThreadGist` (composite key).

Two bespoke ``@classmethod``\\ s carry raw SQL the builder genuinely cannot
express (Critical 3 carve-outs, both computed/SQL-function, not predicates):
:meth:`list_summaries` projects a computed ``has_api_key`` boolean the
builder's column-only ``select`` can't compute, and :meth:`touch` bumps
``updated_at`` via the SQL keyword ``CURRENT_TIMESTAMP`` rather than a
Python-computed timestamp (never substitute a Python value for a
SQL-computed one — the stored format must not change).

Holds no ``mp``, imports no service (Rule-3 depth: pure CRUD). API-key
sealing/unsealing (vault) and role-resolution stay in
:class:`~services.provider_db_service.ProviderDbService` — this model never
touches the vault."""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class Provider(Model):
    """One ``providers`` row, keyed by the plain autoincrement ``id``."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "name",
        "platform",
        "model",
        "host",
        "api_key",
        "dimensions",
        "timeout",
        "supports_vision",
        "created_at",
        "updated_at",
    )

    name: str
    platform: str
    model: str
    host: str | None
    api_key: str | None
    dimensions: int | None
    timeout: int
    supports_vision: int
    created_at: str
    updated_at: str

    @classmethod
    def get_table(cls) -> str:
        return "providers"

    @classmethod
    def list_summaries(cls) -> list[dict[str, object]]:
        """The admin provider list, ordered by name, with a computed
        ``has_api_key`` boolean standing in for the raw (sealed) key so key
        material never leaves this layer.

        Raw SQL: the query builder's ``select`` only projects bare columns —
        it cannot compute ``(api_key IS NOT NULL)`` in the SELECT list."""
        cursor = cls._bound_connection().execute(
            "SELECT id, name, platform, model, host, "
            "(api_key IS NOT NULL) AS has_api_key, "
            "dimensions, timeout, supports_vision "
            "FROM providers ORDER BY name"
        )
        return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def touch(cls, provider_id: int) -> None:
        """Bump ``updated_at`` to the current time on one row.

        Raw SQL: ``CURRENT_TIMESTAMP`` is a SQL-computed value — the builder's
        ``update`` binds every value as a ``?`` parameter, which would store a
        Python-formatted timestamp instead of SQLite's own, changing the
        stored format."""
        cls._bound_connection().execute(
            "UPDATE providers SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (provider_id,),
        )
