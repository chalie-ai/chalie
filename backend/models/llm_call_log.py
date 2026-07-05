"""Active-record row-model for the ``llm_call_log`` table (§4.1).

One persisted per-call token/latency telemetry row. ``save()`` inserts a new
row and lets ``created_at`` default in SQL. Holds the ONLY home of the
``llm_call_log`` read SQL (I6): the last-request probe and the window-bucketed
token aggregates, ported verbatim from the dissolved ``llm_call_log_service``.
No mp, no service, no WS.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class LlmCallLog(Model):
    """One ``llm_call_log`` row plus the table's token-aggregate query scopes."""

    __table__: ClassVar[str] = "llm_call_log"
    __columns__: ClassVar[tuple[str, ...]] = (
        "id",
        "job_name",
        "provider",
        "model",
        "tokens_input",
        "tokens_output",
        "tokens_cache_read",
        "tokens_cache_create",
        "tokens_thinking",
        "latency_ms",
        "usage_class",
        "created_at",
    )

    @classmethod
    def last_request_tokens(cls, job_name: str) -> int | None:
        """``tokens_input`` of the most recent call for ``job_name``, or None.

        Keys off ``job_name`` ('channel:role'), not ``usage_class``, so the
        indicator tracks the real user turn rather than a delegate's tiny
        sub-request that shares usage_class 'chat'.
        """
        row = cls._bound_connection().execute(
            "SELECT tokens_input FROM llm_call_log WHERE job_name = ? "
            "ORDER BY id DESC LIMIT 1",
            (job_name,),
        ).fetchone()
        return int(row["tokens_input"]) if row is not None else None

    @classmethod
    def tokens_today(cls) -> int:
        """Total tokens (all classes) logged since the start of the UTC day."""
        row = cls._bound_connection().execute(
            """SELECT COALESCE(SUM(tokens_input + tokens_output +
                                  tokens_cache_read + tokens_cache_create +
                                  tokens_thinking), 0) AS total
               FROM llm_call_log
               WHERE created_at >= date('now')""",
        ).fetchone()
        return int(row["total"]) if row is not None else 0

    @classmethod
    def usage_buckets(
        cls, bucket_fmt: str, offset: str | None, usage_class: str | None
    ) -> list[dict[str, object]]:
        """Token sums grouped by (bucket, usage_class, model, provider).

        ``bucket_fmt`` is the ``strftime`` bucket width (by-hour or by-day),
        ``offset`` an optional ``datetime('now', ?)`` window bound (None =
        lifetime), ``usage_class`` an optional class filter.
        """
        where_clauses: list[str] = []
        params: list[object] = []
        if offset is not None:
            where_clauses.append("created_at >= datetime('now', ?)")
            params.append(offset)
        if usage_class:
            where_clauses.append("usage_class = ?")
            params.append(usage_class)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cursor = cls._bound_connection().execute(
            f"""SELECT
                   strftime('{bucket_fmt}', created_at) AS bucket,
                   usage_class,
                   model,
                   provider,
                   SUM(tokens_input)        AS tokens_input,
                   SUM(tokens_output)       AS tokens_output,
                   SUM(tokens_cache_read)   AS tokens_cache_read,
                   SUM(tokens_cache_create) AS tokens_cache_create,
                   SUM(tokens_thinking)     AS tokens_thinking
               FROM llm_call_log
               {where_sql}
               GROUP BY bucket, usage_class, model, provider
               ORDER BY bucket DESC""",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]
