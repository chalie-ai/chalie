"""Active-record row-model for the ``llm_call_log`` table (§4.1).

One persisted per-call token/latency telemetry row. ``save()`` inserts a new
row and lets ``created_at`` default in SQL. Pure CRUD — no SELECT/aggregate
reporting SQL (those live in ``services.llm_log_service``).
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class LlmCallLog(Model):
    """One ``llm_call_log`` row — pure CRUD, no analytics."""

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
        "turn_id",
        "created_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "llm_call_log"
