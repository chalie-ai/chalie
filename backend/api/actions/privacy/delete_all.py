"""Privacy delete-all action — DELETE /api/privacy/delete-all.

Irreversible wipe of all user data, gated by the ``X-Confirm-Delete: yes``
header.  The handler truncates every user-owned table and clears user-owned
MemoryStore keys — the result is a :class:`DeleteAllResult` DTO confirming
the wipe plus any per-table truncation warnings.
"""

from __future__ import annotations

import logging
import shutil
from typing import ClassVar

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.privacy import DeleteAllResult
from exceptions import EndpointError
from models.data_graph import DataGraphRow
from models.episode import Episode
from models.llm_call_log import LlmCallLog
from models.list import List
from models.list_item import ListItem
from models.memory_recall_log import MemoryRecallLog
from models.model import Model
from models.scheduled_item import ScheduledItem
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.database import Database
from services.file_index_service import FileIndexService
from services.file_mapper_service import FileMapperService
from services.memory_client import MemoryClientService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# Modeled user-data tables for the nuclear delete, FK-children before
# parents. .truncate() clears each. System / auth / config tables excluded.
_DELETE_ALL_MODELS: tuple[type[Model], ...] = (
    ToolCall,          # tool_calls   FK -> transcript(id)
    ListItem,          # list_items   FK -> lists(id)
    Transcript,
    Episode,
    DataGraphRow,      # data_graph
    List,              # lists
    ScheduledItem,
    MemoryRecallLog,
    LlmCallLog,
)

# User-data tables with no model yet — cleared by explicit DELETE, children
# first so data_graph_edges (FK child of data_graph) goes before the model
# loop empties data_graph.
_DELETE_ALL_MODELLESS_TABLES = (
    "data_graph_edges",   # FK -> data_graph(id) ON DELETE CASCADE
    "concept_lut_misses",
)

# MemoryStore key patterns that belong to the user and must be cleared.
_DELETE_ALL_STORE_PATTERNS = (
    "working_memory:*",
    "deliberation_score:*",
)


class PrivacyDeleteAllAction(Action):
    """Action to irreversibly wipe all user data."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"delete"})
    response_dto = {"delete": DocumentedResponse(DeleteAllResult, not_found=False)}

    def slug(self) -> str:
        return "privacy"

    def verb(self) -> str:
        return "delete-all"

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Irreversibly delete ALL user data — gated by the ``X-Confirm-Delete: yes`` header."""
        from flask import request

        if request.headers.get("X-Confirm-Delete", "") != "yes":
            raise EndpointError("Requires X-Confirm-Delete: yes header")

        truncate_failures: list[str] = []

        with Database.transaction() as conn:
            # Model-less residual tables first (children before parents).
            for table in _DELETE_ALL_MODELLESS_TABLES:
                try:
                    conn.execute(f"DELETE FROM {table}")
                except Exception as e:
                    logger.warning(f"[REST API] Failed to delete from {table}: {e}")
                    truncate_failures.append(table)
            # Every modeled user-data table, FK-children before parents.
            for model in _DELETE_ALL_MODELS:
                try:
                    model.truncate()
                except Exception as e:
                    logger.warning(f"[REST API] Failed to truncate {model.get_table()}: {e}")
                    truncate_failures.append(model.get_table())
        # Clear user-owned MemoryStore keys
        try:
            store = MemoryClientService.create_connection()
            for pattern in _DELETE_ALL_STORE_PATTERNS:
                matched = store.keys(pattern)
                if matched:
                    store.delete(*matched)
        except Exception as e:
            logger.warning(f"[REST API] Failed to clear memory store: {e}")

        # Recursively delete the contents of the documents directory (directory itself stays)
        try:
            docs_path = FileMapperService.get_documents_path()
            if docs_path.exists():
                for item in docs_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception as e:
            logger.warning(f"[REST API] Failed to clear documents directory: {e}")

        # Clear the extracted-content index; the reconcile scan repopulates it.
        try:
            FileIndexService().clear()
        except Exception as e:
            logger.warning(f"[REST API] Failed to clear file index: {e}")

        return DeleteAllResult(
            deleted=True,
            timestamp=utc_now(),
            warnings=f"Failed to truncate: {', '.join(truncate_failures)}" if truncate_failures else None,
        ).single()
