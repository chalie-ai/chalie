"""
DocumentAbility — Search and manage documents via the ACT loop.

LLM-callable actions: search, list, view, delete, restore, create.

``upload`` is NOT in the LLM ``INPUT_SCHEMA``: it is the single MECHANICAL ingest
for every uploaded file (chat attachments and the Documents library), invoked only
by code via ``ToolDispatcher.dispatch``. It takes a file PATH, never bytes — a
path is tiny, so the act-trail (which records every dispatch's params verbatim)
can never carry a multi-megabyte base64 blob and blow the context window. The
``_dispatch`` ``upload`` branch is retained for that mechanical route.
"""

import hashlib as _hashlib
import json as _json
import logging
import mimetypes as _mimetypes
import os as _os
import secrets as _secrets
import shutil as _shutil
from typing import ClassVar, Optional

from abilities._ability import Ability
from abilities._result import ToolResult
from services.document_chunking import create_document_artifacts

logger = logging.getLogger(__name__)


class DocumentAbility(Ability):
    def get_name(self) -> str:
        return "document"

    def get_summary(self) -> str:
        return "Search, view, create, and manage persistent documents and notes in the document library."

    def get_examples(self) -> list[str]:
        return [
            "search my documents for information about coral bleaching",
            "what documents do I have uploaded",
            "show me the climate report",
            "create a note called project-ideas.md with my brainstorm",
            "save this text as a document",
            "delete the old invoice document",
            "what did that report say about carbon emissions",
            "list all my documents",
        ]

    def get_search_tooltip(self) -> str:
        return "document and notes manager"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list", "view", "delete", "restore", "create"],
                "description": (
                    "Document operation to perform; use `create` when the user asks you to "
                    "create, save, write, or store a note or document."
                ),
            },
            "query": {
                "type": "string",
                "description": "Text to search across all documents. Required for search.",
            },
            "id": {
                "type": "string",
                "description": "Document ID for exact match (view, delete, restore).",
            },
            "name": {
                "type": "string",
                "description": (
                    "Required for `create`; use a filename like 'research-notes.md' "
                    "if the user didn't give one. Optional fuzzy match for view/delete/restore."
                ),
            },
            "content": {
                "type": "string",
                "description": "The full text body to write. Required for `create`.",
            },
        },
        "required": ["action"],
    }

    def run(self, params: dict) -> ToolResult:
        action = params.get("action", "search")

        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = DocumentService(db)
            body = _dispatch(service, action, params)
        except Exception as e:
            logger.exception(f"[DOCUMENT SKILL] Error: {e}")
            return ToolResult.err(str(e)[:200], code="error", action=action)

        return ToolResult.ok(body, action=action)


def _parse_extracted_metadata(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _dispatch(service, action: str, params: dict) -> str:
    if action == "search":
        return _handle_search(service, params)
    elif action == "list":
        return _handle_list(service)
    elif action == "view":
        return _handle_view(service, params)
    elif action == "delete":
        return _handle_delete(service, params)
    elif action == "restore":
        return _handle_restore(service, params)
    elif action == "create":
        return _handle_create(service, params)
    elif action == "upload":
        # Mechanical-only ingest (not in INPUT_SCHEMA): dispatched by code, never
        # the LLM. Path-based — see _handle_upload / module docstring.
        return _handle_upload(service, params)
    else:
        valid = "search, list, view, delete, restore, create"
        return f"[DOCUMENT] Unknown action '{action}'. Use: {valid}"


def _resolve_document(service, params: dict) -> Optional[dict]:
    doc_id = params.get("id", "").strip()
    name = params.get("name", "").strip()

    if doc_id:
        return service.get_document(doc_id)

    if name:
        docs = service.search_documents_metadata(name)
        if docs:
            return docs[0]

    return None


def _group_results_by_doc(results: list) -> dict:
    """Bucket data_graph search rows by their owning document id."""
    doc_artifacts: dict = {}
    for row in results:
        source = row.get("source", "") or ""
        if source.startswith("document:"):
            doc_id = source.split(":", 1)[1]
        else:
            parts = (row.get("key", "") or "").split(":")
            doc_id = parts[1] if len(parts) >= 3 else "unknown"
        doc_artifacts.setdefault(doc_id, []).append(row)
    return doc_artifacts


def _handle_search(service, params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return "[DOCUMENT] 'query' is required for search."

    try:
        from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT

        dgs = get_data_graph_service()
        results = dgs.recall(query, kinds=[KIND_DOCUMENT], limit=10)

        if not results:
            return f"[DOCUMENT] No documents match '{query}'."

        doc_artifacts = _group_results_by_doc(results)

        lines = []
        for doc_id, artifacts in doc_artifacts.items():
            doc = service.get_document(doc_id)
            if not doc or doc.get("deleted_at"):
                continue
            doc_name = doc.get("original_name", doc_id)
            chunk_count = doc.get("chunk_count", 0)
            lines.append(
                f"  · id={doc_id}: \"{doc_name}\" ({chunk_count} chunks, {len(artifacts)} match(es))"
            )

        if not lines:
            return f"[DOCUMENT] No documents match '{query}'."

        lines.insert(0, f"[DOCUMENT] Found {len(lines)} document(s) matching '{query}':")
        lines.append("\nUse action \"view\" with the document id to read its full content.")
        return "\n".join(lines)

    except Exception as e:
        logger.exception(f"[DOCUMENT SKILL] Search failed: {e}")
        return f"[DOCUMENT] Search failed: {e}"


def _handle_list(service) -> str:
    docs = service.get_all_documents()
    docs = [d for d in docs if d.get("status") == "ready"]

    if not docs:
        return "[DOCUMENT] No documents in library."

    lines = ["[DOCUMENT] Document library:"]
    for doc in docs:
        _meta = _parse_extracted_metadata(doc.get("extracted_metadata"))
        doc_type = _meta.get("document_type", {})
        if isinstance(doc_type, dict):
            doc_type = doc_type.get("value", "")
        elif not isinstance(doc_type, str):
            doc_type = ""
        type_str = f" [{doc_type}]" if doc_type and doc_type != "document" else ""
        pages = doc.get("page_count")
        page_str = f", {pages}p" if pages else ""
        status = doc.get("status", "unknown")
        created = doc.get("created_at")
        date_str = ""
        if created:
            from services.time_formatter_service import TimeFormatterService
            local = TimeFormatterService.local(created, fmt="%b %d")
            if local:
                date_str = f", uploaded {local}"

        lines.append(
            f"  · {doc['original_name']}{type_str}"
            f" ({status}{page_str}, {doc.get('chunk_count', 0)} chunks{date_str})"
        )

    return "\n".join(lines)


def _append_meta_summary(lines: list, doc: dict, meta: dict) -> None:
    """Append type/pages/companies/dates/values/refs lines to ``lines`` based on parsed meta."""
    doc_type = meta.get("document_type", {})
    if isinstance(doc_type, dict):
        doc_type = doc_type.get("value", "")
    elif not isinstance(doc_type, str):
        doc_type = ""
    if doc_type:
        lines.append(f"  Type: {doc_type}")
    if doc.get("page_count"):
        lines.append(f"  Pages: {doc['page_count']}")
    if meta.get("companies"):
        lines.append("  Companies: " + ", ".join(c["name"] for c in meta["companies"][:5]))
    if meta.get("dates"):
        lines.append("  Dates: " + ", ".join(d["value"] for d in meta["dates"][:5]))
    if meta.get("expiration_dates"):
        lines.append("  Expiration dates: " + ", ".join(d["value"] for d in meta["expiration_dates"][:3]))
    if meta.get("monetary_values"):
        lines.append("  Monetary values: " + ", ".join(
            f"{v['currency']} {v['amount']}" for v in meta["monetary_values"][:5]
        ))
    if meta.get("reference_numbers"):
        lines.append("  References: " + ", ".join(r["value"] for r in meta["reference_numbers"][:5]))


def _fetch_doc_fragments(doc_id: str) -> list:
    """Pull data_graph artifact fragments for a document, ordered by key."""
    from services.data_graph_service import get_data_graph_service
    dgs = get_data_graph_service()
    try:
        with dgs.db.connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM data_graph WHERE source=? AND active=1 ORDER BY key",
                (f'document:{doc_id}',),
            )
            return [row[0] for row in cursor.fetchall() if row[0]]
    except Exception as exc:
        logger.warning("[DOCUMENT SKILL] Fragment query failed: %s", exc, exc_info=True)
        return []


def _handle_view(service, params: dict) -> str:
    doc = _resolve_document(service, params)
    if not doc:
        return "[DOCUMENT] Document not found. Specify 'name' or 'id'."

    status = doc.get("status")
    if status != "ready":
        if status == "failed":
            return f"Failed to process document {doc['original_name']}"
        return f"[DOCUMENT] '{doc['original_name']}' is still being processed or awaiting confirmation."

    meta = _parse_extracted_metadata(doc.get("extracted_metadata"))
    lines = [f"[DOCUMENT] {doc['original_name']}:"]
    _append_meta_summary(lines, doc, meta)

    clean_text = doc.get("clean_text", "")
    if clean_text:
        lines.append(f"\n--- Full Document Text ---\n{clean_text}")
        return "\n".join(lines)

    fragments = _fetch_doc_fragments(doc["id"])
    if fragments:
        lines.append("\n--- Full Document Text ---\n" + "\n\n".join(fragments))
    else:
        lines.append("\n  (No text content available)")

    return "\n".join(lines)


def _handle_delete(service, params: dict) -> str:
    doc = _resolve_document(service, params)
    if not doc:
        return "[DOCUMENT] Document not found. Specify 'name' or 'id'."

    doc_id = doc["id"]
    success = service.soft_delete(doc_id)
    if not success:
        return f"[DOCUMENT] Failed to delete '{doc['original_name']}'."

    from services.data_graph_service import get_data_graph_service
    dgs = get_data_graph_service()
    deleted_count = dgs.hard_delete_by_source_prefix(f"document:{doc_id}")

    return f"[DOCUMENT] Deleted '{doc['original_name']}'. {deleted_count} artifact(s) removed."


def _handle_restore(service, params: dict) -> str:
    doc_id = params.get("id", "").strip()
    name = params.get("name", "").strip()

    if doc_id:
        doc = service.get_document(doc_id)
    elif name:
        all_docs = service.get_all_documents(include_deleted=True)
        doc = next(
            (d for d in all_docs if name.lower() in d["original_name"].lower() and d.get("deleted_at")),
            None,
        )
    else:
        return "[DOCUMENT] Specify 'name' or 'id' to restore."

    if not doc:
        return "[DOCUMENT] Document not found."

    if not doc.get("deleted_at"):
        return f"[DOCUMENT] '{doc['original_name']}' is not deleted."

    success = service.restore(doc["id"])
    if success:
        return f"[DOCUMENT] Restored '{doc['original_name']}'."
    return f"[DOCUMENT] Failed to restore '{doc['original_name']}'."


def ingest_file(
    service,
    src_path: str,
    *,
    name: "str | None" = None,
    subdir: "str | None" = None,
    source_type: str = "upload",
) -> dict:
    """The single mechanical ingest for an uploaded file given by PATH.

    Shared by both upload surfaces — chat attachments (via ``_handle_upload`` /
    ``ToolDispatcher``) and the Documents library (``api.documents.upload_document``).
    Copies the source file into the document store, creates the DB row, then runs
    extraction SYNCHRONOUSLY (images route to vision/OCR, text to the extractor —
    all inside ``_run_upload_extraction`` → ``extract_text``) so the document
    reaches a terminal state (ready/failed) before returning.

    Takes a path, never bytes, so the act-trail (which records every dispatch's
    params verbatim) can never carry a file blob and blow the context window.

    ``subdir`` nests the document under a named folder inside the documents
    root (``<subdir>/<doc_id>/<name>``) so machine-generated files (browser
    screenshots) never clutter the flat upload listing; ``source_type`` is
    stored verbatim on the row.

    Returns ``{"id", "hash", "name", "size", "status"}`` on success, or
    ``{"error": <message>}`` on failure.
    """
    src_path = (src_path or "").strip()
    if not src_path:
        return {"error": "'path' is required for upload."}
    if not _os.path.isfile(src_path):
        return {"error": f"No file at path: {src_path}"}

    # Sanitise the stored filename — ingest serves both chat and library files,
    # so never trust an arbitrary basename into the store path (traversal guard).
    from services.filename_utils import safe_filename
    name = safe_filename(name or _os.path.basename(src_path)) or "unnamed_document"
    if subdir is not None and subdir != safe_filename(subdir):
        return {"error": f"Invalid subdir: {subdir!r}"}
    content_type = _mimetypes.guess_type(name)[0] or "application/octet-stream"

    try:
        file_size = _os.path.getsize(src_path)
        hasher = _hashlib.sha256()
        with open(src_path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(block)
        file_hash = hasher.hexdigest()
    except OSError as e:
        return {"error": f"Failed to read file {src_path}: {e}"}

    doc_id = _secrets.token_hex(4)
    rel_parts = ([subdir] if subdir else []) + [doc_id, name]
    file_path_rel = "/".join(rel_parts)

    try:
        from services.file_mapper_service import FileMapperService

        dir_path = FileMapperService.get_documents_path(*rel_parts[:-1])
        _os.makedirs(dir_path, exist_ok=True)
        full_path = str(FileMapperService.get_documents_path(*rel_parts))
        if not FileMapperService.validate_document_path(full_path):
            return {"error": f"Invalid file path for '{name}'."}
        _shutil.copyfile(src_path, full_path)

        service.create_document(
            original_name=name,
            mime_type=content_type,
            file_size=file_size,
            file_path=file_path_rel,
            file_hash=file_hash,
            source_type=source_type,
            doc_id=doc_id,
        )
    except Exception as e:
        logger.exception(f"[DOCUMENT SKILL] Upload failed: {e}")
        return {"error": f"Failed to upload document: {e}"}

    # Extraction runs synchronously: callers MUST be able to read the document
    # immediately (the ACT loop's follow-up view in the same turn, or the library
    # endpoint's response) — never "still being processed".
    from api.documents import _run_upload_extraction, _mark_upload_failed
    try:
        _run_upload_extraction(doc_id)
    except Exception as exc:
        logger.exception(f"[DOCUMENT SKILL] Extraction failed for {doc_id}: {exc}")
        _mark_upload_failed(doc_id, str(exc))

    doc = service.get_document(doc_id)
    return {
        "id": doc_id,
        "hash": file_hash,
        "name": name,
        "size": file_size,
        "status": doc.get("status") if doc else None,
    }


def _handle_upload(service, params: dict) -> str:
    """Mechanical upload dispatch (not LLM-callable) → string for the act-trail.

    Thin formatter over ``ingest_file``. Preserves the ``(id=<doc_id>, hash=...)``
    token that the turn-0 seed parses to build the transcript-doc link.
    """
    result = ingest_file(
        service,
        params.get("path"),
        name=params.get("name"),
    )
    if result.get("error"):
        return f"[DOCUMENT] {result['error']}"
    if result.get("status") == "ready":
        return (
            f"[DOCUMENT] Uploaded '{result['name']}' "
            f"(id={result['id']}, hash={result['hash']}). "
            f"Call document(action='view', id='{result['id']}') to read contents."
        )
    return f"Failed to process document {result['name']}"


def _handle_create(service, params: dict) -> str:
    name = params.get("name", "").strip()
    content = params.get("content", "").strip()

    if not name:
        return "[DOCUMENT] 'name' is required for create."
    if not content:
        return "[DOCUMENT] 'content' is required for create."

    if "." not in name:
        name = f"{name}.md"

    try:
        doc_id = service.create_document_from_text(
            original_name=name,
            text_content=content,
            source_type='conversation',
        )

        artifact_count = create_document_artifacts(doc_id, content)
        service.update_status(doc_id, "ready", chunk_count=artifact_count)

        summary = content[:500]
        dot_pos = summary.rfind(". ")
        if dot_pos > 200:
            summary = summary[:dot_pos + 1]
        service.update_summary(doc_id, summary)

        return (
            f"[DOCUMENT] Created '{name}' (id={doc_id}). "
            f"{artifact_count} artifact(s) indexed."
        )

    except Exception as e:
        logger.exception(f"[DOCUMENT SKILL] Create failed: {e}")
        return f"[DOCUMENT] Failed to create document: {e}"
