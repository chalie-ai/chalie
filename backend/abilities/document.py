"""
DocumentAbility — Search and manage documents via the ACT loop.

Actions: search, list, view, delete, restore, create, upload
"""

import base64 as _base64
import hashlib as _hashlib
import json as _json
import logging
import os as _os
import secrets as _secrets
from typing import Optional

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class DocumentAbility(Ability):
    NAME = "document"
    SEARCH_TOOLTIP = "document and notes manager"
    POLICY_CATEGORY = "Documents"
    POLICY_LABELS = {
        "create": "Create document",
        "delete": "Delete document",
        "list": "List documents",
        "restore": "Restore document",
        "search": "Search documents",
        "upload": "Upload document",
        "view": "View document",
    }
    SUMMARY = "Search, view, create, and manage persistent documents and notes in the document library."
    EXAMPLES = [
        "search my documents for information about coral bleaching",
        "what documents do I have uploaded",
        "show me the climate report",
        "create a note called project-ideas.md with my brainstorm",
        "save this text as a document",
        "delete the old invoice document",
        "what did that report say about carbon emissions",
        "list all my documents",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list", "view", "delete", "restore", "create", "upload"],
                "description": (
                    "Document operation to perform; use `create` when the user asks you to "
                    "create, save, write, or store a note or document; use `upload` to "
                    "store a binary or base64-encoded file attachment."
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
                    "Required for `create` and `upload`; use a filename like 'research-notes.md' "
                    "if the user didn't give one. Optional fuzzy match for view/delete/restore."
                ),
            },
            "content": {
                "type": "string",
                "description": "The full text body to write. Required for `create`. Base64-encoded file bytes for `upload`.",
            },
            "content_type": {
                "type": "string",
                "description": "MIME type of uploaded file (e.g. 'application/pdf', 'image/png'). Required for upload.",
            },
            "source_type": {
                "type": "string",
                "description": "Origin of the document (default 'conversation').",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 120

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "search")

        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = DocumentService(db)
            body = _dispatch(service, action, params)
        except Exception as e:
            logger.exception(f"[DOCUMENT SKILL] Error: {e}")
            body = str(e)
            return {"text": _skill_tag("document", action=action, error=body[:200])}

        return {"text": _skill_tag("document", body, action=action)}


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
        return _handle_upload(service, params)
    else:
        valid = "search, list, view, delete, restore, create, upload"
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

    if doc.get("status") != "ready":
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


def _split_paragraph_into_sentences(para: str, max_chars: int) -> list:
    """Greedily split a paragraph at sentence boundaries; fall back to fixed-width slicing."""
    sentences = []
    remaining = para
    while remaining:
        for sep in (". ", "! ", "? "):
            idx = remaining.find(sep)
            if idx != -1 and idx + len(sep) <= max_chars:
                sentences.append(remaining[: idx + 1])
                remaining = remaining[idx + len(sep):]
                break
        else:
            sentences.append(remaining[:max_chars])
            remaining = remaining[max_chars:]
    return [s for s in sentences if s]


def _absorb_long_para(buffer: str, para: str, chunks: list, min_chars: int, max_chars: int) -> str:
    """Flush buffer, split the long paragraph, and accumulate sentences back into buffer."""
    if buffer:
        chunks.append(buffer)
    new_buf = ""
    for sentence in _split_paragraph_into_sentences(para, max_chars):
        if len(new_buf) + len(sentence) >= min_chars:
            chunks.append(new_buf + sentence)
            new_buf = ""
        else:
            new_buf += sentence + " "
    return new_buf


def _build_chunks(paragraphs: list, min_chars: int, max_chars: int) -> list:
    """Pack paragraphs into chunks ≥ ``min_chars``; oversize paragraphs are sentence-split."""
    chunks: list = []
    buffer = ""
    for para in paragraphs:
        if len(para) > max_chars:
            buffer = _absorb_long_para(buffer, para, chunks, min_chars, max_chars)
            continue
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _split_into_artifacts(text: str, min_chars: int = 512, max_chars: int = 1024, overlap: int = 48) -> list:
    if not text:
        return []
    if len(text) <= min_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = _build_chunks(paragraphs, min_chars, max_chars)
    if not chunks:
        return [text]

    result = [chunks[0]]
    for chunk in chunks[1:]:
        result.append(result[-1][-overlap:] + chunk)
    return result


def create_document_artifacts(doc_id: str, text_content: str) -> int:
    """Split text into artifacts and store in data_graph. Returns artifact count."""
    artifacts = _split_into_artifacts(text_content)

    from services.data_graph_service import get_data_graph_service, KIND_DOCUMENT
    dgs = get_data_graph_service()

    for i, artifact_text in enumerate(artifacts):
        dgs.store(
            kind=KIND_DOCUMENT,
            key=f"doc:{doc_id}:{i:03d}",
            value=artifact_text,
            source=f"document:{doc_id}",
        )

    return len(artifacts)


def _handle_upload(service, params: dict) -> str:
    name = params.get("name", "attachment")
    content = params.get("content", "")
    content_type = params.get("content_type", "application/octet-stream")

    if not name:
        return "[DOCUMENT] 'name' is required for upload."
    if not content:
        return "[DOCUMENT] 'content' (base64 data) is required for upload."

    try:
        file_bytes = _base64.b64decode(content)
    except Exception as e:
        return f"[DOCUMENT] Failed to decode base64 content: {e}"

    file_hash = _hashlib.sha256(file_bytes).hexdigest()
    doc_id = _secrets.token_hex(4)
    file_path_rel = f"{doc_id}/{name}"

    try:
        from api.documents import _run_upload_extraction
        from services.file_mapper_service import FileMapperService

        dir_path = FileMapperService.get_documents_path(doc_id)
        _os.makedirs(dir_path, exist_ok=True)
        full_path = str(FileMapperService.get_documents_path(doc_id, name))
        with open(full_path, 'wb') as fh:
            fh.write(file_bytes)

        service.create_document(
            original_name=name,
            mime_type=content_type,
            file_size=len(file_bytes),
            file_path=file_path_rel,
            file_hash=file_hash,
            source_type='upload',
            doc_id=doc_id,
        )

        _run_upload_extraction(doc_id)

        return (
            f"[DOCUMENT] Uploaded '{name}' (id={doc_id}). "
            f"Call document(action='view', id='{doc_id}') to read contents."
        )
    except Exception as e:
        logger.exception(f"[DOCUMENT SKILL] Upload failed: {e}")
        return f"[DOCUMENT] Failed to upload document: {e}"


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
            source_type=params.get("source_type", "conversation"),
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
