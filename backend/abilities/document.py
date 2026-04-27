"""
DocumentAbility — Search and manage documents via the ACT loop.

Actions: search, list, view, delete, restore, create
"""

import json as _json
import logging
from typing import Optional

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class DocumentAbility(Ability):
    NAME = "document"
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
                    "Required for `create`; use a filename like 'research-notes.md' if the user "
                    "didn't give one. Optional fuzzy match for view/delete/restore."
                ),
            },
            "content": {
                "type": "string",
                "description": "The full text body to write. Required for `create`.",
            },
            "source_type": {
                "type": "string",
                "description": "Origin of the document (default 'conversation').",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 10

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "search")

        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = DocumentService(db)
            body = _dispatch(service, action, params)
        except Exception as e:
            logger.error(f"[DOCUMENT SKILL] Error: {e}", exc_info=True)
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

        doc_artifacts = {}
        for row in results:
            source = row.get("source", "") or ""
            if source.startswith("document:"):
                doc_id = source.split(":", 1)[1]
            else:
                parts = (row.get("key", "") or "").split(":")
                doc_id = parts[1] if len(parts) >= 3 else "unknown"

            if doc_id not in doc_artifacts:
                doc_artifacts[doc_id] = []
            doc_artifacts[doc_id].append(row)

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
        logger.error(f"[DOCUMENT SKILL] Search failed: {e}", exc_info=True)
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


def _handle_view(service, params: dict) -> str:
    doc = _resolve_document(service, params)
    if not doc:
        return "[DOCUMENT] Document not found. Specify 'name' or 'id'."

    if doc.get("status") != "ready":
        return f"[DOCUMENT] '{doc['original_name']}' is still being processed or awaiting confirmation."

    meta = _parse_extracted_metadata(doc.get("extracted_metadata"))
    lines = [f"[DOCUMENT] {doc['original_name']}:"]

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
        companies = ", ".join(c["name"] for c in meta["companies"][:5])
        lines.append(f"  Companies: {companies}")

    if meta.get("dates"):
        dates = ", ".join(d["value"] for d in meta["dates"][:5])
        lines.append(f"  Dates: {dates}")

    if meta.get("expiration_dates"):
        exps = ", ".join(d["value"] for d in meta["expiration_dates"][:3])
        lines.append(f"  Expiration dates: {exps}")

    if meta.get("monetary_values"):
        vals = ", ".join(f"{v['currency']} {v['amount']}" for v in meta["monetary_values"][:5])
        lines.append(f"  Monetary values: {vals}")

    if meta.get("reference_numbers"):
        refs = ", ".join(r["value"] for r in meta["reference_numbers"][:5])
        lines.append(f"  References: {refs}")

    clean_text = doc.get("clean_text", "")
    if clean_text:
        lines.append(f"\n--- Full Document Text ---\n{clean_text}")
    else:
        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()
        try:
            with dgs.db.connection() as conn:
                cursor = conn.execute(
                    "SELECT value FROM data_graph WHERE source=? AND active=1 ORDER BY key",
                    (f'document:{doc["id"]}',),
                )
                fragments = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception as exc:
            logger.warning("[DOCUMENT SKILL] Fragment query failed: %s", exc, exc_info=True)
            fragments = []

        if fragments:
            full_text = "\n\n".join(fragments)
            lines.append(f"\n--- Full Document Text ---\n{full_text}")
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


def _split_into_artifacts(text: str, min_chars: int = 512, max_chars: int = 1024, overlap: int = 48) -> list:
    if not text:
        return []

    if len(text) <= min_chars:
        return [text]

    def _split_paragraph(para: str) -> list:
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

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for sentence in _split_paragraph(para):
                if len(buffer) + len(sentence) >= min_chars:
                    chunks.append(buffer + sentence)
                    buffer = ""
                else:
                    buffer += sentence + " "
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
            if len(buffer) >= min_chars:
                chunks.append(buffer)
                buffer = ""

    if buffer:
        chunks.append(buffer)

    if not chunks:
        return [text]

    result = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = result[-1][-overlap:]
        result.append(prefix + chunk)

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
        logger.error(f"[DOCUMENT SKILL] Create failed: {e}", exc_info=True)
        return f"[DOCUMENT] Failed to create document: {e}"
