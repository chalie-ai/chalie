"""
Document Skill — Search and manage uploaded documents via the ACT loop.

Actions: search, list, view, delete, restore
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def handle_document(topic: str, params: dict) -> str:
    """
    Search and manage the document library.

    Actions:
    - search:   Semantic search across all documents
    - list:     List all active documents
    - view:     View metadata and content preview for a specific document
    - delete:   Soft-delete a document
    - restore:  Restore a soft-deleted document

    Args:
        topic: Current conversation topic
        params: Action parameters dict

    Returns:
        Formatted result string
    """
    action = params.get('action', 'search')

    try:
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        service = DocumentService(db)
        return _dispatch(service, action, params, topic)

    except Exception as e:
        logger.error(f"[DOCUMENT SKILL] Error: {e}", exc_info=True)
        return f"[DOCUMENT] Error: {e}"


def _dispatch(service, action: str, params: dict, topic: str) -> str:
    if action == 'search':
        return _handle_search(service, params, topic)
    elif action == 'list':
        return _handle_list(service, topic)
    elif action == 'view':
        return _handle_view(service, params, topic)
    elif action == 'delete':
        return _handle_delete(service, params, topic)
    elif action == 'restore':
        return _handle_restore(service, params, topic)
    else:
        valid = 'search, list, view, delete, restore'
        return f"[DOCUMENT] Unknown action '{action}'. Use: {valid}"


def _resolve_document(service, params: dict) -> Optional[dict]:
    """Resolve document by id or name (fuzzy)."""
    doc_id = params.get('id', '').strip()
    name = params.get('name', '').strip()

    if doc_id:
        return service.get_document(doc_id)

    if name:
        # Search by name
        docs = service.search_documents_metadata(name)
        if docs:
            return docs[0]

    return None


def _handle_search(service, params: dict, topic: str) -> str:
    """
    Phase 1: Identify which documents are relevant.

    Returns lightweight metadata (names, types, IDs) — NOT content.
    Use action "view" with a document id to load the full text for analysis.
    """
    query = params.get('query', '').strip()
    if not query:
        return "[DOCUMENT] 'query' is required for search."

    try:
        from services.embedding_service import get_embedding_service

        embedding_service = get_embedding_service()
        query_embedding = embedding_service.generate_embedding(query)

        results = service.search_chunks(query_embedding, query, limit=5)

        if not results:
            return f"[DOCUMENT] No documents match '{query}'."

        # Deduplicate by document_id, skip unconfirmed documents
        seen_docs = {}
        for r in results:
            doc_id = r['document_id']
            if doc_id not in seen_docs:
                doc = service.get_document(doc_id)
                if doc and doc.get('status') == 'ready':
                    seen_docs[doc_id] = doc

        lines = [f"[DOCUMENT] Found {len(seen_docs)} document(s) matching '{query}':"]
        for doc_id, doc in seen_docs.items():
            meta = doc.get('extracted_metadata', {})
            doc_type = meta.get('document_type', {}).get('value', '')
            type_str = f" [{doc_type}]" if doc_type and doc_type != 'document' else ''
            pages = doc.get('page_count')
            page_str = f", {pages}p" if pages else ''
            created = doc.get('created_at')
            date_str = ''
            if created:
                try:
                    date_str = f", uploaded {created.strftime('%b %d')}"
                except Exception:
                    pass

            lines.append(
                f"  · id={doc_id}: \"{doc['original_name']}\"{type_str}"
                f" ({doc.get('chunk_count', 0)} chunks{page_str}{date_str})"
            )

        lines.append("\nUse action \"view\" with the document id to read its full content.")
        return '\n'.join(lines)

    except Exception as e:
        logger.error(f"[DOCUMENT SKILL] Search failed: {e}", exc_info=True)
        return f"[DOCUMENT] Search failed: {e}"


def _handle_list(service, topic: str) -> str:
    """List all confirmed (ready) documents."""
    docs = service.get_all_documents()
    docs = [d for d in docs if d.get('status') == 'ready']

    if not docs:
        return "[DOCUMENT] No documents in library."

    lines = ["[DOCUMENT] Document library:"]
    for doc in docs:
        doc_type = doc.get('extracted_metadata', {}).get('document_type', {}).get('value', '')
        type_str = f" [{doc_type}]" if doc_type and doc_type != 'document' else ''
        pages = doc.get('page_count')
        page_str = f", {pages}p" if pages else ''
        status = doc.get('status', 'unknown')
        created = doc.get('created_at')
        date_str = ''
        if created:
            try:
                date_str = f", uploaded {created.strftime('%b %d')}"
            except Exception:
                pass

        lines.append(
            f"  · {doc['original_name']}{type_str}"
            f" ({status}{page_str}, {doc.get('chunk_count', 0)} chunks{date_str})"
        )

    return '\n'.join(lines)


def _handle_view(service, params: dict, topic: str) -> str:
    """
    Phase 2: Load full document content for analysis.

    Returns complete metadata + full extracted text so the LLM can
    reason over the entire document, not fragments.
    """
    doc = _resolve_document(service, params)
    if not doc:
        return "[DOCUMENT] Document not found. Specify 'name' or 'id'."

    if doc.get('status') != 'ready':
        return f"[DOCUMENT] '{doc['original_name']}' is still being processed or awaiting confirmation."

    # Format for act_history — full content for analysis
    meta = doc.get('extracted_metadata', {})
    lines = [f"[DOCUMENT] {doc['original_name']}:"]

    doc_type = meta.get('document_type', {}).get('value', '')
    if doc_type:
        lines.append(f"  Type: {doc_type}")

    if doc.get('page_count'):
        lines.append(f"  Pages: {doc['page_count']}")

    # Show extracted metadata signals
    if meta.get('companies'):
        companies = ', '.join(c['name'] for c in meta['companies'][:5])
        lines.append(f"  Companies: {companies}")

    if meta.get('dates'):
        dates = ', '.join(d['value'] for d in meta['dates'][:5])
        lines.append(f"  Dates: {dates}")

    if meta.get('expiration_dates'):
        exps = ', '.join(d['value'] for d in meta['expiration_dates'][:3])
        lines.append(f"  Expiration dates: {exps}")

    if meta.get('monetary_values'):
        vals = ', '.join(f"{v['currency']} {v['amount']}" for v in meta['monetary_values'][:5])
        lines.append(f"  Monetary values: {vals}")

    if meta.get('reference_numbers'):
        refs = ', '.join(r['value'] for r in meta['reference_numbers'][:5])
        lines.append(f"  References: {refs}")

    # Full document text — the whole point of the view action
    clean_text = doc.get('clean_text', '')
    if clean_text:
        lines.append(f"\n--- Full Document Text ---\n{clean_text}")
    else:
        # Fallback to concatenated chunks if clean_text not available
        chunks = service.get_chunks_for_document(doc['id'])
        if chunks:
            full_text = '\n\n'.join(c['content'] for c in chunks)
            lines.append(f"\n--- Full Document Text ---\n{full_text}")
        else:
            lines.append("\n  (No text content available)")

    return '\n'.join(lines)


def _handle_delete(service, params: dict, topic: str) -> str:
    """Soft-delete a document."""
    doc = _resolve_document(service, params)
    if not doc:
        return "[DOCUMENT] Document not found. Specify 'name' or 'id'."

    success = service.soft_delete(doc['id'])
    if success:
        try:
            from services.document_card_service import DocumentCardService
            DocumentCardService().emit_delete_card(topic, doc['original_name'])
        except Exception as card_err:
            logger.warning(f"[DOCUMENT SKILL] Card emit failed (non-fatal): {card_err}")
        return f"[DOCUMENT] Deleted '{doc['original_name']}'. Can be restored within 30 days."
    return f"[DOCUMENT] Failed to delete '{doc['original_name']}'."


def _handle_restore(service, params: dict, topic: str) -> str:
    """Restore a soft-deleted document."""
    doc_id = params.get('id', '').strip()
    name = params.get('name', '').strip()

    if doc_id:
        doc = service.get_document(doc_id)
    elif name:
        # Search including deleted
        all_docs = service.get_all_documents(include_deleted=True)
        doc = next((d for d in all_docs if name.lower() in d['original_name'].lower()
                     and d.get('deleted_at')), None)
    else:
        return "[DOCUMENT] Specify 'name' or 'id' to restore."

    if not doc:
        return "[DOCUMENT] Document not found."

    if not doc.get('deleted_at'):
        return f"[DOCUMENT] '{doc['original_name']}' is not deleted."

    success = service.restore(doc['id'])
    if success:
        try:
            from services.document_card_service import DocumentCardService
            DocumentCardService().emit_restore_card(topic, doc['original_name'])
        except Exception as card_err:
            logger.warning(f"[DOCUMENT SKILL] Card emit failed (non-fatal): {card_err}")
        return f"[DOCUMENT] Restored '{doc['original_name']}'."
    return f"[DOCUMENT] Failed to restore '{doc['original_name']}'."
