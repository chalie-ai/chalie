"""
DocumentAbility — Search and manage documents via the ACT loop.

LLM-callable actions: search, list, view, delete, restore, create.

``upload`` is NOT in the LLM ``INPUT_SCHEMA``: it is the single MECHANICAL ingest
for every uploaded file (chat attachments and the Documents library), invoked only
by code via ``ToolDispatcher.dispatch``. It takes a file PATH, never bytes — a
path is tiny, so the act-trail (which records every dispatch's params verbatim)
can never carry a multi-megabyte base64 blob and blow the context window. The
``_dispatch`` ``upload`` branch is retained for that mechanical route.

Every action returns a :class:`abilities._result.ToolResult` built only via
``ok()`` / ``err()``; the dispatcher renders the wire envelope. Errors carry a
stable kebab-case ``code`` (never the ``code="error"`` placeholder) so a weak
model can self-correct without re-reading the schema.

The destructive ``delete`` never silently picks the first fuzzy hit: a ``name``
that matches more than one document — or matches a single document only as a
non-exact substring — returns ``code=ambiguous-match`` with the candidate rows
(id, name, snippet) so the model can re-call with the exact ``id``. Deletion
proceeds only on an exact ``id`` or a single exact-name match.

Chunking/extraction-enqueue math lives in ``services.document_chunking`` (shared
by the library upload pipeline and the watched-folder ingest); this ability only
orchestrates it.
"""

import hashlib as _hashlib
import json as _json
import logging
import mimetypes as _mimetypes
import os as _os
import secrets as _secrets
import shutil as _shutil
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from services.document_chunking import create_document_artifacts

if TYPE_CHECKING:
    from services.document_service import DocumentService as _DocumentService

logger = logging.getLogger(__name__)

# Substring of a candidate's text shown in an ambiguous-match candidate row so the
# model has enough context to pick the right document by id.
_SNIPPET_CHARS = 120


class DocumentAbility(Ability):
    # Required params per action, validated by the dispatcher's ACTION_REQUIRED
    # pre-gate BEFORE the policy gate or run(). ``delete``/``view``/``restore``
    # require neither id nor name here because either addresses the document; the
    # handlers raise the precise missing-target error. ``upload`` is mechanical
    # (not in INPUT_SCHEMA) and validated inside ingest_file.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "search": (Keys.query,),
        "list": (),
        "view": (),
        "delete": (),
        "restore": (),
        "create": (Keys.name_, Keys.content),
        "upload": (),
    }

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

    def get_follow_up(self, tr: ToolResult) -> str:
        """Point the model to view a document's full text by id."""
        return "search and list only return document ids, names and snippets, not full text. If you need the actual content of a result, call document again with action=view and that id."

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["search", "list", "view", "delete", "restore", "create"],
                "description": (
                    "Document operation to perform; use `create` when the user asks you to "
                    "create, save, write, or store a note or document."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": "Text to search across all documents. Required for search.",
            },
            Keys.id: {
                "type": "string",
                "description": (
                    "Exact document ID (view, delete, restore). Prefer this over `name`: "
                    "delete only proceeds on an exact id or a single exact-name match."
                ),
            },
            Keys.name_: {
                "type": "string",
                "description": (
                    "Required for `create`; use a filename like 'research-notes.md' "
                    "if the user didn't give one. For view/delete/restore an EXACT name; "
                    "an inexact or non-unique name returns the candidate list to pick from by id."
                ),
            },
            Keys.content: {
                "type": "string",
                "description": "The full text body to write. Required for `create`.",
            },
        },
        "required": [Keys.action],
    }

    def run(self, params: dict[str, object]) -> ToolResult:
        action = cast(str, params.get(Keys.action, "search"))

        from services.document_service import DocumentService

        service = DocumentService()
        return _dispatch(service, action, params)


_VALID_ACTIONS = ("search", "list", "view", "delete", "restore", "create")


def _dispatch(service: "_DocumentService", action: str, params: dict[str, object]) -> ToolResult:
    if action == "search":
        return _handle_search(service, params)
    if action == "list":
        return _handle_list(service)
    if action == "view":
        return _handle_view(service, params)
    if action == "delete":
        return _handle_delete(service, params)
    if action == "restore":
        return _handle_restore(service, params)
    if action == "create":
        return _handle_create(service, params)
    if action == "upload":
        # Mechanical-only ingest (not in INPUT_SCHEMA): dispatched by code, never
        # the LLM. Path-based — see _handle_upload / module docstring.
        return _handle_upload(service, params)
    return ToolResult.err(
        f"Unknown action {action!r}.",
        code="unknown-action",
        valid=_VALID_ACTIONS,
    )


def _exact_name_matches(docs: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    lowered = name.lower()
    return [d for d in docs if cast(str, d.get("original_name") or "").lower() == lowered]


def _candidate_rows(docs: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for d in docs:
        snippet = cast(str, d.get("summary") or d.get("clean_text") or "").strip()[:_SNIPPET_CHARS]
        rows.append({
            "id": d.get("id"),
            "name": d.get("original_name"),
            "snippet": snippet,
        })
    return rows


def _resolve_unique(service: "_DocumentService", params: dict[str, object]) -> "dict[str, object] | ToolResult":
    """Resolve a single document for a destructive/addressed action.

    Returns the document dict on an unambiguous resolution, or an error
    ``ToolResult`` (``missing-target`` / ``not-found`` / ``ambiguous-match``).
    Resolution is consent-safe: an exact ``id`` always proceeds; a ``name``
    proceeds ONLY on a single exact-name match — more than one candidate, or a
    single non-exact (substring) match, returns ``ambiguous-match`` with the
    candidate rows rather than silently picking the first hit.
    """
    doc_id = cast(str, params.get(Keys.id) or "").strip()
    if doc_id:
        doc = service.get_document(doc_id)
        if not doc:
            return ToolResult.err(
                f"No document with id {doc_id!r}.",
                code="not-found",
                hint="call document with action=list to see existing ids.",
            )
        return doc

    name = cast(str, params.get(Keys.name_) or "").strip()
    if not name:
        return ToolResult.err(
            "id or name is required to address the document.",
            code="missing-target",
            hint="pass the document's id, or an exact name.",
        )

    candidates = service.search_documents_metadata(name)
    if not candidates:
        return ToolResult.err(
            f"No document matching {name!r} was found.",
            code="not-found",
            hint="call document with action=list to see what exists.",
        )

    exact = _exact_name_matches(candidates, name)
    if len(exact) == 1:
        return exact[0]

    # >1 candidate, or a single non-exact (substring) match, or several exact
    # matches → never a silent first-hit pick. Surface the candidate rows.
    rows = _candidate_rows(exact if len(exact) > 1 else candidates)
    return ToolResult.err(
        f"{name!r} matches {len(rows)} document(s); re-call with the exact id. "
        f"Candidates: {_json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}",
        code="ambiguous-match",
        hint="re-issue the action with action and id set to the chosen candidate.",
        count=len(rows),
    )


def _group_results_by_doc(results: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    doc_artifacts: dict[str, list[dict[str, object]]] = {}
    for row in results:
        source = cast(str, row.get("source", "") or "")
        if source.startswith("document:"):
            doc_id = source.split(":", 1)[1]
        else:
            parts = cast(str, row.get("key", "") or "").split(":")
            doc_id = parts[1] if len(parts) >= 3 else "unknown"
        doc_artifacts.setdefault(doc_id, []).append(row)
    return doc_artifacts


def _handle_search(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    query = cast(str, params.get(Keys.query, "")).strip()

    from services.memory_recall_service import recall

    results = recall(query, kinds=["document"], limit=10)

    rows = []
    for doc_id, artifacts in _group_results_by_doc(results).items():
        doc = service.get_document(doc_id)
        if not doc or doc.get("deleted_at"):
            continue
        rows.append({
            "id": doc_id,
            "name": doc.get("original_name", doc_id),
            "chunks": doc.get("chunk_count", 0),
            "matches": len(artifacts),
        })

    return ToolResult.ok(rows, action="search", count=len(rows))


def _handle_list(service: "_DocumentService") -> ToolResult:
    docs = [d for d in service.get_all_documents() if d.get("status") == "ready"]

    rows = []
    for doc in docs:
        meta = cast("dict[str, object]", doc.get("extracted_metadata") or {})
        doc_type = meta.get("document_type", {})
        if isinstance(doc_type, dict):
            doc_type = doc_type.get("value", "")
        elif not isinstance(doc_type, str):
            doc_type = ""
        rows.append({
            "id": doc.get("id"),
            "name": doc.get("original_name"),
            "type": doc_type,
            "pages": doc.get("page_count"),
            "chunks": doc.get("chunk_count", 0),
            "status": doc.get("status", "unknown"),
            "created_at": doc.get("created_at"),
        })

    return ToolResult.ok(rows, action="list", count=len(rows))


def _append_meta_summary(lines: list[str], doc: dict[str, object], meta: dict[str, object]) -> None:
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
        lines.append("  Companies: " + ", ".join(cast(str, c["name"]) for c in cast(list[dict[str, object]], meta["companies"])[:5]))
    if meta.get("dates"):
        lines.append("  Dates: " + ", ".join(cast(str, d["value"]) for d in cast(list[dict[str, object]], meta["dates"])[:5]))
    if meta.get("expiration_dates"):
        lines.append("  Expiration dates: " + ", ".join(cast(str, d["value"]) for d in cast(list[dict[str, object]], meta["expiration_dates"])[:3]))
    if meta.get("monetary_values"):
        lines.append("  Monetary values: " + ", ".join(
            f"{v['currency']} {v['amount']}" for v in cast(list[dict[str, object]], meta["monetary_values"])[:5]
        ))
    if meta.get("reference_numbers"):
        lines.append("  References: " + ", ".join(cast(str, r["value"]) for r in cast(list[dict[str, object]], meta["reference_numbers"])[:5]))


def _fetch_doc_fragments(doc_id: str) -> list[str]:
    from models.document import DocumentRow
    return [f.value for f in DocumentRow.for_document_id(doc_id).get() if f.value]


def _handle_view(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    doc = _resolve_unique(service, params)
    if isinstance(doc, ToolResult):
        return doc

    status = doc.get("status")
    if status != "ready":
        if status == "failed":
            return ToolResult.err(
                f"Document {doc['original_name']!r} failed to process.",
                code="extraction-failed",
                action="view",
            )
        return ToolResult.err(
            f"{doc['original_name']!r} is still being processed or awaiting confirmation.",
            code="not-ready",
            hint="try again once processing completes.",
            action="view",
        )

    meta = cast("dict[str, object]", doc.get("extracted_metadata") or {})
    lines = [f"[DOCUMENT] {doc['original_name']}:"]
    _append_meta_summary(lines, doc, meta)

    clean_text = doc.get("clean_text", "")
    if clean_text:
        lines.append(f"\n--- Full Document Text ---\n{clean_text}")
    else:
        fragments = _fetch_doc_fragments(cast(str, doc["id"]))
        if fragments:
            lines.append("\n--- Full Document Text ---\n" + "\n\n".join(fragments))
        else:
            lines.append("\n  (No text content available)")

    return ToolResult.ok("\n".join(lines), action="view", id=doc["id"])


def _handle_delete(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    doc = _resolve_unique(service, params)
    if isinstance(doc, ToolResult):
        return doc

    doc_id = cast(str, doc["id"])
    if not service.soft_delete(doc_id):
        return ToolResult.err(
            f"Failed to delete {doc['original_name']!r}.",
            code="delete-failed",
            action="delete",
        )

    from models.document import DocumentRow
    deleted_count = DocumentRow.purge_by_document_id(doc_id)

    return ToolResult.ok(
        {"id": doc_id, "name": doc["original_name"], "artifacts_removed": deleted_count},
        action="delete",
    )


def _handle_restore(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    doc_id = cast(str, params.get(Keys.id) or "").strip()
    name = cast(str, params.get(Keys.name_) or "").strip()

    if doc_id:
        doc = service.get_document(doc_id)
    elif name:
        deleted = [
            d for d in service.get_all_documents(include_deleted=True)
            if d.get("deleted_at") and cast(str, d.get("original_name") or "").lower() == name.lower()
        ]
        if len(deleted) > 1:
            rows = _candidate_rows(deleted)
            return ToolResult.err(
                f"{name!r} matches {len(rows)} deleted document(s); re-call with the exact id. "
                f"Candidates: {_json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}",
                code="ambiguous-match",
                hint="re-issue restore with id set to the chosen candidate.",
                count=len(rows),
            )
        doc = deleted[0] if deleted else None
    else:
        return ToolResult.err(
            "id or name is required to restore a document.",
            code="missing-target",
            hint="pass the document's id, or an exact name.",
        )

    if not doc:
        return ToolResult.err(
            "Document not found.",
            code="not-found",
            hint="call document with action=list to see what exists.",
        )
    if not doc.get("deleted_at"):
        return ToolResult.err(
            f"{doc['original_name']!r} is not deleted.",
            code="not-deleted",
            action="restore",
        )

    if not service.restore(cast(str, doc["id"])):
        return ToolResult.err(
            f"Failed to restore {doc['original_name']!r}.",
            code="restore-failed",
            action="restore",
        )
    return ToolResult.ok(
        {"id": doc["id"], "name": doc["original_name"]},
        action="restore",
    )


def ingest_file(
    service: "_DocumentService",
    src_path: str,
    *,
    name: "str | None" = None,
    subdir: "str | None" = None,
    source_type: str = "upload",
) -> dict[str, object]:
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
    from api.documents import _mark_upload_failed, _run_upload_extraction
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


def _handle_upload(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    """Mechanical upload dispatch (not LLM-callable).

    Thin orchestrator over ``ingest_file``. The structured success body
    ``{"id","hash","name","status"}`` is parsed by the turn-0 seed
    (``message_processor._SEED_UPLOAD_ID_RE``) to build the transcript-doc link.
    """
    result = ingest_file(
        service,
        cast(str, params.get(Keys.path)),
        name=cast("str | None", params.get(Keys.name_)),
    )
    if result.get("error"):
        return ToolResult.err(cast(str, result["error"]), code="upload-failed", action="upload")
    if result.get("status") != "ready":
        return ToolResult.err(
            f"Failed to process document {result['name']!r}.",
            code="extraction-failed",
            action="upload",
            id=result["id"],
        )
    return ToolResult.ok(
        {
            "id": result["id"],
            "hash": result["hash"],
            "name": result["name"],
            "status": result["status"],
        },
        action="upload",
    )


def _handle_create(service: "_DocumentService", params: dict[str, object]) -> ToolResult:
    name = cast(str, params.get(Keys.name_, "")).strip()
    content = cast(str, params.get(Keys.content, "")).strip()

    if "." not in name:
        name = f"{name}.md"

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

    return ToolResult.ok(
        {"id": doc_id, "name": name, "artifacts": artifact_count},
        action="create",
    )
