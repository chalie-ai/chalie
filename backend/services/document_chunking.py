"""Document chunking + artifact-storage engine.

The substance behind document ingestion's "split extracted text into searchable
fragments and persist them" step. Pure, deterministic packing logic (paragraph →
sentence → fixed-width fallback) plus the data-graph write that enqueues each
fragment for embedding + FTS indexing.

Lives in ``services`` (not on the ``document`` ability) because three callers
need it and two of them are not abilities: the Documents library upload pipeline
(``api/documents.py``) and the watched-folder ingest (``folder_watcher_service``),
alongside the ability's own ``create`` action and ``ingest_file`` orchestration.
Keeping the engine here lets the non-ability callers reach it without importing
an ability.

The chunk sizing (``min_chars=512``, ``max_chars=1024``, ``overlap=48``) and the
``KIND_DOCUMENT`` / ``document:<id>`` / ``doc:<id>:<NNN>`` key scheme are the
established contract — preserved verbatim so existing indexes and search keys
stay valid.
"""

import logging

logger = logging.getLogger(__name__)


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


def split_into_artifacts(text: str, min_chars: int = 512, max_chars: int = 1024, overlap: int = 48) -> list:
    """Split extracted document text into overlapping artifact fragments."""
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
    artifacts = split_into_artifacts(text_content)

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
