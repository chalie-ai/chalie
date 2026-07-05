
import logging

logger = logging.getLogger(__name__)


def _split_paragraph_into_sentences(para: str, max_chars: int) -> list[str]:
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


def _absorb_long_para(buffer: str, para: str, chunks: list[str], min_chars: int, max_chars: int) -> str:
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


def _build_chunks(paragraphs: list[str], min_chars: int, max_chars: int) -> list[str]:
    chunks: list[str] = []
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


def split_into_artifacts(text: str, min_chars: int = 512, max_chars: int = 1024, overlap: int = 48) -> list[str]:
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
    artifacts = split_into_artifacts(text_content)

    from models.data_graph import DataGraph
    from services.data_graph_service import KIND_DOCUMENT

    for i, artifact_text in enumerate(artifacts):
        DataGraph.store(
            kind=KIND_DOCUMENT,
            key=f"doc:{doc_id}:{i:03d}",
            value=artifact_text,
            source=f"document:{doc_id}",
        )

    return len(artifacts)
