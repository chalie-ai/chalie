"""
Document Processing Service — Text extraction, metadata extraction, chunking, and embedding.

Pipeline: extract text → detect language → extract metadata → generate summary →
          chunk text → embed chunks → store everything.

All metadata extraction is deterministic (regex, no LLM). Confidence scores
reflect pattern specificity × surrounding context quality.

Text extraction is delegated to services.text_extractor (shared with the `read` innate skill).
"""

import hashlib
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from services.text_extractor import extract_text as _extract_text_from_file
from services.text_extractor import normalize_text as _normalize_text_fn

logger = logging.getLogger(__name__)

# Adaptive chunk sizing by document type
CHUNK_PARAMS = {
    'contract':  {'chunk_size': 800, 'overlap': 150},
    'policy':    {'chunk_size': 800, 'overlap': 150},
    'agreement': {'chunk_size': 800, 'overlap': 150},
    'manual':    {'chunk_size': 500, 'overlap': 100},
    'guide':     {'chunk_size': 500, 'overlap': 100},
    'receipt':   {'chunk_size': 200, 'overlap': 50},
    'invoice':   {'chunk_size': 200, 'overlap': 50},
    'default':   {'chunk_size': 500, 'overlap': 100},
}

# Document type detection keywords
DOC_TYPE_KEYWORDS = {
    'warranty': ['warranty', 'coverage', 'defect', 'manufacturer', 'repair', 'replacement'],
    'contract': ['contract', 'agreement', 'party', 'parties', 'obligations', 'binding', 'clause'],
    'invoice': ['invoice', 'bill', 'amount due', 'payment', 'subtotal', 'total due'],
    'receipt': ['receipt', 'transaction', 'paid', 'purchased', 'order confirmation'],
    'manual': ['manual', 'instructions', 'operating', 'user guide', 'how to', 'troubleshooting'],
    'policy': ['policy', 'terms', 'conditions', 'coverage', 'exclusions', 'premium'],
    'agreement': ['agreement', 'lease', 'rental', 'tenant', 'landlord', 'signed'],
}

# Summary length (chars)
SUMMARY_MAX_CHARS = 500


class DocumentProcessingService:
    """Processes uploaded documents: extract, analyze, chunk, embed."""

    def __init__(self, db_service=None):
        """Initialize the document processing service.

        Args:
            db_service: Optional :class:`~services.database_service.DatabaseService`
                instance. Falls back to the shared singleton when ``None``.
        """
        self.db = db_service

    def process_document(self, doc_id: str) -> bool:
        """
        Orchestrate full document processing pipeline.

        Returns True on success, False on failure.
        """
        from services.document_service import DocumentService, DOCUMENTS_ROOT
        from services.database_service import get_shared_db_service

        db = self.db or get_shared_db_service()
        doc_service = DocumentService(db)

        doc = doc_service.get_document(doc_id)
        if not doc:
            logger.error(f"[DOC PROC] Document {doc_id} not found")
            return False

        try:
            import os
            import time as _time

            t_start = _time.monotonic()
            def _elapsed(since=None):
                return f"{_time.monotonic() - (since or t_start):.1f}s"

            # Step 1: Set processing status
            doc_service.update_status(doc_id, 'processing')

            # Watched folder docs store absolute paths; uploaded docs store relative paths.
            # os.path.join already handles this correctly (absolute second arg wins).
            file_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])
            fname = doc.get('original_name', doc_id)
            logger.info(f"[DOC PROC] Starting: {fname} ({doc_id})")

            # Step 2: Extract text
            t2 = _time.monotonic()
            text = _extract_text_from_file(file_path, doc['mime_type'])

            # Step 2b: OCR fallback for image-only PDFs and images
            if not text or not text.strip():
                text = self._try_ocr(file_path, doc['mime_type'])

            if not text or not text.strip():
                doc_service.update_status(doc_id, 'failed', 'No text could be extracted from this document.')
                return False

            logger.info(f"[DOC PROC] {fname} — text extraction: {_elapsed(t2)} ({len(text):,} chars)")

            # Step 3: Clean text and detect language
            clean_text = _normalize_text_fn(text)
            language = self._detect_language(text)

            if not clean_text or not clean_text.strip():
                doc_service.update_status(doc_id, 'failed',
                                          'Text extraction produced empty content after normalization.')
                return False

            # Step 4: Extract structured metadata
            metadata = self._extract_metadata(text, language)

            # Step 5: Generate summary
            summary = self._generate_summary(clean_text)

            # Step 6: Generate summary embedding + fingerprint
            t6 = _time.monotonic()
            from services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            summary_embedding = embedding_service.generate_embedding(summary)
            fingerprint = self._simhash(clean_text)
            logger.info(f"[DOC PROC] {fname} — summary embedding: {_elapsed(t6)}")

            # Step 7: Store metadata
            page_count = self._count_pages(file_path, doc['mime_type'])
            doc_service.update_extracted_metadata(
                doc_id,
                metadata=metadata,
                summary=summary,
                summary_embedding=summary_embedding,
                clean_text=clean_text,
                language=language,
                fingerprint=fingerprint,
                page_count=page_count,
            )

            # Step 8: Check for duplicates (informational — stored in metadata)
            text_length = len(clean_text) if clean_text else 0
            duplicates = doc_service.find_duplicates(doc['file_hash'], summary_embedding, text_length, exclude_id=doc_id)
            if duplicates:
                # Store duplicate info in metadata for the API to relay
                metadata['_duplicates'] = [
                    {'id': d['id'], 'name': d['original_name'], 'match_type': d['match_type']}
                    for d in duplicates
                ]
                doc_service.update_extracted_metadata(
                    doc_id, metadata=metadata, summary=summary,
                    summary_embedding=summary_embedding,
                )

            # Step 9: Adaptive chunking
            doc_type = metadata.get('document_type', {}).get('value', 'default')
            chunks = self._chunk_text(text, doc_type)

            # Step 10: Embed all chunks in adaptive batches
            t10 = _time.monotonic()
            chunk_texts = [c['content'] for c in chunks]
            chunk_embeddings = self._generate_chunk_embeddings(embedding_service, chunk_texts)
            logger.info(f"[DOC PROC] {fname} — chunk embeddings: {_elapsed(t10)} ({len(chunks)} chunks)")

            # Step 11: Store chunks
            chunk_records = []
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = chunk_embeddings[i].tolist()
                chunk['chunk_index'] = i
                chunk_records.append(chunk)

            doc_service.store_chunks(doc_id, chunk_records)

            if not chunk_records:
                doc_service.update_status(doc_id, 'failed',
                                          'No text chunks could be created from this document.')
                return False

            # Step 12: Set status — watched folder and moment docs auto-confirm
            if doc.get('source_type') in ('watched_folder', 'moment'):
                doc_service.update_status(doc_id, 'ready',
                                          chunk_count=len(chunk_records))
                logger.info(f"[DOC PROC] {fname} auto-confirmed ({doc['source_type']}): "
                            f"{len(chunk_records)} chunks — pipeline so far: {_elapsed()}")
            else:
                doc_service.update_status(doc_id, 'awaiting_confirmation',
                                          chunk_count=len(chunk_records))
                logger.info(f"[DOC PROC] {fname} processed: {len(chunk_records)} chunks — "
                            f"pipeline so far: {_elapsed()}")

            # Step 13: LLM synthesis (non-blocking enrichment, 60s hard timeout)
            # Runs AFTER status change so the user isn't stuck waiting.
            # If it succeeds, metadata is updated with synthesis + key_facts.
            # If it fails or times out, the confirmation card shows the truncated summary instead.
            # NOTE: signal.alarm() only works in the main thread; use a thread-based timeout instead.
            import threading
            _synthesis_result = {}
            t13 = _time.monotonic()
            def _run_synthesis():
                try:
                    result = self._generate_llm_synthesis(
                        clean_text, metadata, doc['original_name'])
                    if result:
                        _synthesis_result['data'] = result
                except Exception as e:
                    logger.warning(f"[DOC PROC] Post-confirmation synthesis failed (non-fatal): {e}")

            synth_thread = threading.Thread(target=_run_synthesis, daemon=True)
            synth_thread.start()
            synth_thread.join(timeout=60)

            if synth_thread.is_alive():
                logger.warning(f"[DOC PROC] {fname} synthesis timed out after 60s")
            elif _synthesis_result.get('data'):
                synthesis_data = _synthesis_result['data']
                metadata['_synthesis'] = synthesis_data.get('synthesis', '')
                metadata['_key_facts'] = synthesis_data.get('key_facts', [])
                doc_service.update_extracted_metadata(
                    doc_id, metadata=metadata, summary=summary,
                )
                logger.info(f"[DOC PROC] {fname} — LLM synthesis: {_elapsed(t13)}")

            logger.info(f"[DOC PROC] {fname} complete — total: {_elapsed()}")

            # Step 14: Document classification (non-fatal enrichment)
            # Infers category, project, and date via LLM.
            try:
                from services.document_classification_service import DocumentClassificationService
                folder_context = ''
                if doc.get('watched_folder_id'):
                    from services.folder_watcher_service import FolderWatcherService
                    watcher_svc = FolderWatcherService(db)
                    folder = watcher_svc.get_folder(doc['watched_folder_id'])
                    if folder:
                        folder_context = f"Watched folder: {folder.get('label') or folder['folder_path']}"
                        rel = os.path.relpath(os.path.dirname(doc['file_path']), folder['folder_path'])
                        if rel != '.':
                            folder_context += f" / {rel}"

                cls_svc = DocumentClassificationService(db)
                cls_svc.classify_document(
                    doc_id=doc_id,
                    summary=summary,
                    clean_text=clean_text,
                    metadata=metadata,
                    original_name=doc['original_name'],
                    folder_context=folder_context,
                )
            except Exception as e:
                logger.warning(f"[DOC PROC] Classification failed (non-fatal): {e}")

            return True

        except Exception as e:
            logger.error(f"[DOC PROC] Processing failed for {doc_id}: {e}", exc_info=True)
            doc_service.update_status(doc_id, 'failed', str(e)[:500])
            return False

    # ─────────────────────────────────────────────
    # Metadata extraction (deterministic, no LLM)
    # ─────────────────────────────────────────────

    def _extract_metadata(self, text: str, language: str = 'en') -> dict:
        """Extract structured metadata from document text using regex patterns.

        Args:
            text: Raw extracted text from the document.
            language: Detected language code (default ``'en'``).

        Returns:
            Metadata dict with keys ``dates``, ``expiration_dates``, ``companies``,
            ``monetary_values``, ``people``, ``reference_numbers``,
            ``document_type``, ``key_terms``, ``language``, and
            ``extraction_warnings``.
        """
        metadata = {
            'dates': [],
            'expiration_dates': [],
            'companies': [],
            'monetary_values': [],
            'people': [],
            'reference_numbers': [],
            'document_type': {},
            'key_terms': [],
            'language': language or 'en',
            'extraction_warnings': [],
        }

        try:
            metadata['dates'] = self._extract_dates(text)
            metadata['expiration_dates'] = self._extract_expiration_dates(text)
            metadata['companies'] = self._extract_companies(text)
            metadata['monetary_values'] = self._extract_monetary_values(text)
            metadata['reference_numbers'] = self._extract_reference_numbers(text)
            metadata['document_type'] = self._classify_document_type(text)
            metadata['key_terms'] = self._extract_key_terms(text)
        except Exception as e:
            logger.warning(f"[DOC PROC] Metadata extraction partial failure: {e}")
            metadata['extraction_warnings'].append(f"partial_extraction_failure: {e}")

        return metadata

    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract dates from text using multiple format patterns.

        Args:
            text: Document text to scan.

        Returns:
            List of date dicts (up to 20), each with ``value``, ``label``,
            ``context``, and ``confidence`` keys.
        """
        results = []
        seen = set()

        # ISO dates: 2025-03-15
        for m in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', text):
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                context = text[max(0, m.start() - 40):m.end() + 40].strip()
                results.append({
                    'value': f"{val}T00:00:00Z",
                    'label': 'date',
                    'context': context,
                    'confidence': 0.95,
                })

        # Natural dates: March 15, 2025 or 15 March 2025
        months = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        for m in re.finditer(rf'\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b', text, re.IGNORECASE):
            val = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
            if val not in seen:
                seen.add(val)
                context = text[max(0, m.start() - 40):m.end() + 40].strip()
                results.append({
                    'value': val,
                    'label': 'date',
                    'context': context,
                    'confidence': 0.90,
                })

        # DD/MM/YYYY or MM/DD/YYYY
        for m in re.finditer(r'\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b', text):
            val = m.group(0)
            if val not in seen:
                seen.add(val)
                context = text[max(0, m.start() - 40):m.end() + 40].strip()
                results.append({
                    'value': val,
                    'label': 'date',
                    'context': context,
                    'confidence': 0.75,
                })

        return results[:20]  # Cap to avoid noise

    def _extract_expiration_dates(self, text: str) -> List[Dict]:
        """Extract expiration and validity dates from document text.

        Args:
            text: Document text to scan for expiration patterns.

        Returns:
            List of date dicts (up to 10), each with ``value``, ``label``,
            ``context``, and ``confidence`` keys.
        """
        results = []
        patterns = [
            r'(?:valid\s+until|expires?\s+(?:on)?|coverage\s+ends?|expir(?:ation|y)\s+date|'
            r'effective\s+(?:through|until)|g[üu]ltig\s+bis|validu?\s+sa)\s*[:\-]?\s*'
            r'(\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{4}|'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
            r'\s+\d{1,2},?\s+\d{4})',
        ]

        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                date_str = m.group(1)
                context = text[max(0, m.start() - 40):m.end() + 40].strip()
                results.append({
                    'value': date_str,
                    'label': 'expiration',
                    'context': context,
                    'confidence': 0.85,
                })

        return results[:10]

    def _extract_companies(self, text: str) -> List[Dict]:
        """Extract company and organization names from document text.

        Args:
            text: Document text to scan for company name patterns.

        Returns:
            List of company dicts (up to 10), each with ``name`` and
            ``confidence`` keys.
        """
        results = []
        seen = set()

        # Names near indicators: "by", "from", "manufacturer:", etc.
        company_patterns = [
            r'(?:(?:manufactured|made|produced|provided|issued|sold)\s+by|'
            r'(?:from|manufacturer|company|corporation|organization|vendor|supplier|insurer)\s*[:\-]?)\s+'
            r'([A-Z][A-Za-z\s&.,]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|GmbH|SA|SL)?)',
        ]

        for pattern in company_patterns:
            for m in re.finditer(pattern, text):
                name = m.group(1).strip().rstrip('.,')
                name_lower = name.lower()
                if name_lower not in seen and len(name) > 2:
                    seen.add(name_lower)
                    results.append({'name': name, 'confidence': 0.80})

        # Also look for common suffixes indicating company names
        for m in re.finditer(r'\b([A-Z][A-Za-z\s&]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|GmbH))\b', text):
            name = m.group(1).strip()
            name_lower = name.lower()
            if name_lower not in seen and len(name) > 3:
                seen.add(name_lower)
                results.append({'name': name, 'confidence': 0.70})

        return results[:10]

    def _extract_monetary_values(self, text: str) -> List[Dict]:
        """Extract monetary values with currency detection from document text.

        Args:
            text: Document text to scan for currency patterns.

        Returns:
            List of monetary value dicts (up to 15), each with ``amount``,
            ``currency``, ``context``, and ``confidence`` keys.
        """
        results = []
        seen = set()

        # $1,299.99 or €1.299,99 or 1,299.99 USD
        patterns = [
            (r'([$€£¥])\s*(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)', 'symbol_prefix'),
            (r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)\s*(USD|EUR|GBP|JPY|CHF|AUD|CAD)', 'code_suffix'),
        ]

        currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}

        for pattern, ptype in patterns:
            for m in re.finditer(pattern, text):
                if ptype == 'symbol_prefix':
                    currency = currency_map.get(m.group(1), m.group(1))
                    amount_str = m.group(2)
                else:
                    currency = m.group(2)
                    amount_str = m.group(1)

                # Normalize amount
                amount_str = amount_str.replace(' ', '')
                # Detect locale: if last separator is comma with 2 digits → EU format
                if re.match(r'.*,\d{2}$', amount_str):
                    amount_str = amount_str.replace('.', '').replace(',', '.')
                else:
                    amount_str = amount_str.replace(',', '')

                try:
                    amount = float(amount_str)
                except ValueError:
                    continue

                key = f"{currency}:{amount}"
                if key not in seen:
                    seen.add(key)
                    context = text[max(0, m.start() - 40):m.end() + 40].strip()
                    results.append({
                        'amount': amount,
                        'currency': currency,
                        'context': context,
                        'confidence': 0.85,
                    })

        return results[:15]

    def _extract_reference_numbers(self, text: str) -> List[Dict]:
        """Extract reference identifiers such as policy numbers, serial numbers, and order IDs.

        Args:
            text: Document text to scan for reference number patterns.

        Returns:
            List of reference dicts (up to 10), each with ``type``,
            ``value``, and ``confidence`` keys.
        """
        results = []
        patterns = [
            (r'(?:policy|serial|order|ref(?:erence)?|invoice|account|claim|case|ticket|contract)'
             r'\s*(?:#|no\.?|number|id|nr\.?|numru)\s*[:\-]?\s*([A-Z0-9][\w\-]{3,30})',
             'reference'),
        ]

        seen = set()
        for pattern, ref_type in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(1).strip()
                if val not in seen and len(val) > 3:
                    seen.add(val)
                    results.append({
                        'type': ref_type,
                        'value': val,
                        'confidence': 0.85,
                    })

        return results[:10]

    def _classify_document_type(self, text: str) -> Dict:
        """Classify document type based on keyword density scoring.

        Args:
            text: Document text to analyse.

        Returns:
            Dict with ``value`` (detected type string) and ``confidence`` (float
            in [0.30, 0.95]).
        """
        text_lower = text.lower()
        scores = {}

        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score

        if not scores:
            return {'value': 'document', 'confidence': 0.30}

        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        total_keywords = len(DOC_TYPE_KEYWORDS[best_type])
        confidence = min(0.95, 0.40 + (max_score / total_keywords) * 0.55)

        return {'value': best_type, 'confidence': round(confidence, 2)}

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract significant terms from document text via word frequency analysis.

        Args:
            text: Document text to analyse.

        Returns:
            List of up to 10 high-frequency term strings, filtered for stopwords.
        """
        # Tokenize into words, filter stopwords and short terms
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'not', 'no', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
            'their', 'them', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her',
            'all', 'each', 'every', 'both', 'any', 'such', 'if', 'when', 'than',
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = {}
        for word in words:
            if word not in stopwords:
                freq[word] = freq.get(word, 0) + 1

        # Extract bigrams as key terms
        terms = []
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:10]:
            if count >= 2:
                terms.append(word)

        return terms[:10]

    # ─────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        doc_type: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[Dict]:
        """
        Adaptive section-aware text chunking.
        Prefers paragraph/heading boundaries over mid-sentence cuts.
        """
        params = CHUNK_PARAMS.get(doc_type, CHUNK_PARAMS['default'])
        chunk_size = chunk_size or params['chunk_size']
        overlap = overlap or params['overlap']

        # Approximate tokens ≈ words * 1.3
        chunk_words = int(chunk_size / 1.3)
        overlap_words = int(overlap / 1.3)

        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_words = 0
        current_page = None
        current_section = None

        for para in paragraphs:
            # Detect page markers
            page_match = re.match(r'^\[Page (\d+)\]', para)
            if page_match:
                current_page = int(page_match.group(1))
                para = re.sub(r'^\[Page \d+\]\s*', '', para)

            # Detect slide markers
            slide_match = re.match(r'^\[Slide (\d+)\]', para)
            if slide_match:
                current_page = int(slide_match.group(1))
                para = re.sub(r'^\[Slide \d+\]\s*', '', para)

            # Detect section headers
            heading_match = re.match(r'^(#{1,6})\s+(.+)', para)
            if heading_match:
                current_section = heading_match.group(2).strip()

            para_words = len(para.split())

            # If adding this paragraph exceeds the chunk size, flush current chunk
            if current_words + para_words > chunk_words and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'page_number': current_page,
                    'section_title': current_section,
                    'token_count': int(current_words * 1.3),
                })

                # Overlap: keep last few lines
                overlap_text = []
                overlap_count = 0
                for line in reversed(current_chunk):
                    line_words = len(line.split())
                    if overlap_count + line_words > overlap_words:
                        break
                    overlap_text.insert(0, line)
                    overlap_count += line_words

                current_chunk = overlap_text
                current_words = overlap_count

            current_chunk.append(para)
            current_words += para_words

        # Flush remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'page_number': current_page,
                'section_title': current_section,
                'token_count': int(current_words * 1.3),
            })

        return chunks

    def _generate_chunk_embeddings(self, embedding_service, texts: List[str]) -> list:
        """Generate embeddings for a list of text chunks using adaptive batch sizing.

        Args:
            embedding_service: :class:`~services.embedding_service.EmbeddingService`
                instance used for batch embedding.
            texts: List of text strings to embed.

        Returns:
            List of numpy arrays (one per input text), L2-normalized.
        """
        if not texts:
            return []

        # Adaptive batch size based on document size
        num_chunks = len(texts)
        if num_chunks > 200:
            batch_size = 8
        elif num_chunks > 50:
            batch_size = 16
        else:
            batch_size = 32

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = embedding_service.generate_embeddings_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _generate_llm_synthesis(
        self, clean_text: str, metadata: dict, original_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an LLM synthesis of the document for user confirmation.

        Returns {"synthesis": "...", "key_facts": [...]} or None on failure.
        Non-fatal — caller proceeds with truncated summary if this fails.
        """
        try:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service
            import json

            agent_cfg = ConfigService.resolve_agent_config('document-synthesis')
            prompt_template = ConfigService.get_agent_prompt('document-synthesis')
            if not prompt_template:
                logger.warning("[DOC PROC] No synthesis prompt found")
                return None

            # Build metadata summary for the prompt
            doc_type = metadata.get('document_type', {}).get('value', 'document')
            meta_lines = []
            if metadata.get('companies'):
                meta_lines.append(f"Companies: {', '.join(c['name'] for c in metadata['companies'][:5])}")
            if metadata.get('dates'):
                meta_lines.append(f"Dates: {', '.join(d['value'] for d in metadata['dates'][:5])}")
            if metadata.get('expiration_dates'):
                meta_lines.append(f"Expiration dates: {', '.join(d['value'] for d in metadata['expiration_dates'][:3])}")
            if metadata.get('monetary_values'):
                money_parts = [f"{v['currency']} {v['amount']}" for v in metadata['monetary_values'][:5]]
                meta_lines.append(f"Monetary values: {', '.join(money_parts)}")
            if metadata.get('reference_numbers'):
                meta_lines.append(f"References: {', '.join(r['value'] for r in metadata['reference_numbers'][:5])}")
            metadata_summary = '\n'.join(meta_lines) if meta_lines else 'No structured metadata extracted.'

            # Truncate text for LLM context
            truncated_text = clean_text[:3000] if clean_text else ''

            system_prompt = (prompt_template
                             .replace('{{original_name}}', original_name or 'Unknown')
                             .replace('{{document_type}}', doc_type)
                             .replace('{{metadata_summary}}', metadata_summary)
                             .replace('{{clean_text}}', truncated_text))

            llm = create_llm_service(agent_cfg)
            response = llm.send_message(system_prompt, "Synthesize this document.")

            result = json.loads(response.text)
            if 'synthesis' in result:
                return result

            logger.warning(f"[DOC PROC] Synthesis response missing 'synthesis' key")
            return None

        except Exception as e:
            logger.warning(f"[DOC PROC] LLM synthesis failed (non-fatal): {e}")
            return None

    def _generate_summary(self, text: str) -> str:
        """Generate a deterministic summary from the first chars of cleaned text.

        Truncates to ``SUMMARY_MAX_CHARS`` at the last sentence boundary if one
        falls in the second half of the window; otherwise trims to the char limit.

        Args:
            text: Cleaned document text.

        Returns:
            Summary string of up to ``SUMMARY_MAX_CHARS`` characters.
        """
        if not text:
            return ''
        # Take first 500 chars, ending at a sentence boundary if possible
        truncated = text[:SUMMARY_MAX_CHARS]
        last_period = truncated.rfind('.')
        if last_period > SUMMARY_MAX_CHARS // 2:
            return truncated[:last_period + 1]
        return truncated.strip()

    def _try_ocr(self, file_path: str, mime_type: str) -> str:
        """Attempt OCR text extraction for image-only PDFs and image files.

        Args:
            file_path: Absolute path to the file on disk.
            mime_type: MIME type of the file (``'application/pdf'`` or ``'image/*'``).

        Returns:
            Extracted text string, or empty string if OCR is unavailable or fails.
        """
        try:
            from services.ocr_service import ocr_pdf, ocr_image

            if mime_type == 'application/pdf':
                return ocr_pdf(file_path)
            elif mime_type and mime_type.startswith('image/'):
                return ocr_image(file_path)
            return ''
        except Exception as e:
            logger.warning(f'[DOC PROC] OCR fallback failed: {e}')
            return ''

    def _detect_language(self, text: str) -> str:
        """Detect the primary language of a document from its first 1000 characters.

        Args:
            text: Document text to sample for language detection.

        Returns:
            BCP-47 language code string (e.g. ``'en'``).  Falls back to ``'en'``
            if langdetect is unavailable or detection fails.
        """
        try:
            from langdetect import detect
            sample = text[:1000]
            return detect(sample)
        except Exception:
            return 'en'

    def _simhash(self, text: str, hash_bits: int = 64) -> str:
        """
        Compute a SimHash fingerprint for fuzzy dedup.
        Returns hex string.
        """
        if not text:
            return ''

        # Tokenize into shingles (3-word windows)
        words = text.lower().split()
        if len(words) < 3:
            return hashlib.md5(text.encode()).hexdigest()[:16]

        vector = [0] * hash_bits
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i + 3])
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
            for j in range(hash_bits):
                if h & (1 << j):
                    vector[j] += 1
                else:
                    vector[j] -= 1

        fingerprint = 0
        for j in range(hash_bits):
            if vector[j] > 0:
                fingerprint |= (1 << j)

        return format(fingerprint, f'0{hash_bits // 4}x')

    def _count_pages(self, file_path: str, mime_type: str) -> Optional[int]:
        """Count pages for paginated document types (PDF, PPTX).

        Args:
            file_path: Absolute path to the file on disk.
            mime_type: MIME type of the file.

        Returns:
            Integer page count, or ``None`` for unsupported MIME types or on error.
        """
        if mime_type == 'application/pdf':
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    return len(pdf.pages)
            except Exception:
                pass
        elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                return len(prs.slides)
            except Exception:
                pass
        return None

