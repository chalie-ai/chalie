"""
Document Processing Service — Text extraction, metadata extraction, chunking, and embedding.

Pipeline: extract text → detect language → extract metadata → generate summary →
          chunk text → embed chunks → store everything.

All metadata extraction is deterministic (regex, no LLM). Confidence scores
reflect pattern specificity × surrounding context quality.
"""

import hashlib
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

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
            # Step 1: Set processing status
            doc_service.update_status(doc_id, 'processing')

            import os
            file_path = os.path.join(DOCUMENTS_ROOT, doc['file_path'])

            # Step 2: Extract text
            text = self._extract_text(file_path, doc['mime_type'])

            if not text or not text.strip():
                doc_service.update_status(doc_id, 'failed', 'No text could be extracted from this document.')
                return False

            # Step 3: Clean text and detect language
            clean_text = self._normalize_text(text)
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
            from services.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            summary_embedding = embedding_service.generate_embedding(summary)
            fingerprint = self._simhash(clean_text)

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
            chunk_texts = [c['content'] for c in chunks]
            chunk_embeddings = self._generate_chunk_embeddings(embedding_service, chunk_texts)

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

            # Step 12: Mark awaiting confirmation — unblocks the frontend immediately
            doc_service.update_status(doc_id, 'awaiting_confirmation',
                                      chunk_count=len(chunk_records))
            logger.info(f"[DOC PROC] Document {doc_id} processed: {len(chunk_records)} chunks, "
                        f"awaiting confirmation")

            # Step 13: LLM synthesis (non-blocking enrichment)
            # Runs AFTER status change so the user isn't stuck waiting.
            # If it succeeds, metadata is updated with synthesis + key_facts.
            # If it fails, the confirmation card shows the truncated summary instead.
            try:
                synthesis_data = self._generate_llm_synthesis(
                    clean_text, metadata, doc['original_name'])
                if synthesis_data:
                    metadata['_synthesis'] = synthesis_data.get('synthesis', '')
                    metadata['_key_facts'] = synthesis_data.get('key_facts', [])
                    doc_service.update_extracted_metadata(
                        doc_id, metadata=metadata, summary=summary,
                    )
                    logger.info(f"[DOC PROC] LLM synthesis stored for {doc_id}")
            except Exception as synth_err:
                logger.warning(f"[DOC PROC] Post-confirmation synthesis failed (non-fatal): {synth_err}")

            return True

        except Exception as e:
            logger.error(f"[DOC PROC] Processing failed for {doc_id}: {e}", exc_info=True)
            doc_service.update_status(doc_id, 'failed', str(e)[:500])
            return False

    # ─────────────────────────────────────────────
    # Text extraction by format
    # ─────────────────────────────────────────────

    def _extract_text(self, file_path: str, mime_type: str) -> str:
        """Dispatch to format-specific text extractor."""
        extractors = {
            'application/pdf': self._extract_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': self._extract_pptx,
            'text/html': self._extract_html,
            'text/plain': self._extract_plain,
            'text/markdown': self._extract_plain,
        }

        extractor = extractors.get(mime_type)
        if extractor:
            return extractor(file_path)

        # Fallback: try plain text for code files and unknown types
        if mime_type and mime_type.startswith('text/'):
            return self._extract_plain(file_path)

        logger.warning(f"[DOC PROC] Unsupported mime type: {mime_type}")
        return self._extract_plain(file_path)

    def _extract_pdf(self, path: str) -> str:
        """Extract text from PDF using pdfplumber with table detection."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ''

                    # Try extracting tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            rows = []
                            for row in table:
                                cells = [str(cell or '').strip() for cell in row]
                                rows.append(' | '.join(cells))
                            page_text += '\n' + '\n'.join(rows)

                    if page_text.strip():
                        pages.append(f"[Page {i + 1}]\n{page_text.strip()}")

            return '\n\n'.join(pages)

        except Exception as e:
            logger.error(f"[DOC PROC] PDF extraction failed: {e}")
            return ''

    def _extract_docx(self, path: str) -> str:
        """Extract text from DOCX with paragraph and table support."""
        try:
            from docx import Document
            doc = Document(path)
            parts = []

            for element in doc.element.body:
                tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
                if tag == 'p':
                    # Paragraph
                    for para in doc.paragraphs:
                        if para._element == element:
                            text = para.text.strip()
                            if text:
                                # Mark headings
                                if para.style and para.style.name.startswith('Heading'):
                                    level = para.style.name.replace('Heading ', '').replace('Heading', '1')
                                    try:
                                        level = int(level)
                                    except ValueError:
                                        level = 1
                                    parts.append(f"{'#' * level} {text}")
                                else:
                                    parts.append(text)
                            break
                elif tag == 'tbl':
                    # Table
                    for table in doc.tables:
                        if table._element == element:
                            rows = []
                            for row in table.rows:
                                cells = [cell.text.strip() for cell in row.cells]
                                rows.append(' | '.join(cells))
                            parts.append('\n'.join(rows))
                            break

            return '\n\n'.join(parts)

        except Exception as e:
            logger.error(f"[DOC PROC] DOCX extraction failed: {e}")
            return ''

    def _extract_pptx(self, path: str) -> str:
        """Extract text from PowerPoint slides as sections."""
        try:
            from pptx import Presentation
            prs = Presentation(path)
            slides = []

            for i, slide in enumerate(prs.slides):
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            text = para.text.strip()
                            if text:
                                texts.append(text)
                if texts:
                    slides.append(f"[Slide {i + 1}]\n" + '\n'.join(texts))

            return '\n\n'.join(slides)

        except Exception as e:
            logger.error(f"[DOC PROC] PPTX extraction failed: {e}")
            return ''

    def _extract_html(self, path: str) -> str:
        """Extract content from HTML using trafilatura, BS4 fallback."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()

            try:
                import trafilatura
                text = trafilatura.extract(raw)
                if text and text.strip():
                    return text
            except Exception:
                pass

            # Fallback: BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            return soup.get_text(separator='\n', strip=True)

        except Exception as e:
            logger.error(f"[DOC PROC] HTML extraction failed: {e}")
            return ''

    def _extract_plain(self, path: str) -> str:
        """Read plain text file."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[DOC PROC] Plain text extraction failed: {e}")
            return ''

    # ─────────────────────────────────────────────
    # Metadata extraction (deterministic, no LLM)
    # ─────────────────────────────────────────────

    def _extract_metadata(self, text: str, language: str = 'en') -> dict:
        """Extract structured metadata using regex patterns."""
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
        """Extract dates from text using multiple format patterns."""
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
        """Extract expiration/validity dates from text."""
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
        """Extract company/organization names from text."""
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
        """Extract monetary values with currency detection."""
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
        """Extract policy numbers, serial numbers, order IDs, etc."""
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
        """Classify document type based on keyword density."""
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
        """Extract significant phrases via simple term frequency."""
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
        """Generate embeddings with adaptive batch sizing."""
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
        """Generate a deterministic summary (first ~500 chars of cleaned text)."""
        if not text:
            return ''
        # Take first 500 chars, ending at a sentence boundary if possible
        truncated = text[:SUMMARY_MAX_CHARS]
        last_period = truncated.rfind('.')
        if last_period > SUMMARY_MAX_CHARS // 2:
            return truncated[:last_period + 1]
        return truncated.strip()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for search: collapse whitespace, strip control chars."""
        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse multiple newlines (keep at most 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _detect_language(self, text: str) -> str:
        """Detect document language from first 1000 chars."""
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
        """Count pages for paginated document types."""
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

