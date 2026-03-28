"""
Document Classification Service — embedding-based category, project, and date inference.

Classifies documents by comparing their summary embedding against existing
category/project label embeddings (cosine similarity via dot product on
L2-normalised vectors from EmbeddingService).

No LLM calls. No fixed taxonomy. Categories grow organically as documents
are added. When no existing label scores above threshold, a new label is
derived from the document's extracted metadata (keyword-detected type,
key terms, or filename).
"""

import logging
import re
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Minimum cosine similarity to reuse an existing label.
# Below this, a new label is generated from metadata.
CATEGORY_THRESHOLD = 0.45
PROJECT_THRESHOLD = 0.50


class DocumentClassificationService:
    """Infer doc_category, doc_project, and doc_date for a document."""

    def __init__(self, db_service=None):
        self.db = db_service

    def classify_document(
        self,
        doc_id: str,
        summary: str,
        clean_text: str,
        metadata: dict,
        original_name: str,
        folder_context: str = '',
    ) -> Optional[Dict[str, Any]]:
        """
        Run embedding-based classification on a document.

        Returns {"category": ..., "project": ..., "date": ...} or None on failure.
        Skips documents with meta_locked=1.
        """
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service

        db = self.db or get_shared_db_service()
        doc_service = DocumentService(db)

        doc = doc_service.get_document(doc_id)
        if not doc:
            return None
        if doc.get('meta_locked'):
            logger.debug(f"[DOC CLASS] Skipping {doc_id} — meta_locked")
            return None

        try:
            existing = self._get_existing_groups(doc_service)

            # Build a rich text representation for embedding
            classify_text = self._build_classify_text(
                summary, metadata, original_name, folder_context,
            )

            category = self._match_or_create_label(
                classify_text,
                existing.get('categories', []),
                CATEGORY_THRESHOLD,
                fallback_label=self._infer_category(metadata, original_name),
            )
            project = self._match_or_create_label(
                classify_text,
                existing.get('projects', []),
                PROJECT_THRESHOLD,
                fallback_label=self._infer_project(metadata, folder_context),
            )

            # Date cascade: extracted dates → created_at
            doc_date = self._fallback_date(metadata, doc)

            doc_service.update_classification(
                doc_id,
                category=category,
                project=project or None,
                doc_date=doc_date or None,
            )

            logger.info(
                f"[DOC CLASS] {doc_id}: category={category}, "
                f"project={project}, date={doc_date}"
            )
            return {'category': category, 'project': project, 'date': doc_date}

        except Exception as e:
            logger.warning(f"[DOC CLASS] Classification failed for {doc_id} (non-fatal): {e}")
            return None

    # ─────────────────────────────────────────────
    # Embedding-based matching
    # ─────────────────────────────────────────────

    def _match_or_create_label(
        self,
        doc_text: str,
        existing_labels: List[str],
        threshold: float,
        fallback_label: str,
    ) -> str:
        """Match doc_text against existing labels by cosine similarity.

        Returns the best-matching existing label if above threshold,
        otherwise returns fallback_label.
        """
        if not existing_labels:
            return fallback_label

        from services.embedding_service import get_embedding_service
        emb_svc = get_embedding_service()

        doc_emb = emb_svc.generate_embedding_np(doc_text)
        label_embs = emb_svc.generate_embeddings_batch(existing_labels)

        best_sim = -1.0
        best_label = fallback_label
        for label, label_emb in zip(existing_labels, label_embs, strict=True):
            sim = float(np.dot(doc_emb, label_emb))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim >= threshold:
            return best_label
        return fallback_label

    # ─────────────────────────────────────────────
    # Text construction
    # ─────────────────────────────────────────────

    def _build_classify_text(
        self,
        summary: str,
        metadata: dict,
        original_name: str,
        folder_context: str,
    ) -> str:
        """Build a rich text blob for embedding from document signals."""
        parts = []
        if original_name:
            parts.append(original_name)
        if folder_context:
            parts.append(folder_context)
        if summary:
            parts.append(summary[:500])

        doc_type = metadata.get('document_type', {}).get('value', '')
        if doc_type and doc_type != 'document':
            parts.append(f"Type: {doc_type}")
        if metadata.get('companies'):
            names = [c['name'] for c in metadata['companies'][:5]]
            parts.append(f"Companies: {', '.join(names)}")
        if metadata.get('key_terms'):
            parts.append(f"Terms: {', '.join(metadata['key_terms'][:8])}")

        return ' — '.join(parts) if parts else 'document'

    # ─────────────────────────────────────────────
    # Fallback label generation (no LLM)
    # ─────────────────────────────────────────────

    def _infer_category(self, metadata: dict, original_name: str) -> str:
        """Derive a category label from metadata when no existing label matches."""
        # Prefer keyword-detected document type
        doc_type = metadata.get('document_type', {}).get('value', '')
        if doc_type and doc_type != 'document':
            return doc_type.title()

        # Fall back to filename hints
        name_lower = (original_name or '').lower()
        for hint, label in (
            ('invoice', 'Invoice'), ('receipt', 'Receipt'),
            ('contract', 'Contract'), ('warranty', 'Warranty'),
            ('manual', 'Manual'), ('policy', 'Policy'),
            ('agreement', 'Agreement'), ('report', 'Report'),
            ('letter', 'Letter'), ('memo', 'Memo'),
            ('resume', 'Resume'), ('cv', 'Resume'),
        ):
            if hint in name_lower:
                return label

        # Last resort: use the most prominent key term
        terms = metadata.get('key_terms', [])
        if terms:
            return terms[0].title()

        return 'Document'

    def _infer_project(self, metadata: dict, folder_context: str) -> Optional[str]:
        """Derive a project label from folder context or metadata."""
        # Watched folder label is a strong project signal
        if folder_context:
            # Extract label from "Watched folder: <label>" or subfolder path
            pattern = r'Watched folder:\s*(.+?)(?:\s*/\s*(.+))?$'
            match = re.match(pattern, folder_context)
            if match:
                subfolder = match.group(2)
                if subfolder and subfolder != '.':
                    return subfolder.strip().title()
                label = match.group(1).strip()
                if label and label != '/':
                    return label.title()

        # Companies can hint at a project
        companies = metadata.get('companies', [])
        if len(companies) == 1:
            return companies[0]['name']

        return None

    # ─────────────────────────────────────────────
    # Existing groups
    # ─────────────────────────────────────────────

    def _get_existing_groups(self, doc_service) -> dict:
        """Fetch existing category and project values for similarity matching."""
        categories = doc_service.get_classification_groups('doc_category')
        projects = doc_service.get_classification_groups('doc_project')
        return {
            'categories': [g['value'] for g in categories if g['value'] != 'Uncategorized'][:20],
            'projects': [g['value'] for g in projects if g['value'] != 'Uncategorized'][:20],
        }

    # ─────────────────────────────────────────────
    # Date handling (unchanged)
    # ─────────────────────────────────────────────

    def _fallback_date(self, metadata: dict, doc: dict) -> str:
        """Apply date cascade: extracted dates → document created_at."""
        dates = metadata.get('dates', [])
        if dates:
            for d in dates:
                val = d.get('value', '')
                normalized = self._normalize_date(val)
                if normalized:
                    return normalized

        created = doc.get('created_at', '')
        if created:
            return created[:10]

        return ''

    def _normalize_date(self, date_str: str) -> str:
        """Normalize a date string to YYYY-MM-DD format."""
        if not date_str:
            return ''
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]
        for fmt in (
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
        ):
            m = re.match(fmt, date_str)
            if m:
                a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
                if a > 12:
                    return f"{year}-{b:02d}-{a:02d}"
                return f"{year}-{a:02d}-{b:02d}"
        return ''
