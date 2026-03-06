"""
Document Classification Service — LLM-driven category, project, and date inference.

Classifies documents into natural-language categories and projects by sending
the document summary, extracted metadata, and existing groups to an LLM.
Registered as cognitive job 'document-classification' in the Brain dashboard.
"""

import json
import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


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
        Run LLM classification on a document.

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
            result = self._call_llm(
                summary=summary,
                clean_text=clean_text,
                metadata=metadata,
                original_name=original_name,
                folder_context=folder_context,
                existing_groups=self._get_existing_groups(doc_service),
            )
            if not result:
                return None

            # Apply date cascade: LLM date → extracted dates → created_at
            doc_date = result.get('date') or ''
            if not doc_date or doc_date == 'unknown':
                doc_date = self._fallback_date(metadata, doc)

            category = result.get('category') or None
            project = result.get('project') or None
            if project and project.lower() in ('generic', 'none', 'unknown', 'n/a', ''):
                project = None

            doc_service.update_classification(
                doc_id,
                category=category,
                project=project,
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

    def _call_llm(
        self,
        summary: str,
        clean_text: str,
        metadata: dict,
        original_name: str,
        folder_context: str,
        existing_groups: dict,
    ) -> Optional[dict]:
        """Send classification request to the assigned LLM provider."""
        try:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service

            agent_cfg = ConfigService.resolve_agent_config('document-classification')
            prompt_template = ConfigService.get_agent_prompt('document-classification')
            if not prompt_template:
                logger.warning("[DOC CLASS] No classification prompt found")
                return None

            # Build metadata summary
            meta_lines = []
            doc_type = metadata.get('document_type', {}).get('value', '')
            if doc_type and doc_type != 'document':
                meta_lines.append(f"Detected type: {doc_type}")
            if metadata.get('companies'):
                meta_lines.append(
                    f"Companies: {', '.join(c['name'] for c in metadata['companies'][:5])}"
                )
            if metadata.get('dates'):
                meta_lines.append(
                    f"Dates found: {', '.join(d['value'] for d in metadata['dates'][:5])}"
                )
            if metadata.get('monetary_values'):
                money = [f"{v['currency']} {v['amount']}" for v in metadata['monetary_values'][:5]]
                meta_lines.append(f"Monetary values: {', '.join(money)}")
            if metadata.get('key_terms'):
                meta_lines.append(f"Key terms: {', '.join(metadata['key_terms'][:8])}")
            metadata_summary = '\n'.join(meta_lines) if meta_lines else 'No metadata extracted.'

            # Build existing groups context
            groups_text = ''
            if existing_groups.get('categories'):
                groups_text += f"Existing categories: {', '.join(existing_groups['categories'])}\n"
            if existing_groups.get('projects'):
                groups_text += f"Existing projects: {', '.join(existing_groups['projects'])}\n"

            # Truncate text for context
            truncated_text = (clean_text or '')[:2000]

            system_prompt = (
                prompt_template
                .replace('{{original_name}}', original_name or 'Unknown')
                .replace('{{summary}}', summary or '')
                .replace('{{metadata_summary}}', metadata_summary)
                .replace('{{folder_context}}', folder_context or 'None')
                .replace('{{existing_groups}}', groups_text or 'None yet — this is the first document.')
                .replace('{{clean_text}}', truncated_text)
            )

            llm = create_llm_service(agent_cfg)
            response = llm.send_message(system_prompt, "Classify this document.")

            result = json.loads(response.text)
            if 'category' in result:
                return result

            logger.warning("[DOC CLASS] Response missing 'category' key")
            return None

        except json.JSONDecodeError as e:
            logger.warning(f"[DOC CLASS] Failed to parse LLM JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"[DOC CLASS] LLM call failed: {e}")
            return None

    def _get_existing_groups(self, doc_service) -> dict:
        """Fetch existing category and project values for LLM context."""
        categories = doc_service.get_classification_groups('doc_category')
        projects = doc_service.get_classification_groups('doc_project')
        return {
            'categories': [g['value'] for g in categories if g['value'] != 'Uncategorized'][:20],
            'projects': [g['value'] for g in projects if g['value'] != 'Uncategorized'][:20],
        }

    def _fallback_date(self, metadata: dict, doc: dict) -> str:
        """Date cascade: extracted dates → file created_at."""
        # Try extracted dates
        dates = metadata.get('dates', [])
        if dates:
            # Pick the first well-formed date
            for d in dates:
                val = d.get('value', '')
                # Try to normalize to YYYY-MM-DD
                normalized = self._normalize_date(val)
                if normalized:
                    return normalized

        # Fall back to document created_at
        created = doc.get('created_at', '')
        if created:
            return created[:10]  # YYYY-MM-DD from ISO datetime

        return ''

    def _normalize_date(self, date_str: str) -> str:
        """Attempt to normalize a date string to YYYY-MM-DD."""
        if not date_str:
            return ''
        # Already in ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]
        # Common formats: DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY
        for fmt in (
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
        ):
            m = re.match(fmt, date_str)
            if m:
                a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
                # Assume DD/MM/YYYY if first > 12
                if a > 12:
                    return f"{year}-{b:02d}-{a:02d}"
                return f"{year}-{a:02d}-{b:02d}"
        return ''
