"""
Document Card Service — Generates inline HTML cards for document operations
and publishes them via OutputService.enqueue_card().

Pattern: ListCardService — cyan accent (#00F0FF) for document cards.
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TOOL_NAME = "document"
ACCENT = "#00F0FF"
BG = "rgba(0,240,255,0.06)"
LOG_PREFIX = "[DOC CARD]"
MAX_SNIPPET_LENGTH = 200


class DocumentCardService:
    """Generates and emits document-related cards to the conversation spine."""

    # ─────────────────────────────────────────────────────────────────────────
    # Public emit methods
    # ─────────────────────────────────────────────────────────────────────────

    def emit_search_results_card(
        self,
        topic: str,
        query: str,
        results: list,
    ) -> None:
        """Emit source attribution card showing search results."""
        try:
            html = self._build_search_results_html(query, results)
            self._emit(topic, html, "search", "Document Search")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_search_results_card failed: {e}")

    def emit_upload_card(self, topic: str, doc_name: str, status: str) -> None:
        """Emit upload confirmation card."""
        try:
            html = self._build_upload_html(doc_name, status)
            self._emit(topic, html, "upload", "Document Uploaded")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_upload_card failed: {e}")

    def emit_view_card(
        self,
        topic: str,
        doc_metadata: dict,
        preview_chunks: list,
    ) -> None:
        """Emit document preview card."""
        try:
            html = self._build_view_html(doc_metadata, preview_chunks)
            self._emit(topic, html, "view", doc_metadata.get('original_name', 'Document'))
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_view_card failed: {e}")

    def emit_delete_card(self, topic: str, doc_name: str) -> None:
        """Emit deletion confirmation card."""
        try:
            html = self._build_delete_html(doc_name)
            self._emit(topic, html, "delete", "Document Deleted")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_delete_card failed: {e}")

    def emit_restore_card(self, topic: str, doc_name: str) -> None:
        """Emit restore confirmation card."""
        try:
            html = self._build_restore_html(doc_name)
            self._emit(topic, html, "restore", "Document Restored")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_restore_card failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core emit
    # ─────────────────────────────────────────────────────────────────────────

    def _emit(self, topic: str, html: str, card_type: str, title: str) -> None:
        from services.output_service import OutputService

        scope_id = uuid.uuid4().hex[:8]
        card_data = {
            "html": html,
            "css": "",
            "scope_id": scope_id,
            "title": title,
            "accent_color": ACCENT,
            "background_color": BG,
            "tool_name": TOOL_NAME,
        }
        OutputService().enqueue_card(topic, card_data)
        logger.debug(f"{LOG_PREFIX} Emitted {card_type} card (scope={scope_id}, topic={topic})")

    # ─────────────────────────────────────────────────────────────────────────
    # HTML builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_search_results_html(self, query: str, results: list) -> str:
        """Build source attribution card for search results."""
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        header = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:12px;">'
            f'Sources for <span style="font-weight:600;color:rgba(234,230,242,0.50);">'
            f'&ldquo;{self._escape(query)}&rdquo;</span></div>'
        )

        if not results:
            body = (
                f'<div style="text-align:center;font-size:14px;color:rgba(234,230,242,0.40);'
                f'padding:16px 0;">No matching documents found.</div>'
            )
            return (
                f'<div data-card-type="search" data-scope-id="{scope_id}"'
                f' data-tool="document" data-created-at="{created_at}"'
                f' style="padding:16px 18px;font-family:inherit;">'
                f'{header}{body}</div>'
            )

        rows = ""
        for i, result in enumerate(results[:5]):
            border = "border-top:1px solid rgba(255,255,255,0.04);" if i > 0 else ""
            doc_name = self._escape(result.get('document_name', 'Unknown'))
            page = result.get('page_number')
            section = result.get('section_title')

            # Location info
            location_parts = []
            if page:
                location_parts.append(f"Page {page}")
            if section:
                location_parts.append(self._escape(section))
            location = ' · '.join(location_parts) if location_parts else ''

            # Snippet
            content = result.get('content', '')
            snippet = content[:MAX_SNIPPET_LENGTH]
            if len(content) > MAX_SNIPPET_LENGTH:
                snippet += '…'

            # Document type badge
            doc_type = result.get('document_type', '')
            type_badge = ''
            if doc_type and doc_type != 'document':
                type_badge = (
                    f'<span style="font-size:10px;font-weight:500;letter-spacing:0.3px;'
                    f'text-transform:uppercase;padding:1px 5px;border-radius:3px;'
                    f'background:rgba(0,240,255,0.08);color:{ACCENT};margin-left:6px;">'
                    f'{self._escape(doc_type)}</span>'
                )

            # Confidence indicator
            distance = result.get('distance')
            confidence_badge = ''
            if distance is not None:
                similarity = 1.0 - distance
                if similarity < 0.5:
                    confidence_badge = (
                        f'<span style="font-size:10px;color:rgba(234,230,242,0.30);'
                        f'margin-left:6px;">Weak match</span>'
                    )
                elif similarity < 0.8:
                    confidence_badge = (
                        f'<span style="font-size:10px;color:rgba(234,230,242,0.38);'
                        f'margin-left:6px;">Partial match</span>'
                    )

            # Upload date
            created = result.get('document_created_at')
            date_str = ''
            if created:
                try:
                    if hasattr(created, 'strftime'):
                        date_str = created.strftime('%b %d, %Y')
                    else:
                        date_str = str(created)[:10]
                except Exception:
                    pass

            rows += (
                f'<div style="padding:10px 0;{border}">'
                f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:4px;">'
                f'<span style="font-size:13px;font-weight:600;color:#eae6f2;">'
                f'{doc_name}</span>'
                f'{type_badge}{confidence_badge}'
                f'</div>'
                f'<div style="font-size:11px;color:rgba(234,230,242,0.38);margin-bottom:6px;">'
                f'{location}'
                f'{" · " + date_str if date_str and location else date_str}'
                f'</div>'
                f'<div style="font-size:13px;color:rgba(234,230,242,0.65);'
                f'border-left:2px solid {ACCENT};padding-left:10px;line-height:1.5;">'
                f'{self._escape(snippet)}</div>'
                f'</div>'
            )

        return (
            f'<div data-card-type="search" data-scope-id="{scope_id}"'
            f' data-tool="document" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{header}{rows}</div>'
        )

    def _build_upload_html(self, doc_name: str, status: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        status_color = "#34d399" if status == 'ready' else ACCENT
        status_label = {
            'pending': 'Queued',
            'processing': 'Processing',
            'ready': 'Ready',
            'failed': 'Failed',
        }.get(status, status.title())

        badge = (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba({self._hex_to_rgb(status_color)},0.10);color:{status_color};">'
            f'{status_label}</span>'
        )

        return (
            f'<div data-card-type="upload" data-scope-id="{scope_id}"'
            f' data-tool="document" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;display:flex;'
            f'align-items:center;gap:10px;">'
            f'<span style="font-size:15px;font-weight:600;color:#eae6f2;">'
            f'{self._escape(doc_name)}</span>'
            f'{badge}'
            f'</div>'
        )

    def _build_view_html(self, doc_metadata: dict, preview_chunks: list) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        name = self._escape(doc_metadata.get('original_name', 'Document'))
        doc_type = doc_metadata.get('extracted_metadata', {}).get('document_type', {}).get('value', '')
        chunk_count = doc_metadata.get('chunk_count', 0)
        page_count = doc_metadata.get('page_count')

        # Header
        header = (
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
            f'margin-bottom:8px;">'
            f'<span style="font-size:15px;font-weight:600;color:#eae6f2;">{name}</span>'
            f'<span style="font-size:12px;color:rgba(234,230,242,0.38);">'
            f'{chunk_count} chunks'
            f'{f" · {page_count} pages" if page_count else ""}'
            f'</span>'
            f'</div>'
        )

        # Type badge
        type_badge = ''
        if doc_type and doc_type != 'document':
            type_badge = (
                f'<div style="margin-bottom:10px;">'
                f'<span style="font-size:10px;font-weight:500;letter-spacing:0.3px;'
                f'text-transform:uppercase;padding:2px 6px;border-radius:3px;'
                f'background:rgba(0,240,255,0.08);color:{ACCENT};">'
                f'{self._escape(doc_type)}</span></div>'
            )

        # Preview chunks
        body = ''
        for chunk in preview_chunks[:3]:
            content = chunk.get('content', '')[:300]
            if len(chunk.get('content', '')) > 300:
                content += '…'
            page = chunk.get('page_number')
            page_label = f'<span style="font-size:10px;color:rgba(234,230,242,0.30);">p.{page}</span> ' if page else ''

            body += (
                f'<div style="margin-bottom:8px;padding:8px 10px;'
                f'border-left:2px solid rgba(0,240,255,0.20);'
                f'background:rgba(255,255,255,0.02);border-radius:0 4px 4px 0;">'
                f'{page_label}'
                f'<span style="font-size:13px;color:rgba(234,230,242,0.60);line-height:1.5;">'
                f'{self._escape(content)}</span></div>'
            )

        return (
            f'<div data-card-type="view" data-scope-id="{scope_id}"'
            f' data-tool="document" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{header}{type_badge}{body}</div>'
        )

    def _build_delete_html(self, doc_name: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        badge = (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(239,68,68,0.10);color:#f87171;">Deleted</span>'
        )

        return (
            f'<div data-card-type="delete" data-scope-id="{scope_id}"'
            f' data-tool="document" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;display:flex;'
            f'align-items:center;gap:10px;">'
            f'<span style="font-size:15px;color:rgba(234,230,242,0.45);'
            f'text-decoration:line-through;">{self._escape(doc_name)}</span>'
            f'{badge}'
            f'</div>'
        )

    def _build_restore_html(self, doc_name: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        badge = (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(52,211,153,0.10);color:#34d399;">Restored</span>'
        )

        return (
            f'<div data-card-type="restore" data-scope-id="{scope_id}"'
            f' data-tool="document" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;display:flex;'
            f'align-items:center;gap:10px;">'
            f'<span style="font-size:15px;font-weight:600;color:#eae6f2;">'
            f'{self._escape(doc_name)}</span>'
            f'{badge}'
            f'</div>'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _escape(self, text: str) -> str:
        """Basic HTML escaping."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to r,g,b string for rgba()."""
        h = hex_color.lstrip('#')
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
