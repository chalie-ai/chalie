"""
Moment Card Service — generates inline HTML cards for moment operations
and publishes them via OutputService.enqueue_card().
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TOOL_NAME = "moment"
ACCENT = "#8A5CFF"
BG = "rgba(138,92,255,0.06)"
LOG_PREFIX = "[MOMENT CARD]"


class MomentCardService:
    """Generates and emits moment-related cards to the conversation spine."""

    # ─────────────────────────────────────────────────────────────────────────
    # Public emit methods
    # ─────────────────────────────────────────────────────────────────────────

    def emit_moment_card(self, topic: str, moment: dict) -> None:
        """Emit a full moment card (used by recall skill and search results)."""
        try:
            html = self._build_moment_html(moment)
            title = moment.get("title") or "Moment"
            self._emit(topic, html, "moment", title)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_moment_card failed: {e}")

    def emit_moment_list_card(self, topic: str, moments: list) -> None:
        """Emit a summary card listing multiple moments."""
        try:
            html = self._build_moment_list_html(moments)
            self._emit(topic, html, "moment_list", "Your Moments")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_moment_list_card failed: {e}")

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

    def _build_moment_html(self, moment: dict) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        title = self._escape(moment.get("title") or "Moment")
        message_text = self._escape(moment.get("message_text") or "")
        summary = self._escape(moment.get("summary") or "")
        gists = moment.get("gists") or []
        pinned_at = moment.get("pinned_at")

        # Format timestamp
        pinned_str = ""
        if pinned_at:
            try:
                if isinstance(pinned_at, str):
                    pinned_at = datetime.fromisoformat(pinned_at)
                pinned_str = pinned_at.strftime("%d %b %Y, %H:%M")
            except Exception:
                pinned_str = str(pinned_at)

        # 1. Title
        title_html = (
            f'<div style="font-size:15px;font-weight:600;color:#eae6f2;'
            f'margin-bottom:10px;">{title}</div>'
        )

        # 2. Pinned message (quoted block with left violet border)
        message_html = (
            f'<div style="border-left:2px solid {ACCENT};padding-left:12px;'
            f'margin-bottom:10px;font-size:14px;color:#eae6f2;'
            f'line-height:1.6;">{message_text}</div>'
        )

        # 3. Summary
        summary_html = ""
        if summary:
            summary_html = (
                f'<div style="font-size:13px;color:rgba(234,230,242,0.58);'
                f'margin-bottom:10px;line-height:1.5;">{summary}</div>'
            )

        # 4. Gists (bullet points)
        gists_html = ""
        if gists:
            gist_items = ""
            for gist in gists[:4]:
                gist_items += (
                    f'<div style="display:flex;gap:6px;padding:2px 0;">'
                    f'<span style="color:rgba(234,230,242,0.30);flex-shrink:0;">&bull;</span>'
                    f'<span style="font-size:13px;color:rgba(234,230,242,0.45);'
                    f'line-height:1.5;">{self._escape(gist)}</span>'
                    f'</div>'
                )
            gists_html = f'<div style="margin-bottom:10px;">{gist_items}</div>'

        # 5. Pinned time footer
        footer_html = ""
        if pinned_str:
            footer_html = (
                f'<div style="font-size:12px;color:rgba(234,230,242,0.30);'
                f'margin-top:4px;">Pinned {pinned_str}</div>'
            )

        return (
            f'<div data-card-type="moment" data-scope-id="{scope_id}"'
            f' data-tool="moment" data-created-at="{created_at}"'
            f' data-moment-id="{self._escape(moment.get("id", ""))}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{title_html}{message_html}{summary_html}{gists_html}{footer_html}'
            f'</div>'
        )

    def _build_moment_list_html(self, moments: list) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:12px;">Your Moments</div>'
        )

        if not moments:
            return (
                f'<div data-card-type="moment_list" data-scope-id="{scope_id}"'
                f' data-tool="moment" data-created-at="{created_at}"'
                f' style="padding:16px 18px;font-family:inherit;">'
                f'{label}'
                f'<div style="text-align:center;font-size:14px;color:rgba(234,230,242,0.40);'
                f'padding:16px 0;">No moments yet.</div>'
                f'</div>'
            )

        rows = ""
        for i, m in enumerate(moments):
            border = "border-top:1px solid rgba(255,255,255,0.04);" if i > 0 else ""
            title = self._escape(m.get("title") or "Untitled")
            summary = self._escape(m.get("summary") or m.get("message_text", "")[:80])
            rows += (
                f'<div style="padding:8px 0;{border}">'
                f'<div style="font-size:14px;font-weight:500;color:#eae6f2;'
                f'margin-bottom:2px;">{title}</div>'
                f'<div style="font-size:13px;color:rgba(234,230,242,0.45);'
                f'line-height:1.4;">{summary}</div>'
                f'</div>'
            )

        return (
            f'<div data-card-type="moment_list" data-scope-id="{scope_id}"'
            f' data-tool="moment" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{rows}'
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
