"""
List Card Service — generates inline HTML cards for list operations
and publishes them via OutputService.enqueue_card().
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

TOOL_NAME = "list"
ACCENT = "#8A5CFF"
BG = "rgba(138,92,255,0.06)"
MAX_VISIBLE_ITEMS = 15
CHECK_COLOR = "#34d399"
LOG_PREFIX = "[LIST CARD]"


class ListCardService:
    """Generates and emits list-related cards to the conversation spine."""

    # ─────────────────────────────────────────────────────────────────────────
    # Public emit methods
    # ─────────────────────────────────────────────────────────────────────────

    def emit_add_card(self, topic: str, list_name: str, items: list, skipped: int) -> None:
        """Emit a card showing items added to a list."""
        try:
            html = self._build_add_html(list_name, items, skipped)
            self._emit(topic, html, "add", f"Added to {list_name}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_add_card failed: {e}")

    def emit_remove_card(self, topic: str, list_name: str, items: list) -> None:
        """Emit a card showing items removed from a list."""
        try:
            html = self._build_remove_html(list_name, items)
            self._emit(topic, html, "remove", f"Removed from {list_name}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_remove_card failed: {e}")

    def emit_check_card(self, topic: str, list_name: str, items: list, is_check: bool) -> None:
        """Emit a card showing items checked or unchecked."""
        try:
            html = self._build_check_html(list_name, items, is_check)
            verb = "Checked" if is_check else "Unchecked"
            self._emit(topic, html, "check", f"{verb} on {list_name}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_check_card failed: {e}")

    def emit_view_card(self, topic: str, list_name: str, items: list, checked_count: int, total_count: int) -> None:
        """Emit a rich view card for a list with progress bar and checkboxes."""
        try:
            html = self._build_view_html(list_name, items, checked_count, total_count)
            self._emit(topic, html, "view", list_name)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_view_card failed: {e}")

    def emit_create_card(self, topic: str, list_name: str) -> None:
        """Emit a card confirming list creation."""
        try:
            html = self._build_create_html(list_name)
            self._emit(topic, html, "create", "List Created")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_create_card failed: {e}")

    def emit_delete_card(self, topic: str, list_name: str) -> None:
        """Emit a card confirming list deletion."""
        try:
            html = self._build_delete_html(list_name)
            self._emit(topic, html, "delete", "List Deleted")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_delete_card failed: {e}")

    def emit_clear_card(self, topic: str, list_name: str, count: int) -> None:
        """Emit a card confirming list was cleared."""
        try:
            html = self._build_clear_html(list_name, count)
            self._emit(topic, html, "clear", f"Cleared {list_name}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_clear_card failed: {e}")

    def emit_rename_card(self, topic: str, old_name: str, new_name: str) -> None:
        """Emit a card showing a list was renamed."""
        try:
            html = self._build_rename_html(old_name, new_name)
            self._emit(topic, html, "rename", "List Renamed")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_rename_card failed: {e}")

    def emit_list_all_card(self, topic: str, lists_summary: list) -> None:
        """Emit a card summarising all lists."""
        try:
            html = self._build_list_all_html(lists_summary)
            self._emit(topic, html, "list_all", "Your Lists")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_list_all_card failed: {e}")

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

    def _build_add_html(self, list_name: str, items: list, skipped: int) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:10px;">'
            f'Added to <span style="font-weight:600;color:rgba(234,230,242,0.50);">'
            f'{self._escape(list_name)}</span></div>'
        )
        rows = ""
        for item in items:
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                f'<span style="color:{ACCENT};font-size:14px;font-weight:600;flex-shrink:0;">+</span>'
                f'<span style="font-size:14px;color:#eae6f2;word-break:break-word;">'
                f'{self._escape(item)}</span>'
                f'</div>'
            )
        footer = ""
        if skipped > 0:
            footer = (
                f'<div style="font-size:12px;color:rgba(234,230,242,0.38);margin-top:8px;">'
                f'{skipped} already on list</div>'
            )
        return (
            f'<div data-card-type="add" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{rows}{footer}'
            f'</div>'
        )

    def _build_remove_html(self, list_name: str, items: list) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:10px;">'
            f'Removed from <span style="font-weight:600;color:rgba(234,230,242,0.50);">'
            f'{self._escape(list_name)}</span></div>'
        )
        rows = ""
        for item in items:
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                f'<span style="color:rgba(234,230,242,0.45);font-size:14px;flex-shrink:0;">&ndash;</span>'
                f'<span style="font-size:14px;color:rgba(234,230,242,0.45);'
                f'text-decoration:line-through;word-break:break-word;">'
                f'{self._escape(item)}</span>'
                f'</div>'
            )
        return (
            f'<div data-card-type="remove" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{rows}'
            f'</div>'
        )

    def _build_check_html(self, list_name: str, items: list, is_check: bool) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        verb = "Checked" if is_check else "Unchecked"
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:10px;">'
            f'{verb} on <span style="font-weight:600;color:rgba(234,230,242,0.50);">'
            f'{self._escape(list_name)}</span></div>'
        )
        rows = ""
        for item in items:
            checkbox = self._checkbox_html(is_check)
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                f'{checkbox}'
                f'<span style="font-size:14px;color:#eae6f2;word-break:break-word;">'
                f'{self._escape(item)}</span>'
                f'</div>'
            )
        return (
            f'<div data-card-type="check" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{rows}'
            f'</div>'
        )

    def _build_view_html(self, list_name: str, items: list, checked_count: int, total_count: int) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        pct = min(100, int(checked_count / total_count * 100)) if total_count > 0 else 0
        progress_bar = (
            f'<div style="height:2px;background:rgba(255,255,255,0.07);border-radius:1px;margin:8px 0 12px;">'
            f'<div style="height:2px;background:{ACCENT};border-radius:1px;width:{pct}%;"></div>'
            f'</div>'
        )

        header = (
            f'<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px;">'
            f'<span style="font-size:15px;font-weight:600;color:#eae6f2;">'
            f'{self._escape(list_name)}</span>'
            f'<span style="font-size:12px;color:rgba(234,230,242,0.38);font-variant-numeric:tabular-nums;">'
            f'{checked_count}/{total_count}</span>'
            f'</div>'
        )

        if not items:
            body = (
                f'<div style="text-align:center;font-size:14px;color:rgba(234,230,242,0.40);'
                f'padding:16px 0;">No items yet.</div>'
            )
            return (
                f'<div data-card-type="view" data-scope-id="{scope_id}"'
                f' data-tool="list" data-created-at="{created_at}"'
                f' style="padding:16px 18px;font-family:inherit;">'
                f'{header}{progress_bar}{body}'
                f'</div>'
            )

        shown = items[:MAX_VISIBLE_ITEMS]
        overflow = len(items) - MAX_VISIBLE_ITEMS
        rows = ""
        for i, item in enumerate(shown):
            border = "border-top:1px solid rgba(255,255,255,0.04);" if i > 0 else ""
            is_checked = item.get('checked', False)
            checkbox = self._checkbox_html(is_checked)
            dim = "opacity:0.7;" if is_checked else ""
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;{border}">'
                f'{checkbox}'
                f'<span style="font-size:14px;color:#eae6f2;word-break:break-word;{dim}">'
                f'{self._escape(item.get("content", ""))}</span>'
                f'</div>'
            )

        footer = ""
        if overflow > 0:
            footer = (
                f'<div style="font-size:12px;color:rgba(138,92,255,0.70);'
                f'margin-top:8px;text-align:right;">+{overflow} more items</div>'
            )

        return (
            f'<div data-card-type="view" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{header}{progress_bar}{rows}{footer}'
            f'</div>'
        )

    def _build_create_html(self, list_name: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        badge = (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(52,211,153,0.10);color:{CHECK_COLOR};">Created</span>'
        )
        return (
            f'<div data-card-type="create" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:15px;font-weight:600;color:#eae6f2;">'
            f'{self._escape(list_name)}</span>'
            f'{badge}'
            f'</div>'
        )

    def _build_delete_html(self, list_name: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        badge = (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(239,68,68,0.10);color:#f87171;">Deleted</span>'
        )
        return (
            f'<div data-card-type="delete" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:15px;color:rgba(234,230,242,0.45);text-decoration:line-through;">'
            f'{self._escape(list_name)}</span>'
            f'{badge}'
            f'</div>'
        )

    def _build_clear_html(self, list_name: str, count: int) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:6px;">List Cleared</div>'
        )
        body = (
            f'<div style="font-size:14px;color:#eae6f2;">'
            f'Cleared {count} item{"s" if count != 1 else ""} from '
            f'<span style="font-weight:600;">{self._escape(list_name)}</span></div>'
        )
        return (
            f'<div data-card-type="clear" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{body}'
            f'</div>'
        )

    def _build_rename_html(self, old_name: str, new_name: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:10px;">List Renamed</div>'
        )
        body = (
            f'<div style="font-size:14px;color:#eae6f2;display:flex;align-items:center;gap:6px;'
            f'flex-wrap:wrap;">'
            f'<span style="color:rgba(234,230,242,0.45);text-decoration:line-through;">'
            f'{self._escape(old_name)}</span>'
            f'<span style="color:rgba(234,230,242,0.38);">&nbsp;&rarr;&nbsp;</span>'
            f'<span style="font-weight:600;color:#eae6f2;">{self._escape(new_name)}</span>'
            f'</div>'
        )
        return (
            f'<div data-card-type="rename" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{body}'
            f'</div>'
        )

    def _build_list_all_html(self, lists_summary: list) -> str:
        scope_id = uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        label = (
            f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:12px;">Your Lists</div>'
        )

        if not lists_summary:
            return (
                f'<div data-card-type="list_all" data-scope-id="{scope_id}"'
                f' data-tool="list" data-created-at="{created_at}"'
                f' style="padding:16px 18px;font-family:inherit;">'
                f'{label}'
                f'<div style="text-align:center;font-size:14px;color:rgba(234,230,242,0.40);'
                f'padding:16px 0;">No lists yet.</div>'
                f'</div>'
            )

        rows = ""
        for i, lst in enumerate(lists_summary):
            border = "border-top:1px solid rgba(255,255,255,0.04);" if i > 0 else ""
            item_count = lst.get('item_count', 0)
            checked_count = lst.get('checked_count', 0)
            counts = (
                f'<span style="font-size:12px;color:rgba(234,230,242,0.38);'
                f'font-variant-numeric:tabular-nums;">'
                f'{checked_count}/{item_count}</span>'
            )
            rows += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:8px 0;{border}">'
                f'<span style="font-size:14px;color:#eae6f2;word-break:break-word;">'
                f'{self._escape(lst.get("name", ""))}</span>'
                f'{counts}'
                f'</div>'
            )

        return (
            f'<div data-card-type="list_all" data-scope-id="{scope_id}"'
            f' data-tool="list" data-created-at="{created_at}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label}{rows}'
            f'</div>'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _checkbox_html(self, checked: bool) -> str:
        """Render a CSS checkbox (not Unicode) for cross-font safety."""
        if checked:
            return (
                f'<div style="width:14px;height:14px;border-radius:3px;'
                f'background:{CHECK_COLOR};display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;">'
                f'<span style="font-size:10px;color:#06080e;line-height:1;">&#10003;</span>'
                f'</div>'
            )
        return (
            f'<div style="width:14px;height:14px;border-radius:3px;'
            f'border:1.5px solid rgba(234,230,242,0.25);flex-shrink:0;"></div>'
        )

    def _escape(self, text: str) -> str:
        """Basic HTML escaping — all user-origin text must pass through here."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
