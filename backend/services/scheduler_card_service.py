"""
Scheduler Card Service — generates inline HTML cards for scheduler events
and publishes them via OutputService.enqueue_card().
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER CARD]"


class SchedulerCardService:
    """Generates and emits scheduler-related cards to the conversation spine."""

    def emit_create_card(self, topic: str, item_data: dict) -> None:
        """Emit a 'created' card for a newly scheduled item."""
        try:
            html = self._build_create_html(item_data)
            self._emit(topic, html, "create", "Schedule Created")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_create_card failed: {e}")

    def emit_cancel_card(self, topic: str, item_data: dict) -> None:
        """Emit a 'cancelled' card for a cancelled item."""
        try:
            html = self._build_cancel_html(item_data)
            self._emit(topic, html, "cancel", "Schedule Cancelled")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_cancel_card failed: {e}")

    def emit_query_card(self, topic: str, items: list, time_range_label: str) -> None:
        """Emit a query list card for a set of scheduled items."""
        try:
            html = self._build_query_html(items, time_range_label)
            self._emit(topic, html, "query", time_range_label)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} emit_query_card failed: {e}")

    def _emit(self, topic: str, html: str, card_type: str, title: str) -> None:
        from services.output_service import OutputService

        scope_id = uuid.uuid4().hex[:8]
        card_data = {
            "html": html,
            "css": "",
            "scope_id": scope_id,
            "title": title,
            "accent_color": "#8A5CFF",
            "background_color": "rgba(138,92,255,0.06)",
            "tool_name": "scheduler",
        }
        OutputService().enqueue_card(topic, card_data)
        logger.debug(f"{LOG_PREFIX} Emitted {card_type} card (scope={scope_id}, topic={topic})")

    # ─────────────────────────────────────────────────────────────────────────
    # HTML builders
    # ─────────────────────────────────────────────────────────────────────────

    def _bell_icon(self, color: str = "#8A5CFF") -> str:
        """Return a 12×12 inline SVG bell icon."""
        return (
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="{color}"'
            f' style="flex-shrink:0;opacity:0.85;" aria-hidden="true">'
            f'<path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6V11c0-3.07-1.64-5.64-4.5-6.32V4'
            f'c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/>'
            f'</svg>'
        )

    def _build_create_html(self, item: dict) -> str:
        due_at = item.get("due_at")
        scope_id = uuid.uuid4().hex[:8]
        badges = self._type_badge(item.get("item_type", "reminder"))
        badges += self._status_badge("pending")
        rec = self._recurrence_badge(item.get("recurrence"))
        if rec:
            badges += rec
        return (
            f'<div data-card-type="create" data-scope-id="{scope_id}"'
            f' data-due-at="{self._iso(due_at)}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">'
            f'{self._bell_icon("#8A5CFF")}'
            f'{badges}'
            f'</div>'
            f'<div style="font-size:15px;font-weight:500;color:#eae6f2;line-height:1.5;margin-bottom:8px;">'
            f'{self._escape(item.get("message", ""))}'
            f'</div>'
            f'<div style="font-size:13px;color:rgba(234,230,242,0.58);">'
            f'{self._format_due_at(due_at)}'
            f'</div>'
            f'</div>'
        )

    def _build_cancel_html(self, item: dict) -> str:
        due_at = item.get("due_at")
        scope_id = uuid.uuid4().hex[:8]
        badges = self._type_badge(item.get("item_type", "reminder"))
        badges += self._status_badge("cancelled")
        rec = self._recurrence_badge(item.get("recurrence"))
        if rec:
            badges += rec
        return (
            f'<div data-card-type="cancel" data-scope-id="{scope_id}"'
            f' data-due-at="{self._iso(due_at)}"'
            f' style="padding:16px 18px;font-family:inherit;opacity:0.55;">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">'
            f'{self._bell_icon("rgba(234,230,242,0.30)")}'
            f'{badges}'
            f'</div>'
            f'<div style="font-size:15px;font-weight:500;color:#eae6f2;line-height:1.5;margin-bottom:8px;'
            f'text-decoration:line-through;">'
            f'{self._escape(item.get("message", ""))}'
            f'</div>'
            f'<div style="font-size:13px;color:rgba(234,230,242,0.58);">'
            f'{self._format_due_at(due_at)}'
            f'</div>'
            f'</div>'
        )

    def _build_query_html(self, items: list, time_range_label: str) -> str:
        scope_id = uuid.uuid4().hex[:8]
        label_html = (
            f'<div style="display:flex;align-items:center;gap:5px;font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.5px;'
            f'color:rgba(234,230,242,0.30);margin-bottom:12px;">'
            f'{self._bell_icon("rgba(234,230,242,0.30)")}'
            f'{self._escape(time_range_label)}</div>'
        )

        if not items:
            empty_msg = self._empty_state_text(time_range_label)
            return (
                f'<div data-card-type="query" data-scope-id="{scope_id}"'
                f' style="padding:16px 18px;font-family:inherit;">'
                f'{label_html}'
                f'<div style="text-align:center;font-size:14px;color:rgba(234,230,242,0.40);padding:16px 0;">'
                f'{empty_msg}</div>'
                f'</div>'
            )

        shown = items[:10]
        overflow = len(items) - 10
        rows = ""
        for i, item in enumerate(shown):
            due_at = item.get("due_at")
            is_fired = item.get("status") == "fired"
            border = "border-top:1px solid rgba(255,255,255,0.04);" if i > 0 else ""
            time_style = (
                "font-size:13px;white-space:nowrap;min-width:80px;"
                "text-decoration:line-through;color:rgba(234,230,242,0.25);"
                if is_fired else
                "font-size:13px;color:rgba(234,230,242,0.45);white-space:nowrap;min-width:80px;"
            )
            msg_style = (
                "font-size:14px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                "color:rgba(234,230,242,0.35);text-decoration:line-through;"
                if is_fired else
                "font-size:14px;color:#eae6f2;flex:1;overflow:hidden;"
                "text-overflow:ellipsis;white-space:nowrap;"
            )
            check_mark = (
                f'<span style="font-size:11px;color:rgba(138,92,255,0.55);flex-shrink:0;">&#10003;</span>'
                if is_fired else ""
            )
            rows += (
                f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;{border}">'
                f'<div style="{time_style}">{self._short_time(due_at)}</div>'
                f'<div style="{msg_style}">{self._escape(item.get("message", ""))}</div>'
                f'{check_mark}'
                f'<div style="flex-shrink:0;">{self._type_badge(item.get("item_type", "reminder"))}</div>'
                f'</div>'
            )

        footer = ""
        if overflow > 0:
            footer = (
                f'<div style="font-size:12px;color:rgba(138,92,255,0.70);'
                f'margin-top:8px;text-align:right;">+{overflow} more</div>'
            )

        return (
            f'<div data-card-type="query" data-scope-id="{scope_id}"'
            f' style="padding:16px 18px;font-family:inherit;">'
            f'{label_html}'
            f'{rows}'
            f'{footer}'
            f'</div>'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Badge helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _type_badge(self, item_type: str) -> str:
        color = "#00F0FF" if item_type == "prompt" else "#8A5CFF"
        label = "prompt" if item_type == "prompt" else "notification"
        return (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(255,255,255,0.06);color:{color};">{label}</span>'
        )

    def _status_badge(self, status: str) -> str:
        color = "rgba(234,230,242,0.30)" if status == "cancelled" else "#8A5CFF"
        return (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(255,255,255,0.06);color:{color};">{status}</span>'
        )

    def _recurrence_badge(self, recurrence: Optional[str]) -> str:
        if not recurrence:
            return ""
        label = self._format_recurrence(recurrence)
        return (
            f'<span style="font-size:11px;font-weight:500;letter-spacing:0.4px;'
            f'text-transform:uppercase;padding:2px 7px;border-radius:4px;'
            f'background:rgba(34,211,238,0.08);color:#22d3ee;">\u21bb {label}</span>'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Formatting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _format_recurrence(self, recurrence: str) -> str:
        mapping = {
            "daily": "Daily",
            "weekly": "Weekly",
            "monthly": "Monthly",
            "weekdays": "Weekdays",
            "hourly": "Hourly",
        }
        if recurrence in mapping:
            return mapping[recurrence]
        if recurrence.startswith("interval:"):
            try:
                mins = int(recurrence.split(":", 1)[1])
                if mins >= 60 and mins % 60 == 0:
                    hours = mins // 60
                    return f"Every {hours}h"
                return f"Every {mins}m"
            except (ValueError, IndexError):
                return recurrence
        return recurrence.capitalize()

    def _format_due_at(self, due_at) -> str:
        if not due_at:
            return ""
        try:
            if isinstance(due_at, str):
                due_at = datetime.fromisoformat(due_at)
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff = due_at - now
            # %-d works on Linux; use lstrip("0") for portability
            date_str = due_at.strftime("%d %b · %H:%M").lstrip("0")
            if 0 < diff.total_seconds() < 86400:
                hours = int(diff.total_seconds() // 3600)
                minutes = int((diff.total_seconds() % 3600) // 60)
                if hours > 0:
                    rel = f"in {hours}h {minutes}m" if minutes else f"in {hours}h"
                else:
                    rel = f"in {minutes}m"
                return f"{date_str} · {rel}"
            return date_str
        except Exception:
            return str(due_at)

    def _short_time(self, due_at) -> str:
        """Format a compact date+time for list rows."""
        if not due_at:
            return ""
        try:
            if isinstance(due_at, str):
                due_at = datetime.fromisoformat(due_at)
            if due_at.tzinfo is None:
                due_at = due_at.replace(tzinfo=timezone.utc)
            return due_at.strftime("%d %b %H:%M").lstrip("0")
        except Exception:
            return str(due_at)

    def _iso(self, due_at) -> str:
        if not due_at:
            return ""
        if isinstance(due_at, datetime):
            return due_at.isoformat()
        return str(due_at)

    def _empty_state_text(self, time_range_label: str) -> str:
        label = time_range_label.lower()
        if "today" in label:
            return "You're clear today."
        if "tomorrow" in label:
            return "Nothing coming up tomorrow."
        if "week" in label:
            return "Nothing coming up this week."
        if "hour" in label:
            return "Nothing in the next hour."
        if "soon" in label:
            return "Nothing coming up soon."
        return "No scheduled items."

    def _escape(self, text: str) -> str:
        """Basic HTML escaping."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
