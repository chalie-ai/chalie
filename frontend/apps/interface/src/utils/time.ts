/**
 * Relative-time formatter for scheduler labels.
 * Ported from frontend/interface/utils.js — exact output strings preserved.
 */

/**
 * Convert an ISO date string to a short relative label.
 *
 * Output examples (matching legacy utils.js exactly):
 *   "now"        — less than 1 minute in the future
 *   "in 5m"      — minutes
 *   "in 2h"      — hours (< 24h)
 *   "tomorrow"   — exactly 1 day
 *   "in 3d"      — more than 1 day
 *   "overdue"    — date is in the past
 *   ""           — parse error
 */
export function relativeTime(isoStr: string): string {
  try {
    const due = new Date(isoStr);
    const now = Date.now();
    const diffMs = due.getTime() - now;
    if (diffMs < 0) return 'overdue';
    const mins = Math.round(diffMs / 60_000);
    if (mins < 1) return 'now';
    if (mins < 60) return `in ${mins}m`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `in ${hrs}h`;
    const days = Math.round(hrs / 24);
    if (days === 1) return 'tomorrow';
    return `in ${days}d`;
  } catch {
    return '';
  }
}
