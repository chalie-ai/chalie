// Short relative-time label for scheduler UI ("now", "in 5m", "tomorrow",
// "overdue", "" on parse error).
export function relativeTime(isoStr: string): string {
  try {
    const diffMs = new Date(isoStr).getTime() - Date.now();
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
