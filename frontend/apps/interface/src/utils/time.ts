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

/**
 * Elapsed wall-clock time since an ISO start, for a delegate's live timer:
 * mm:ss under an hour, h:mm:ss beyond. `nowMs` is supplied by the caller (not
 * read from the clock) so a reactive tick drives the update and this stays a
 * pure, deterministic formatter.
 */
export function elapsedSince(isoStr: string, nowMs: number): string {
  const started = new Date(isoStr).getTime();
  if (!Number.isFinite(started)) return '00:00';
  const secs = Math.max(0, Math.floor((nowMs - started) / 1000));
  const pad = (n: number): string => String(n).padStart(2, '0');
  const h = Math.floor(secs / 3600);
  const m = Math.floor(secs / 60) % 60;
  const s = secs % 60;
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
