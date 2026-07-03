/**
 * Chat-bubble timestamp in the same "%d %b %H:%M" shape the backend emits
 * (locale_service.CHAT_TIMESTAMP_FMT). History rows already carry that string;
 * this formatter covers live-sent messages, which have no backend echo yet, so
 * the browser's local zone stands in for the server-configured one until the
 * next history reload. "" on parse error.
 */
export function chatTimestamp(iso?: string): string {
  const d = iso ? new Date(iso) : new Date();
  if (Number.isNaN(d.getTime())) return '';
  const day = String(d.getDate()).padStart(2, '0');
  const month = d.toLocaleString('en', { month: 'short' });
  const time = d.toLocaleString('en', { hour: '2-digit', minute: '2-digit', hour12: false });
  return `${day} ${month} ${time}`;
}

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
