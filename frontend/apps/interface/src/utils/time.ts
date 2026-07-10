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
  return formatDuration(secs);
}

/**
 * Format a non-negative whole-seconds duration as a clock string:
 * "MM:SS" under an hour, "h:MM:SS" at or beyond one hour. Minutes and
 * seconds are always zero-padded to two digits.
 */
export function formatDuration(totalSeconds: number): string {
  const pad = (n: number): string => String(n).padStart(2, '0');
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor(totalSeconds / 60) % 60;
  const s = totalSeconds % 60;
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

/** Short weekday labels, indexed by Date.getDay() (0 = Sun). */
export const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;

/** Short month labels, indexed by Date.getMonth() (0 = Jan). */
export const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
] as const;

/**
 * Parse a date/ISO string into a Date, or null when absent or unparseable.
 * (The backend already localizes these strings; no client-side offset math.)
 */
export function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Wall-clock time in 24h form: "HH:MM", or "HH:MM:SS" when opts.seconds is set.
 */
export function formatClock(d: Date, opts?: { seconds?: boolean }): string {
  return d.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    ...(opts?.seconds ? { second: '2-digit' as const } : {}),
    hour12: false,
  });
}
