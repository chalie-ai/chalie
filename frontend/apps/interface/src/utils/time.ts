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
 * Local calendar-day key "YYYY-MM-DD" for grouping rows by the day they fall on
 * (browser-local, matching parseDate's local-getter convention — the backend
 * already localizes the source strings). Stable, sortable, locale-independent.
 */
export function localDayKey(d: Date): string {
  const pad = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
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
