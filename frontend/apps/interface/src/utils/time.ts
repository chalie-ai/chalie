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
