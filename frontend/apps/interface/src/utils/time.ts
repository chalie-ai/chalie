/**
 * Human cadence label for a scheduler cron triple — `day`/`hour`/`minute` mirror
 * the backend's `cron_dom`/`cron_hour`/`cron_minute` (`null` = "every" on that
 * field). Only four shapes are legal — the coarse-to-fine "every-prefix" rule
 * enforced by `validate_cron` (see services/cron_schedule.py) — so these four
 * branches are exhaustive.
 */
export function formatCadence(
  day: number | null,
  hour: number | null,
  minute: number | null,
): string {
  const pad = (n: number): string => String(n).padStart(2, '0');
  if (day == null && hour == null && minute == null) return 'Every minute';
  if (day == null && hour == null) return `Every hour at :${pad(minute as number)}`;
  if (day == null) return `Every day at ${pad(hour as number)}:${pad(minute as number)}`;
  return `Monthly on day ${day} at ${pad(hour as number)}:${pad(minute as number)}`;
}

/**
 * Whether a scheduled item's cron fields encode a recurring cadence, as
 * opposed to a one-time item that fires once at `start_at` and never again.
 *
 * The backend's cron model (`services/cron_schedule.py::validate_cron`) only
 * permits four "every-prefix" shapes — EEE/EEF/EFF/FFF (E = "every"/null,
 * F = fixed) — and a one-time create leaves every field unset, landing on
 * EEE (all three null): the same on-disk shape `formatCadence` reads as
 * "every minute". Pinning ANY field (EEF/EFF/FFF) is only reachable through
 * an explicit recurring request, so "at least one field is non-null" is the
 * unambiguous, evidence-backed discriminator over the legal shape space —
 * not a heuristic. Call this BEFORE `formatCadence` and only invoke that
 * formatter when this returns true.
 */
export function isRecurringCadence(
  day: number | null,
  hour: number | null,
  minute: number | null,
): boolean {
  return day != null || hour != null || minute != null;
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
