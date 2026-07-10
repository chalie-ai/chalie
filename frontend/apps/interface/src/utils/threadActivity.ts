/**
 * True when a thread's last activity was within the 1-hour active window.
 * Display/ordering only — no behavioral branch. Shared by
 * `useConversationFeed` (legacy buffer-driven surfaces) and `SpineTurn` (the
 * DOM-contract pill) so the rule lives in exactly one place.
 */
export function isThreadActive(lastActivityAt: string | null): boolean {
  if (!lastActivityAt) return false;
  // SQLite stores naive UTC ("YYYY-MM-DD HH:MM:SS"); mark it as UTC.
  const ts = new Date(`${lastActivityAt.replace(' ', 'T')}Z`).getTime();
  if (Number.isNaN(ts)) return false;
  return Date.now() - ts < 3_600_000;
}
