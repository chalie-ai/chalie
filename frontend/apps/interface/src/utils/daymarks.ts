// Date dividers for the conversation spine. Turn hosts are mounted imperatively
// (see utils/turnDom.ts), so there is no Vue `v-for` to interleave dividers
// into — instead each host carries a `data-day` stamp and this reconciler
// rebuilds the `.daymark` divider row at every day boundary after each upsert.
//
// Called synchronously from ConversationFeed's `turn-upserted` handler: the
// full remove-then-reinsert happens in one task with no intervening paint, so
// there is no flicker even though it runs on every live-turn growth signal.

import { localDayKey } from './time';

const DAYMARK_CLASS = 'daymark';

/** Human label for a `YYYY-MM-DD` key — "Today · 12 July" for the current day,
 *  else the full weekday, e.g. "Friday · 11 July" (matches the redesign mock).
 *  Month/weekday names come from the browser locale; the day-of-month leads to
 *  keep the "D Month" order stable regardless of locale default. */
function daymarkLabel(key: string, todayKey: string): string {
  const [y, m, d] = key.split('-').map(Number);
  const date = new Date(y, m - 1, d);
  const dayMonth = `${date.getDate()} ${date.toLocaleDateString(undefined, { month: 'long' })}`;
  if (key === todayKey) return `Today · ${dayMonth}`;
  return `${date.toLocaleDateString(undefined, { weekday: 'long' })} · ${dayMonth}`;
}

function makeDaymark(key: string, label: string): HTMLElement {
  const el = document.createElement('div');
  el.className = DAYMARK_CLASS;
  el.dataset.daymark = key;
  el.textContent = label;
  return el;
}

/**
 * Reconcile the `.daymark` dividers inside a turn container. Removes every
 * existing divider, then walks the turn hosts in DOM order (already sorted by
 * turn_id) and inserts one divider before the first host of each new local day.
 * Hosts with no `data-day` stamp join the group above them (no divider forced).
 */
export function syncDaymarks(container: HTMLElement): void {
  // The one date this module still computes locally, and deliberately the only
  // one: `data-day` is the backend's (it alone resolves the user's configured
  // timezone), while "which day is today" is asked of the browser here.
  // Normally identical — the heartbeat reports this same browser's
  // `Intl...timeZone` (composables/useHeartbeat.ts) and the backend formats
  // `day` through it. They can disagree only while the two are out of step:
  // before the first heartbeat lands, across a device timezone change, or on a
  // telemetry row left by another device. Accepted and bounded: the dividers
  // stay correct either way — only the `Today ·` label degrades to that day's
  // weekday, and the next heartbeat restores it. Closing it entirely would
  // mean the backend sending "today" as well.
  const todayKey = localDayKey(new Date());

  for (const mark of Array.from(container.querySelectorAll(`.${DAYMARK_CLASS}`))) {
    mark.remove();
  }

  let prevDay: string | null = null;
  for (const host of Array.from(container.children) as HTMLElement[]) {
    const day = host.dataset.day;
    if (!day) continue;
    if (day !== prevDay) {
      container.insertBefore(makeDaymark(day, daymarkLabel(day, todayKey)), host);
      prevDay = day;
    }
  }
}
