/**
 * Ask labels and wait time — the readable label for a gated action (and an
 * off-turn ask's origin) plus whole seconds since the ask parked.
 */
import { ConfigType } from '@chalie/shared';
import type { PermissionOrigin } from '../api/policies';

/** Human prefix for an ask shown away from its turn. */
export function originLabel(origin: PermissionOrigin): string {
  return origin.type === ConfigType.SCHEDULED ? 'Scheduled task' : 'Thread';
}

const ACTION_LABELS: Record<string, string> = {
  // email/calendar/contacts gate at the OUTER `pim` permission only — the inner
  // tools are INTERNAL on the backend and can never raise a permission_request.
  pim: 'Access Email, Calendar & Contacts',
  code_agent: 'Coding Agent',
  'browser.render': 'Read Webpage',
  'browser.interact': 'Interact with Webpage',
  'browser.screenshot': 'Screenshot Webpage',
  'browser.monitor': 'Monitor Webpage',
  'list.delete': 'Delete List',
  'memory.store': 'Store Memory',
  'memory.recall': 'Recall Memory',
  'memory.forget': 'Forget Memory',
  'memory.reflect': 'Reflect on Memory',
  'schedule.create': 'Create Schedule',
  'schedule.cancel': 'Cancel Schedule',
  'schedule.list': 'List Schedules',
  'schedule.search': 'Search Schedules',
  news: 'Fetch News',
  search: 'Web Search',
  weather: 'Check Weather',
  timer: 'Set Timer',
};

/** Readable label for an action_id; falls back to formatting the id (dots/underscores → spaces, title case). */
export function actionLabel(actionId: string): string {
  if (ACTION_LABELS[actionId]) return ACTION_LABELS[actionId];
  return actionId.replace(/[._]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Whole seconds since an ask parked — 0 when the stamp is unparseable or in
 * the future (a clock skew); the server stamp is what makes the age survive a
 * reload. */
export function waitedSeconds(askedAt: string, now: number): number {
  const parked = Date.parse(askedAt);
  return Number.isNaN(parked) ? 0 : Math.max(0, Math.floor((now - parked) / 1000));
}
