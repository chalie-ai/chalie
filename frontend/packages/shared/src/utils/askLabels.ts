/**
 * Ask labels and wait time — the readable label for a gated action (and an
 * off-turn ask's origin), the words of a sign-in code ask, plus whole seconds
 * since the ask parked.
 */
import { ConfigType } from '../config/configType';
import type { AskOrigin, CodeAskData } from '../stores/asks';

/** Where an ask shown away from its turn came from: a scheduled task, the main chat, or a thread. */
export function originLabel(origin: AskOrigin): string {
  if (origin.type === ConfigType.SCHEDULED) return 'Scheduled task';
  return origin.type === ConfigType.USER && !origin.forked ? 'Chat' : 'Thread';
}

const ACTION_LABELS: Record<string, string> = {
  'browser.render': 'Read Webpage',
  'browser.interact': 'Interact with Webpage',
  'browser.screenshot': 'Screenshot Webpage',
  'browser.monitor': 'Monitor Webpage',
  'calendar.list_events': 'List Calendar Events',
  'calendar.get_event': 'Read Calendar Event',
  'calendar.create_event': 'Create Calendar Event',
  'calendar.update_event': 'Update Calendar Event',
  'calendar.delete_event': 'Delete Calendar Event',
  'contacts.list': 'List Contacts',
  'contacts.get': 'Read Contact',
  'email.search': 'Search Email',
  'email.read': 'Read Email',
  'email.send': 'Send Email',
  'email.reply': 'Reply to Email',
  'email.forward': 'Forward Email',
  'email.draft': 'Draft Email',
  'email.download': 'Download Email Attachments',
  'email.manage': 'Manage Email',
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

/** What a sign-in code ask's row says, by the service whose sign-in waits on the code. */
export const CODE_ASK_WORDS: Record<
  CodeAskData['service'],
  { heading: string; field: string; button: string }
> = {
  unifi: {
    heading: 'UniFi sign-in needs your 2FA code',
    field: '6-digit code',
    button: 'Sign in',
  },
};

/** Whole seconds since an ask parked — 0 when the stamp is unparseable or in
 * the future (a clock skew); the server stamp is what makes the age survive a
 * reload. */
export function waitedSeconds(askedAt: string, now: number): number {
  const parked = Date.parse(askedAt);
  return Number.isNaN(parked) ? 0 : Math.max(0, Math.floor((now - parked) / 1000));
}
