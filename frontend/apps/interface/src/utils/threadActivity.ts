/**
 * Thread Activity — forked-thread state for the Activity drawer + Scheduler
 * dock, derived entirely from the rendered DOM contract (no client store).
 *
 * The spine-scoped list (`deriveThreadActivity` / `useThreadActivity`) reads
 * `data-turn-id`/`data-type`/`data-forked`/`data-gist`/`data-preview`/
 * `data-last-activity` stamped by TurnView.vue, plus `data-working`/
 * `data-done` stamped by `utils/turnDom.ts` — scoped to the registered spine
 * surface only (`SPINE_SURFACE_ID`), since that's the one surface that
 * renders every forked opener.
 *
 * `threadPhase` is the per-turn primitive the Scheduler dock uses instead —
 * scheduled turns aren't rendered on the spine, so it queries `turnDom`'s
 * global (document-wide) working/done state for an arbitrary turn_id/type
 * pair rather than scanning a surface.
 */
import { onBeforeUnmount, onMounted, ref } from 'vue';
import type { Ref } from 'vue';
import { ConfigType } from '@chalie/shared';
import { getSurfaceContainer, isTurnDone, isTurnWorking, SPINE_SURFACE_ID } from './turnDom';

/** True when a thread's last activity was within the 1-hour active window.
 *  Display/ordering only — no behavioral branch. Shared by SpineTurn's pill
 *  status and `deriveThreadActivity` below, so the rule lives in exactly
 *  one place. */
export function isThreadActive(lastActivityAt: string | null): boolean {
  if (!lastActivityAt) return false;
  // SQLite stores naive UTC ("YYYY-MM-DD HH:MM:SS"); mark it as UTC.
  const ts = new Date(`${lastActivityAt.replace(' ', 'T')}Z`).getTime();
  if (Number.isNaN(ts)) return false;
  return Date.now() - ts < 3_600_000;
}

/** A forked thread surfaced in the Activity drawer: its reply is streaming right
 *  now (`working`, pink) or has settled unseen (`done`, blue). */
export interface ThreadActivityItem {
  turn_id: number;
  label: string;
  snippet: string;
  kind: 'working' | 'done';
}

/** A forked turn's Activity phase: working → done → null (seen / no activity).
 *  Global (document-wide), NOT scoped to any one surface — for turns that
 *  never render on the spine (e.g. a Scheduler-dock row). */
export function threadPhase(turnId: number, type: string = ConfigType.USER): 'working' | 'done' | null {
  if (isTurnWorking(turnId, type)) return 'working';
  if (isTurnDone(turnId, type)) return 'done';
  return null;
}

/** Read the current forked-thread activity straight off the spine's rendered
 *  DOM. Synchronous, non-reactive — see `useThreadActivity` for a
 *  live-updating wrapper. */
export function deriveThreadActivity(type: string = ConfigType.USER): ThreadActivityItem[] {
  const container = getSurfaceContainer(SPINE_SURFACE_ID);
  if (!container) return [];

  const rows: { item: ThreadActivityItem; lastActivityAt: string }[] = [];
  const selector = `[data-turn-id][data-type="${type}"][data-forked]`;
  for (const el of container.querySelectorAll<HTMLElement>(selector)) {
    const working = el.hasAttribute('data-working');
    const done = el.hasAttribute('data-done');
    if (!working && !done) continue;

    const turnId = Number(el.getAttribute('data-turn-id'));
    if (Number.isNaN(turnId)) continue;

    const gist = el.getAttribute('data-gist');
    const preview = el.getAttribute('data-preview') ?? '';
    rows.push({
      item: {
        turn_id: turnId,
        label: gist || preview,
        snippet: preview,
        kind: working ? 'working' : 'done',
      },
      lastActivityAt: el.getAttribute('data-last-activity') ?? '',
    });
  }

  return rows
    .sort((a, b) => b.lastActivityAt.localeCompare(a.lastActivityAt))
    .map((r) => r.item);
}

/**
 * Reactive wrapper around `deriveThreadActivity` — the Activity drawer's
 * live-updating source. Re-derives on every 'turn-state-changed' (a
 * working/done flip) and 'turn-upserted' (a turn mounts/remounts, changing
 * which turns even qualify as forked) DOM event, matching the
 * listen-and-refresh pattern `composables/useDockBusy.ts` established.
 */
export function useThreadActivity(type: string = ConfigType.USER): Ref<ThreadActivityItem[]> {
  const items = ref<ThreadActivityItem[]>([]) as Ref<ThreadActivityItem[]>;

  function refresh(): void {
    items.value = deriveThreadActivity(type);
  }

  onMounted(() => {
    refresh();
    document.addEventListener('turn-state-changed', refresh);
    document.addEventListener('turn-upserted', refresh);
  });
  onBeforeUnmount(() => {
    document.removeEventListener('turn-state-changed', refresh);
    document.removeEventListener('turn-upserted', refresh);
  });

  return items;
}
