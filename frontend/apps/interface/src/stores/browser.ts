/**
 * Browser store — the server browser's open tabs and the browser panel that
 * shows one of them live.
 *
 * The tab list is read over REST only: `refresh()` runs on every WS connect,
 * on each `browser_tabs_changed` frame, and after every change made here.
 * Only the newest read applies, so a read that started before a change can
 * never undo it.
 *
 * A pending hand-off means Chalie gave a tab to the user (a login, a
 * captcha). Its `browser_handoff` frame opens the panel on that tab when the
 * turn that asked is on screen. The input dock lists, from the tab list
 * alone, every hand-off the panel is not showing, so one that arrived while
 * the socket was down, before a reload, or that the user closed the panel
 * on still waits there until the user is done with it.
 */
import { defineStore } from 'pinia';
import { HttpError, ConfigType } from '@chalie/shared';
import type { WsPushEvent } from '@chalie/shared';
import { browser as browserApi } from '../api';
import type { BrowserHistoryAction, BrowserTab } from '../api/browser';
import type { PermissionOrigin } from '../api/policies';
import { showToast } from '../utils/toast';
import { readOrigin } from './permissions';

/** A pending hand-off the panel is not showing — one row on the input dock. */
export interface AwayHandoff {
  handoff_id: string;
  tab_id: string;
  reason: string;
  /** The instant the hand-off was asked for (ISO-8601) — the row's "waiting" clock ages from it. */
  asked_at: string;
  origin: PermissionOrigin;
}

/** The turn the thread panel shows, if any. */
export interface PanelView {
  turnId: number | null;
  type: string;
}

/** A main-chat turn is on screen; a scheduled task or a thread only while the thread panel shows it. */
export function originOnScreen(origin: PermissionOrigin, panel: PanelView): boolean {
  if (origin.type === ConfigType.USER && !origin.forked) return true;
  return panel.turnId === origin.turn_id && panel.type === origin.type;
}

function reportFailure(action: string, err: unknown): void {
  console.warn(`[browser] ${action} failed:`, err);
  showToast(err instanceof HttpError && err.error ? err.error : `Couldn't ${action}`);
}

/** A 404 on close or Done: the tab or hand-off is already gone, which is what was asked for. */
function alreadyGone(err: unknown): boolean {
  return err instanceof HttpError && err.status === 404;
}

/** Bumped by every read and every local change; a read whose number is stale is dropped. */
let refreshSeq = 0;

export const useBrowserStore = defineStore('browser', {
  state: () => ({
    open: false,
    tabs: [] as BrowserTab[],
    selectedTabId: null as string | null,
  }),

  getters: {
    selectedTab(state): BrowserTab | null {
      return state.tabs.find((t) => t.id === state.selectedTabId) ?? null;
    },
    /** Some tab is waiting for the user — the top bar's browser button carries the cue. */
    waiting(state): boolean {
      return state.tabs.some((t) => t.handoff != null);
    },
    /**
     * The dock's hand-off rows, given what the thread panel shows: every
     * pending hand-off while the browser panel is closed, and while it is
     * open those whose turn is not on screen.
     */
    awayFrom(state): (panel: PanelView) => AwayHandoff[] {
      return (panel) =>
        state.tabs.flatMap((t) => {
          const origin = readOrigin(t.handoff?.origin);
          if (!t.handoff || !origin || (state.open && originOnScreen(origin, panel))) return [];
          const { id, reason, asked_at } = t.handoff;
          return [{ handoff_id: id, tab_id: t.id, reason, asked_at, origin }];
        });
    },
  },

  actions: {
    /**
     * Re-read the tabs, moving the selection off a tab that closed.
     * Best-effort: a failed read leaves everything as it was; the next
     * connect or frame retries.
     */
    async refresh(): Promise<void> {
      const seq = ++refreshSeq;
      let tabs: BrowserTab[];
      try {
        tabs = await browserApi.tabs();
      } catch (err) {
        console.warn('[browser] tabs fetch failed:', err);
        return;
      }
      if (seq !== refreshSeq) return;
      this.tabs = tabs;
      this.reconcileSelection();
    },

    /** Keep the selected tab if it is still open, else prefer one waiting for the user, else the first. */
    reconcileSelection(): void {
      if (this.tabs.some((t) => t.id === this.selectedTabId)) return;
      this.selectedTabId = (this.tabs.find((t) => t.handoff) ?? this.tabs[0])?.id ?? null;
    },

    openPanel(tabId?: string): void {
      this.open = true;
      if (tabId) this.selectedTabId = tabId;
      else this.reconcileSelection();
      void this.refresh();
    },

    closePanel(): void {
      this.open = false;
    },

    select(tabId: string): void {
      this.selectedTabId = tabId;
    },

    /**
     * Open a user tab, on `url` when given, and show it. Null when the server
     * refused. The re-read that follows the change starts in the same tick,
     * so it drops any read already in flight; so does `closeTab`'s.
     */
    async createTab(url?: string): Promise<BrowserTab | null> {
      let tab: BrowserTab;
      try {
        tab = await browserApi.open(url);
      } catch (err) {
        reportFailure('open a tab', err);
        return null;
      }
      if (!this.tabs.some((t) => t.id === tab.id)) this.tabs.push(tab);
      this.selectedTabId = tab.id;
      void this.refresh();
      return tab;
    },

    async closeTab(tabId: string): Promise<void> {
      try {
        await browserApi.close(tabId);
      } catch (err) {
        if (!alreadyGone(err)) {
          reportFailure('close the tab', err);
          return;
        }
      }
      this.tabs = this.tabs.filter((t) => t.id !== tabId);
      this.reconcileSelection();
      void this.refresh();
    },

    async navigate(tabId: string, url: string): Promise<void> {
      try {
        await browserApi.navigate(tabId, url);
      } catch (err) {
        reportFailure('open that address', err);
      }
    },

    async goHistory(tabId: string, action: BrowserHistoryAction): Promise<void> {
      try {
        await browserApi.history(tabId, action);
      } catch (err) {
        reportFailure(action === 'reload' ? 'reload the page' : `go ${action}`, err);
      }
    },

    /** Mirror the live stream's address and title onto the tab strip. */
    patchTab(tabId: string, url: string, title: string): void {
      const tab = this.tabs.find((t) => t.id === tabId);
      if (!tab) return;
      tab.url = url;
      tab.title = title;
    },

    /**
     * The user pressed Done: clear the hand-off at once, then re-read the
     * tabs, which brings it back if the server did not take it. A read
     * already in flight was taken before Done, so it is dropped.
     */
    async finishHandoff(tabId: string): Promise<void> {
      const tab = this.tabs.find((t) => t.id === tabId);
      const handoff = tab?.handoff;
      if (!tab || !handoff) return;
      refreshSeq++;
      tab.handoff = null;
      try {
        await browserApi.finishHandoff(handoff.id);
      } catch (err) {
        if (!alreadyGone(err)) reportFailure('finish the hand-off', err);
      }
      void this.refresh();
    },

    /**
     * A `browser_handoff` frame. `panel` is what the thread panel shows now,
     * which decides whether the asking turn is on screen; if it is, or the
     * frame names no turn, the panel opens on the tab. Either way the tabs
     * are re-read, and the server lists the hand-off before it sends the
     * frame, so the Done bar and any dock row come from that read.
     */
    receiveHandoff(data: WsPushEvent, panel: PanelView): void {
      const frame = data as { tab?: unknown; origin?: unknown };
      if (typeof frame.tab !== 'string') {
        console.error('[browser] hand-off frame without a tab dropped:', data);
        return;
      }
      const origin = readOrigin(frame.origin);
      if (origin == null || originOnScreen(origin, panel)) {
        this.openPanel(frame.tab);
        return;
      }
      void this.refresh();
    },
  },
});
