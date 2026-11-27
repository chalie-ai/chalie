/**
 * Browser store — feature specs for what each read of the tab list shows the
 * user: which tab the panel opens on, when it opens at all, and where the
 * selection goes. The tab list is the only announcement: a tab Chalie opened,
 * or a hand-off it asked, opens the panel only when the read that lists it
 * finds its conversation on screen, and each does so once.
 *
 * A hand-off is an ask: it reaches the page in the open-asks listing read on
 * every connect, or as an `ask_opened` frame mid-session, which re-reads the
 * tabs.
 *
 * Real Pinia, real stores, real socket dispatcher, real REST client. Only the
 * network is stood in for: `fetch` answers `GET /api/browser/tabs` and
 * `GET /api/asks/open` with whatever the test's server lists, as JSON, the
 * same way the backend does.
 *
 * The stores keep what the thread panel shows and the read counter at module
 * level, so each test loads a fresh module graph, as a page load would.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { flushPromises } from '@vue/test-utils';
import type { AskOrigin, WsPushEvent } from '@chalie/shared';
import type { BrowserTab } from '../api/browser';
import type { PanelView } from './browser';

// ── The network boundary: what the server's browser and open asks list right now.
let served: BrowserTab[] = [];
let servedAsks: object[] = [];

const mainChat: AskOrigin = { type: 'user', turn_id: 7, forked: false };
const thread: AskOrigin = { type: 'user', turn_id: 9, forked: true };
const onMainChat: PanelView = { turnId: null, type: 'user' };
const onThread: PanelView = { turnId: 9, type: 'user' };

function handoff(askId: string, tabId: string, origin: AskOrigin) {
  return {
    ask_id: askId,
    kind: 'handoff',
    asked_at: '2026-11-09T10:00:00Z',
    data: { tab_id: tabId, reason: 'Please sign in', origin },
  };
}

function usersTab(id: string): BrowserTab {
  return { id, url: `https://example.com/${id}`, title: id, opener: 'user', origin: null, certificate_error: null };
}

function chaliesTab(id: string, origin: AskOrigin): BrowserTab {
  return { id, url: `https://example.com/${id}`, title: id, opener: 'chalie', origin, certificate_error: null };
}

/** A page load: fresh store modules whose reads go over the network stand-in. */
async function loadPage() {
  vi.resetModules();
  setActivePinia(createPinia());
  const { useBrowserStore, registerPanelView } = await import('./browser');
  const { useAsksStore } = await import('@chalie/shared');
  const { dispatchDrift } = await import('../utils/driftDispatcher');
  const store = useBrowserStore();
  const asks = useAsksStore();
  return {
    store,
    registerPanelView,
    /** What every socket connect does: restore the open asks, then read the tabs. */
    connect: async () => {
      await asks.refreshOpen();
      await store.refresh();
    },
    /** A hand-off asked mid-session: its frame arrives over the socket. */
    handOff: async (ask: object) => {
      servedAsks = [...servedAsks, ask];
      dispatchDrift({ type: 'ask_opened', ...ask } as WsPushEvent);
      await flushPromises();
    },
  };
}

const ok = (result: unknown) =>
  new Response(JSON.stringify({ success: true, result }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

beforeEach(() => {
  served = [];
  servedAsks = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if ((init?.method ?? 'GET') !== 'GET') return new Response(null, { status: 404 });
    if (String(input).endsWith('/api/browser/tabs')) return ok({ tabs: served });
    if (String(input).endsWith('/api/asks/open')) return ok(servedAsks);
    return new Response(null, { status: 404 });
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('browser store — what a read of the tab list shows', () => {
  it('opens the panel on a tab Chalie opens for the thread the user has open, over the tab they were viewing', async () => {
    const { store, registerPanelView } = await loadPage();
    registerPanelView(() => onThread);
    const mine = usersTab('mine');
    served = [mine];
    await store.refresh();
    store.openPanel(mine.id);

    const run = chaliesTab('run', thread);
    served = [mine, run];
    await store.refresh();

    expect([store.open, store.selectedTabId]).toEqual([true, run.id]);
  });

  it('opens the panel on a hand-off the main chat asked on the first read after a page load, not on one a thread off screen asked', async () => {
    const { store, connect } = await loadPage();
    const fromMain = chaliesTab('main-run', mainChat);
    const fromThread = chaliesTab('thread-run', thread);
    served = [usersTab('mine'), fromMain, fromThread];
    servedAsks = [handoff('h-main', fromMain.id, mainChat), handoff('h-thread', fromThread.id, thread)];

    await connect();

    expect([store.open, store.selectedTabId]).toEqual([true, fromMain.id]);
  });

  it('selects a tab Chalie opens for a thread off screen only while the panel is closed, and never opens the panel for it', async () => {
    const { store } = await loadPage();
    const mine = usersTab('mine');
    served = [mine];
    await store.refresh();

    const first = chaliesTab('first', thread);
    served = [mine, first];
    await store.refresh();
    expect([store.open, store.selectedTabId]).toEqual([false, first.id]);

    store.openPanel(mine.id);
    served = [mine, first, chaliesTab('second', thread)];
    await store.refresh();
    expect([store.open, store.selectedTabId]).toEqual([true, mine.id]);
  });

  it('opens nothing on the first read after a page load for a tab Chalie already had open for the main chat', async () => {
    const { store } = await loadPage();
    const mine = usersTab('mine');
    served = [mine, chaliesTab('run', mainChat)];

    await store.refresh();

    expect([store.open, store.selectedTabId]).toEqual([false, mine.id]);
  });

  it('opens the panel once for a tab Chalie opens; once the user closes it, the next read leaves it closed', async () => {
    const { store } = await loadPage();
    await store.refresh();
    const run = chaliesTab('run', mainChat);
    served = [run];
    await store.refresh();
    expect(store.open).toBe(true);

    store.closePanel();
    await store.refresh();

    expect([store.open, store.selectedTabId]).toEqual([false, run.id]);
  });

  it('opening the panel from the top bar shows a tab waiting on the user over one that is not, but keeps a waiting tab already selected', async () => {
    const { store, connect, handOff } = await loadPage();
    const mine = usersTab('mine');
    const first = chaliesTab('first', thread);
    served = [mine, first];
    await connect();
    await handOff(handoff('h-first', first.id, thread));
    expect([store.open, store.selectedTabId]).toEqual([false, mine.id]);

    store.openPanel();
    expect([store.open, store.selectedTabId]).toEqual([true, first.id]);

    store.closePanel();
    const second = chaliesTab('second', thread);
    served = [...served, second];
    await handOff(handoff('h-second', second.id, thread));
    expect(store.selectedTabId).toBe(second.id);
    store.openPanel();
    expect([store.open, store.selectedTabId]).toEqual([true, second.id]);
  });

  it('a hand-off the user closed the panel on stays on the dock and does not reopen the panel when the list is read again', async () => {
    const { store, connect, handOff } = await loadPage();
    served = [usersTab('mine'), chaliesTab('run', mainChat)];
    await connect();
    await handOff(handoff('h-run', 'run', mainChat));
    expect([store.open, store.selectedTabId]).toEqual([true, 'run']);

    store.closePanel();
    await connect();

    expect(store.open).toBe(false);
    expect(store.awayFrom(onMainChat).map((row) => [row.ask_id, row.data.tab_id])).toEqual([['h-run', 'run']]);
  });

  it('when the tab the user is viewing closes, the selection moves to the tab waiting on them, not the first tab', async () => {
    const { store, connect } = await loadPage();
    const mine = usersTab('mine');
    const viewing = usersTab('viewing');
    const waiting = chaliesTab('waiting', thread);
    served = [mine, viewing, waiting];
    servedAsks = [handoff('h-waiting', waiting.id, thread)];
    await connect();
    store.openPanel(viewing.id);

    served = [mine, waiting];
    await store.refresh();

    expect([store.open, store.selectedTabId]).toEqual([true, waiting.id]);
  });
});
