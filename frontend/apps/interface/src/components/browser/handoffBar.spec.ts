// @vitest-environment happy-dom
/**
 * The browser panel's hand-off bar — feature spec on the real panel.
 *
 * When Chalie hands a tab to the user (a sign-in, a captcha), the panel shows
 * a bar over that tab saying what is needed, with Done. Done answers the ask:
 * the bar leaves at once and the server is told the user is finished. If the
 * server could not take the answer, the user is told; if it says the hand-off
 * was already finished elsewhere, nothing is wrong and the bar stays gone.
 *
 * The real BrowserPanel on the real browser and asks stores and REST client,
 * with the chat app's toast registered for failed answers as the app does at
 * startup.
 * The network is the one stand-in: `fetch` serves the tab list and the open
 * asks and records each answer the panel sends, and the page view's stream
 * socket never connects.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { registerReplyFailureReporter, useAsksStore } from '@chalie/shared';
import BrowserPanel from './BrowserPanel.vue';
import { useBrowserStore } from '../../stores/browser';
import type { BrowserTab } from '../../api/browser';
import { showToast } from '../../utils/toast';

const backend = {
  tabs: [] as BrowserTab[],
  asks: [] as object[],
  sent: [] as { path: string; body: unknown }[],
  /** How the server answers Done: a status, or `offline` for a request that never reaches it. */
  replyAnswer: 204 as number | 'offline',
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const path = new URL(String(input), 'http://chalie.test').pathname;
  const method = init?.method ?? 'GET';
  if (method === 'GET' && path === '/api/browser/tabs') return json({ success: true, result: { tabs: backend.tabs } });
  if (method === 'GET' && path === '/api/asks/open') {
    return json({ success: true, result: backend.asks, pagination: { page: 1, limit: 50, total: backend.asks.length } });
  }
  if (method === 'POST' && path.startsWith('/api/asks/reply/')) {
    backend.sent.push({ path, body: init?.body ? JSON.parse(String(init.body)) : null });
    const answer = backend.replyAnswer;
    if (answer === 'offline') throw new TypeError('Failed to fetch');
    if (answer === 204 || answer === 404) {
      const askId = decodeURIComponent(path.slice('/api/asks/reply/'.length));
      backend.asks = backend.asks.filter((a) => (a as { ask_id: string }).ask_id !== askId);
    }
    if (answer === 204) return new Response(null, { status: 204 });
    if (answer === 404) return json({ error: 'That ask was already answered' }, 404);
    return json({ error: 'The browser is not responding' }, answer);
  }
  return json({ error: 'unrouted' }, 404);
}

/** The page view's stream socket: it never connects. */
class StreamSocket {
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  binaryType = 'blob';
  send(): void {}
  close(): void {}
}

const signInTab: BrowserTab = {
  id: 'login',
  url: 'https://accounts.example.com/signin',
  title: 'Sign in',
  opener: 'chalie',
  origin: { type: 'user', turn_id: 7, forked: false },
  certificate_error: null,
};

const signInAsk = {
  ask_id: 'hand-1',
  kind: 'handoff',
  asked_at: '2026-11-20T09:00:00+00:00',
  data: { tab_id: signInTab.id, reason: 'Please sign in', origin: { type: 'user', turn_id: 7, forked: false } },
};

let panel: VueWrapper | null = null;

/** A page load with the hand-off open: the open asks and the tabs are read, then the panel shows the tab. */
async function openOnHandoff(): Promise<VueWrapper> {
  backend.tabs = [signInTab];
  backend.asks = [signInAsk];
  await useAsksStore().refreshOpen();
  const browser = useBrowserStore();
  await browser.refresh();
  panel = mount(BrowserPanel, { attachTo: document.body });
  browser.openPanel(signInTab.id);
  await flushPromises();
  return panel;
}

async function pressDone(w: VueWrapper): Promise<void> {
  const done = w.findAll('.browser-panel__handoff button').find((b) => b.text() === 'Done');
  if (!done) throw new Error('no Done on the hand-off bar');
  await done.trigger('click');
  await flushPromises();
}

const toast = () => document.querySelector('.chalie-toast')?.textContent ?? null;

beforeEach(() => {
  backend.tabs = [];
  backend.asks = [];
  backend.sent = [];
  backend.replyAnswer = 204;
  setActivePinia(createPinia());
  registerReplyFailureReporter((message) => showToast(message));
  vi.stubGlobal('fetch', fakeBackend);
  vi.stubGlobal('WebSocket', StreamSocket);
});

afterEach(() => {
  panel?.unmount();
  panel = null;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('browser panel — hand-off bar', () => {
  it('Done tells the server the user is finished with that hand-off, and the bar leaves', async () => {
    const w = await openOnHandoff();
    expect(w.get('.browser-panel__handoff').text()).toContain('Please sign in');

    await pressDone(w);

    expect(backend.sent).toEqual([{ path: '/api/asks/reply/hand-1', body: { value: null } }]);
    expect(w.find('.browser-panel__handoff').exists()).toBe(false);
    expect(toast()).toBeNull();
  });

  it("a Done the server fails on tells the user why, and the bar comes back while the hand-off is still open", async () => {
    backend.replyAnswer = 500;
    const w = await openOnHandoff();

    await pressDone(w);

    expect(toast()).toContain('The browser is not responding');
    expect(w.find('.browser-panel__handoff').exists()).toBe(true);
  });

  it('a Done that never reaches the server tells the user it did not go through', async () => {
    backend.replyAnswer = 'offline';
    const w = await openOnHandoff();

    await pressDone(w);

    expect(toast()).toBe("Couldn't finish the hand-off");
  });

  it('a Done the server says came after the hand-off was already finished leaves the bar gone, with nothing to tell the user', async () => {
    backend.replyAnswer = 404;
    const w = await openOnHandoff();

    await pressDone(w);

    expect(toast()).toBeNull();
    expect(w.find('.browser-panel__handoff').exists()).toBe(false);
  });
});
