// @vitest-environment happy-dom
/**
 * The input dock's off-screen asks — feature spec on the real dock.
 *
 * A subagent the chat started can ask for permission after the chat turn that
 * started it has ended: no live line shows that turn any more, so the ask
 * waits in the input dock. The spec loads the open asks the way the app does
 * on every connect, from the backend's listing, mounts the real OffScreenAsks
 * on the real asks, browser and session stores, and asserts what a person
 * reads and can press:
 *
 *   - an ask no live line shows gets one row naming its turn and what it asks,
 *     with Deny, Allow and Open;
 *   - each row says where its ask came from: the chat, a thread or a scheduled
 *     task — a browser hand-off's row too;
 *   - Allow and Deny each send their own answer for that ask alone, and its
 *     row goes;
 *   - an answer the server fails on tells the user why and brings its row
 *     back, since the ask still waits; one that never reaches the server says
 *     it did not go through; one the server says came after the ask was
 *     already answered elsewhere leaves the row gone, with nothing to tell;
 *   - Open shows the ask's turn in the thread panel and answers nothing;
 *   - an ask from a run no chat turn shows has no origin before its words; its
 *     row still offers Open when it is a hand-off, whose tab can be shown, and
 *     not when it is a permission ask, which has no turn to open.
 *
 * The network is the one stand-in: the spec serves the open-asks listing and
 * the browser's tab list, and reads each answer the dock sent. The chat app's
 * toast is registered for failed answers as the app does at startup.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import OffScreenAsks from './OffScreenAsks.vue';
import { registerReplyFailureReporter, useAsksStore } from '@chalie/shared';
import type { AskOrigin } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { useBrowserStore } from '../../stores/browser';
import type { BrowserTab } from '../../api/browser';
import { showToast } from '../../utils/toast';

/** The chat turn that started the subagent; it ended before the ask came. */
const CHAT_TURN = 7;
const ON_CHAT: AskOrigin = { type: 'user', turn_id: CHAT_TURN, forked: false };

const backend = {
  open: [] as object[],
  tabs: [] as BrowserTab[],
  answers: [] as { path: string; body: unknown }[],
  /** How the server answers each answer: a status, or `offline` for a request that never reaches it. */
  replyAnswer: 204 as number | 'offline',
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname === '/api/asks/open' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: backend.open, pagination: { page: 1, limit: 50, total: backend.open.length } });
  }
  if (url.pathname === '/api/browser/tabs' && (init?.method ?? 'GET') === 'GET') {
    return json({ success: true, result: { tabs: backend.tabs } });
  }
  if (url.pathname.startsWith('/api/asks/reply/') && init?.method === 'POST') {
    backend.answers.push({ path: url.pathname, body: JSON.parse(String(init.body)) });
    const answer = backend.replyAnswer;
    if (answer === 'offline') throw new TypeError('Failed to fetch');
    if (answer !== 204 && answer !== 404) return json({ error: 'Could not answer the ask' }, answer);
    const answered = decodeURIComponent(url.pathname.slice('/api/asks/reply/'.length));
    backend.open = backend.open.filter((a) => (a as { ask_id: string }).ask_id !== answered);
    if (answer === 404) return json({ error: 'That ask was already answered' }, 404);
    return new Response(null, { status: 204 });
  }
  return json({ error: 'unrouted' }, 404);
}

/** A subagent's ask on the turn that started it — the chat turn unless given — as the backend lists it. */
function ask(ask_id: string, action_id: string, summary: string, origin: AskOrigin | null = ON_CHAT) {
  return { ask_id, kind: 'permission', asked_at: new Date().toISOString(), data: { action_id, summary, origin } };
}

/** A hand-off of a browser tab Chalie opened, as the backend lists it. */
function handoff(ask_id: string, tab: BrowserTab, reason: string) {
  return { ask_id, kind: 'handoff', asked_at: new Date().toISOString(), data: { tab_id: tab.id, reason, origin: tab.origin } };
}

function chaliesTab(id: string, origin: AskOrigin | null): BrowserTab {
  return { id, url: 'https://example.com/login', title: 'Sign in', opener: 'chalie', origin, certificate_error: null };
}

let wrapper: VueWrapper | null = null;

/** The dock with the backend's open asks and tabs loaded, as on connect, the browser panel closed. */
async function mountDock(...asks: object[]): Promise<VueWrapper> {
  backend.open = asks;
  await useAsksStore().refreshOpen();
  const browser = useBrowserStore();
  await browser.refresh();
  browser.closePanel();
  wrapper = mount(OffScreenAsks);
  await flushPromises();
  return wrapper;
}

const rows = (w: VueWrapper) => w.findAll('.offscreen-asks > .activity');
const labels = (w: VueWrapper): string[] => rows(w).map((r) => r.find('.activity__label').text());
const origins = (w: VueWrapper): string[] => rows(w).map((r) => r.find('.activity__origin').text());
const rowNamed = (w: VueWrapper, label: string) => {
  const found = rows(w).find((r) => r.find('.activity__label').text() === label);
  if (!found) throw new Error(`no dock row for "${label}"`);
  return found;
};
const toast = () => document.querySelector('.chalie-toast')?.textContent ?? null;

beforeEach(() => {
  setActivePinia(createPinia());
  backend.open = [];
  backend.tabs = [];
  backend.answers = [];
  backend.replyAnswer = 204;
  registerReplyFailureReporter((message) => showToast(message));
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('an ask no live line shows', () => {
  it('gets one row naming its turn and what it asks, with Deny, Allow and Open', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));

    expect(rows(w)).toHaveLength(1);
    const row = rows(w)[0];
    expect(row.find('.activity__origin').text()).toBe(`Chat · #${CHAT_TURN} ·`);
    expect(row.find('.activity__label').text()).toBe('Read the ferry booking email');
    expect(row.find('.activity__ask').text()).toBe('Read Email');
    expect(row.text()).toContain('waiting for you');
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('says where each ask came from: the chat, a thread or a scheduled task', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script', { type: 'user', turn_id: 12, forked: true }),
      ask('r3', 'email.search', 'Check the timetable inbox', { type: 'scheduled', turn_id: 31, forked: false }),
    );

    expect(origins(w)).toEqual([`Chat · #${CHAT_TURN} ·`, 'Thread · #12 ·', 'Scheduled task · #31 ·']);
  });

  it('a browser hand-off the chat asked for says Chat once the person closes the browser panel', async () => {
    const run = chaliesTab('run', ON_CHAT);
    backend.tabs = [run];
    const w = await mountDock(handoff('h1', run, 'Please sign in'));

    expect(labels(w)).toEqual(['Please sign in']);
    expect(origins(w)).toEqual([`Chat · #${CHAT_TURN} ·`]);
  });

  it('Allow sends approval for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(labels(w)).toEqual(['Write the fare table script']);
  });

  it('Deny sends a refusal for that ask alone, and its row goes', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );

    await rowNamed(w, 'Write the fare table script').find('.activity__answer--deny').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r2', body: { value: false } }]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('an Allow the server fails on tells the user why, and its row comes back since the ask still waits', async () => {
    const w = await mountDock(
      ask('r1', 'email.read', 'Read the ferry booking email'),
      ask('r2', 'bash.modify_file', 'Write the fare table script'),
    );
    backend.replyAnswer = 500;

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(toast()).toBe('Could not answer the ask');
    expect(labels(w)).toEqual(['Write the fare table script', 'Read the ferry booking email']);
    expect(rowNamed(w, 'Read the ferry booking email').findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow', 'Open']);
  });

  it('a Deny the server fails on tells the user why, and its row comes back too', async () => {
    const w = await mountDock(ask('r1', 'email.send', 'Send the weekly digest'));
    backend.replyAnswer = 500;

    await rowNamed(w, 'Send the weekly digest').find('.activity__answer--deny').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: false } }]);
    expect(toast()).toBe('Could not answer the ask');
    expect(labels(w)).toEqual(['Send the weekly digest']);
  });

  it('an answer that never reaches the server tells the user it did not go through, and its row comes back', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    backend.replyAnswer = 'offline';

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(toast()).toBe("Couldn't send your answer");
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('an answer the server says came after the ask was already answered leaves the row gone, with nothing to tell the user', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    backend.replyAnswer = 404;

    await rowNamed(w, 'Read the ferry booking email').find('.activity__answer--allow').trigger('click');
    await flushPromises();

    expect(backend.answers).toEqual([{ path: '/api/asks/reply/r1', body: { value: true } }]);
    expect(toast()).toBeNull();
    expect(labels(w)).toEqual([]);
  });

  it('Open shows its turn in the thread panel and answers nothing', async () => {
    const w = await mountDock(ask('r1', 'email.read', 'Read the ferry booking email'));
    const session = useSessionStore();

    await rowNamed(w, 'Read the ferry booking email').find('.activity__stop').trigger('click');
    await flushPromises();

    expect([session.panelThreadId, session.panelType]).toEqual([CHAT_TURN, 'user']);
    expect(backend.answers).toEqual([]);
    expect(labels(w)).toEqual(['Read the ferry booking email']);
  });

  it('a hand-off from a run no chat turn shows has no origin before its words, and Open still shows its tab', async () => {
    const run = chaliesTab('run', null);
    backend.tabs = [run];
    const w = await mountDock(handoff('h1', run, 'Please solve the captcha'));
    const row = rowNamed(w, 'Please solve the captcha');

    expect(row.find('.activity__origin').exists()).toBe(false);
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Open']);

    await row.find('.activity__stop').trigger('click');
    const browser = useBrowserStore();
    expect([browser.open, browser.selectedTabId]).toEqual([true, run.id]);
    expect(backend.answers).toEqual([]);
  });

  it('a permission ask from a run no chat turn shows has no origin before its words and no Open, only Deny and Allow', async () => {
    const w = await mountDock(ask('r1', 'email.send', 'Send the weekly digest', null));
    const row = rowNamed(w, 'Send the weekly digest');

    expect(row.find('.activity__origin').exists()).toBe(false);
    expect(row.findAll('button').map((b) => b.text())).toEqual(['Deny', 'Allow']);
  });
});
