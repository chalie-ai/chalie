/**
 * Asks store — feature specs for which of Chalie's questions wait on the
 * user. Two feeds fill one queue: the `ask_opened` / `ask_closed` frames off
 * the socket, and the open-asks listing read on every connect. The listing is
 * a snapshot, so a read must never bring back an ask that closed while it was
 * in flight; an answer leaves its row at once and posts the user's choice,
 * and a skipped question leaves at once and is dismissed.
 *
 * Real Pinia, real store, real socket dispatcher, real REST client. Only the
 * network is stood in for: `fetch` answers `GET /api/asks/open`,
 * `POST /api/asks/reply/<id>`, `POST /api/asks/dismiss/<id>` and
 * `GET /api/browser/tabs` the way the backend does (a 502 the way a proxy
 * does, with no body), and a test can hold an answer back to land a frame or a reply while
 * that request is in flight.
 *
 * The store keeps its in-flight bookkeeping at module level, so each test
 * loads a fresh module graph, as a page load would.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { flushPromises } from '@vue/test-utils';
import type { WsPushEvent } from '@chalie/shared';

// ── The network boundary: what the server holds open, and every request it saw.
interface Request {
  method: string;
  path: string;
  body: unknown;
}
let listing: unknown[] = [];
let replyStatus = 204;
let requests: Request[] = [];
/** What the app's toast was asked to tell the user about a failed answer. */
let told: string[] = [];
const held: Record<'open' | 'reply', Promise<void>[]> = { open: [], reply: [] };

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/** Hold the next answer on a route back until the returned release is called. */
function holdNext(route: 'open' | 'reply'): () => Promise<void> {
  let release!: () => void;
  held[route].push(new Promise<void>((resolve) => (release = resolve)));
  return async () => {
    release();
    await flushPromises();
  };
}

async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const path = String(input);
  const method = init?.method ?? 'GET';
  requests.push({ method, path, body: init?.body ? JSON.parse(String(init.body)) : null });
  if (method === 'GET' && path === '/api/asks/open') {
    await held.open.shift();
    return json(200, { success: true, result: listing, pagination: { page: 1, limit: 50, total: listing.length } });
  }
  if (method === 'POST' && (path.startsWith('/api/asks/reply/') || path.startsWith('/api/asks/dismiss/'))) {
    await held.reply.shift();
    if (replyStatus === 204 || replyStatus === 502) return new Response(null, { status: replyStatus });
    return json(replyStatus, { error: replyStatus === 404 ? 'That ask was already answered' : 'Server error' });
  }
  if (method === 'GET' && path === '/api/browser/tabs') {
    return json(200, { success: true, result: { tabs: [] } });
  }
  return new Response(null, { status: 404 });
}

const mainChat = { type: 'user', turn_id: 7, forked: false };

function permissionAsk(id: string, origin: unknown = mainChat) {
  return {
    ask_id: id,
    kind: 'permission',
    asked_at: '2026-11-20T09:00:00+00:00',
    data: { action_id: 'email.send', summary: `Send the ${id} email`, origin },
  };
}

function handoffAsk(id: string, tabId: string) {
  return {
    ask_id: id,
    kind: 'handoff',
    asked_at: '2026-11-20T09:00:00+00:00',
    data: { tab_id: tabId, reason: 'Please sign in', origin: mainChat },
  };
}

function questionAsk(id: string, data: object = {}) {
  return {
    ask_id: id,
    kind: 'question',
    asked_at: '2026-11-20T09:00:00+00:00',
    data: {
      question: 'Which day suits you?',
      answers: [
        { label: 'Tuesday', description: 'Morning only' },
        { label: 'Friday', description: '' },
      ],
      multi_select: false,
      origin: mainChat,
      ...data,
    },
  };
}

/** A sign-in that waits on the code the user types, for the service named. */
function codeAsk(id: string, service = 'unifi', origin: unknown = mainChat) {
  return {
    ask_id: id,
    kind: 'code',
    asked_at: '2026-11-20T09:00:00+00:00',
    data: { service, origin },
  };
}

const opened = (ask: object) => ({ type: 'ask_opened', ...ask }) as WsPushEvent;
const closed = (id: string) => ({ type: 'ask_closed', ask_id: id }) as WsPushEvent;

/** A page load: a fresh module graph; frames arrive through the real socket dispatcher. */
async function loadPage() {
  vi.resetModules();
  setActivePinia(createPinia());
  const { useAsksStore, registerReplyFailureReporter } = await import('@chalie/shared');
  registerReplyFailureReporter((message) => told.push(message));
  const { dispatchDrift } = await import('../utils/driftDispatcher');
  return { asks: useAsksStore(), frame: dispatchDrift };
}

const waiting = (asks: { queue: { ask_id: string }[] }) => asks.queue.map((a) => a.ask_id);
const opensRead = () => requests.filter((r) => r.path === '/api/asks/open').length;

beforeEach(() => {
  listing = [];
  replyStatus = 204;
  requests = [];
  told = [];
  held.open = [];
  held.reply = [];
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('asks store — what waits on the user', () => {
  it('an ask opened over the socket waits once even when its frame repeats, and its close takes away only that ask', async () => {
    const { asks, frame } = await loadPage();

    frame(opened(permissionAsk('send-report')));
    frame(opened(permissionAsk('send-report')));
    frame(opened(handoffAsk('sign-in', 'tab-1')));
    frame(opened(permissionAsk('send-invoice')));
    expect(waiting(asks)).toEqual(['send-report', 'sign-in', 'send-invoice']);
    expect(asks.handoffOn('tab-1')?.ask_id).toBe('sign-in');

    frame(closed('send-report'));
    expect(waiting(asks)).toEqual(['sign-in', 'send-invoice']);
    expect(asks.offScreen.map((a) => a.kind === 'permission' && a.data.summary)).toEqual(['Send the send-invoice email']);
  });

  it('an ask from a run no chat turn shows still waits, with no origin; one missing its action or tab never does', async () => {
    const { asks, frame } = await loadPage();

    frame(opened(permissionAsk('scheduled-send', null)));
    frame(opened(permissionAsk('half-origin', { type: 'user' })));
    frame(opened({ ask_id: 'no-action', kind: 'permission', asked_at: '', data: { action_id: '', summary: 'x', origin: null } }));
    frame(opened({ ask_id: 'no-tab', kind: 'handoff', asked_at: '', data: { reason: 'Sign in', origin: null } }));

    expect(waiting(asks)).toEqual(['scheduled-send', 'half-origin']);
    expect(asks.offScreen.map((a) => a.data.origin)).toEqual([null, null]);
  });

  it('a question waits with its answers as sent, on its card and never among the rows; one with no question, too few or too many answers, or an answer with no label is dropped loudly', async () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { asks, frame } = await loadPage();

    frame(opened(permissionAsk('send-report')));
    frame(opened(questionAsk('which-days', { answers: [{ label: 'Tuesday' }, { label: 'Friday', description: 'After 5' }], multi_select: true })));
    frame(opened(handoffAsk('sign-in', 'tab-1')));
    frame(opened(questionAsk('no-question', { question: '' })));
    frame(opened(questionAsk('one-answer', { answers: [{ label: 'Tuesday' }] })));
    frame(opened(questionAsk('five-answers', { answers: ['A', 'B', 'C', 'D', 'E'].map((label) => ({ label })) })));
    frame(opened(questionAsk('blank-label', { answers: [{ label: 'Tuesday' }, { label: '' }] })));
    frame(opened(questionAsk('not-a-list', { answers: 'Tuesday or Friday' })));
    const dropped = errors.mock.calls.slice();
    errors.mockRestore();

    expect(waiting(asks)).toEqual(['send-report', 'which-days', 'sign-in']);
    expect(asks.queue[1]).toEqual({
      ask_id: 'which-days',
      kind: 'question',
      asked_at: '2026-11-20T09:00:00+00:00',
      data: {
        question: 'Which day suits you?',
        answers: [
          { label: 'Tuesday', description: '' },
          { label: 'Friday', description: 'After 5' },
        ],
        multi_select: true,
        origin: mainChat,
      },
    });
    expect(dropped).toEqual(
      ['no-question', 'one-answer', 'five-answers', 'blank-label', 'not-a-list'].map((id) => ['[asks] unreadable ask dropped:', id]),
    );
    expect(asks.questions.map((a) => a.ask_id)).toEqual(['which-days']);
    expect(asks.rows.map((a) => a.ask_id)).toEqual(['send-report', 'sign-in']);
    expect(asks.offScreen.map((a) => a.ask_id)).toEqual(['send-report']);
  });

  it("a question's answer leaves its row at once and posts the labels picked or the user's own words", async () => {
    const { asks, frame } = await loadPage();
    frame(opened(questionAsk('pick-day')));
    frame(opened(questionAsk('own-words')));

    const picked = asks.reply('pick-day', { labels: ['Tuesday', 'Friday'] });
    expect(waiting(asks)).toEqual(['own-words']);
    const typed = asks.reply('own-words', { other: 'Any evening next week' });
    expect(waiting(asks)).toEqual([]);

    await Promise.all([picked, typed]);
    expect(requests.filter((r) => r.method === 'POST')).toEqual([
      { method: 'POST', path: '/api/asks/reply/pick-day', body: { value: { labels: ['Tuesday', 'Friday'] } } },
      { method: 'POST', path: '/api/asks/reply/own-words', body: { value: { other: 'Any evening next week' } } },
    ]);
  });

  it("what is picked or typed on a question is held until it is sent; picked or typed away to nothing, nothing is held", async () => {
    const { asks, frame } = await loadPage();
    frame(opened(questionAsk('pick-days', { multi_select: true })));

    asks.draft('pick-days', { labels: ['Tuesday', 'Friday'] });
    expect(asks.drafts).toEqual({ 'pick-days': { labels: ['Tuesday', 'Friday'] } });
    asks.draft('pick-days', { other: 'Any evening' });
    expect(asks.drafts).toEqual({ 'pick-days': { other: 'Any evening' } });
    asks.draft('pick-days', { other: '' });
    expect(asks.drafts).toEqual({});
    asks.draft('pick-days', { labels: ['Friday'] });
    asks.draft('pick-days', { labels: [] });
    expect(asks.drafts).toEqual({});

    asks.draft('pick-days', { labels: ['Friday'] });
    await asks.reply('pick-days', { labels: ['Friday'] });
    expect(asks.drafts).toEqual({});
  });

  it("a question whose answer fails to send comes back with what the user had typed, and the failure is told in the server's words or a plain line when it gave none", async () => {
    const { asks, frame } = await loadPage();
    frame(opened(questionAsk('own-words')));
    listing = [questionAsk('own-words')];
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    asks.draft('own-words', { other: 'Any evening next week' });
    replyStatus = 500;
    await asks.reply('own-words', { other: 'Any evening next week' });
    await flushPromises();
    expect(told).toEqual(['Server error']);
    expect(waiting(asks)).toEqual(['own-words']);
    expect(asks.drafts['own-words']).toEqual({ other: 'Any evening next week' });

    replyStatus = 502;
    await asks.reply('own-words', { other: 'Any evening next week' });
    await flushPromises();
    expect(told).toEqual(['Server error', "Couldn't send your answer"]);
    expect(asks.drafts['own-words']).toEqual({ other: 'Any evening next week' });
  });

  it('a skipped question leaves at once and is dismissed on the server; one already closed elsewhere stays gone quietly; a failed skip brings it back with its picks and says why', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(questionAsk('skip-me')));
    frame(opened(questionAsk('closed-elsewhere')));
    frame(opened(questionAsk('skip-fails', { multi_select: true })));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const confirm = holdNext('reply');
    const skipped = asks.dismiss('skip-me');
    expect(waiting(asks)).toEqual(['closed-elsewhere', 'skip-fails']);
    await confirm();
    await skipped;

    replyStatus = 404;
    await asks.dismiss('closed-elsewhere');
    await flushPromises();
    expect(told).toEqual([]);
    expect(warn).not.toHaveBeenCalled();
    expect(waiting(asks)).toEqual(['skip-fails']);

    listing = [questionAsk('skip-fails', { multi_select: true })];
    asks.draft('skip-fails', { labels: ['Tuesday'] });
    replyStatus = 502;
    await asks.dismiss('skip-fails');
    await flushPromises();
    expect(told).toEqual(["Couldn't skip the question"]);
    expect(waiting(asks)).toEqual(['skip-fails']);
    expect(asks.drafts['skip-fails']).toEqual({ labels: ['Tuesday'] });

    expect(requests.filter((r) => r.method === 'POST')).toEqual(
      ['skip-me', 'closed-elsewhere', 'skip-fails'].map((id) => ({ method: 'POST', path: `/api/asks/dismiss/${id}`, body: {} })),
    );
  });

  it('on reconnect, an ask the server no longer lists is dropped, a listed one comes back once, and one that arrived during the read stays', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('closed-while-offline')));
    listing = [permissionAsk('still-open')];

    const release = holdNext('open');
    const read = asks.refreshOpen();
    frame(opened(permissionAsk('arrived-during-read')));
    await release();
    await read;
    expect(waiting(asks)).toEqual(['arrived-during-read', 'still-open']);

    listing = [permissionAsk('still-open'), permissionAsk('arrived-during-read')];
    await asks.refreshOpen();
    expect(waiting(asks)).toEqual(['arrived-during-read', 'still-open']);
  });

  it('a failed read of the open asks leaves everything that was waiting in place', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('send-report')));
    vi.stubGlobal('fetch', async () => json(500, { error: 'Server error' }));

    await asks.refreshOpen();

    expect(waiting(asks)).toEqual(['send-report']);
  });

  it('an ask closed while the open asks are being read is not brought back by that read', async () => {
    const { asks, frame } = await loadPage();
    listing = [permissionAsk('closed-elsewhere'), permissionAsk('never-closed')];

    const release = holdNext('open');
    const read = asks.refreshOpen();
    frame(closed('closed-elsewhere'));
    await release();
    await read;

    expect(waiting(asks)).toEqual(['never-closed']);
  });

  it('an ask the user answers while the open asks are being read is not brought back by that read', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('answered')));
    listing = [permissionAsk('answered'), permissionAsk('never-closed')];

    const release = holdNext('open');
    const read = asks.refreshOpen();
    await asks.reply('answered', true);
    await release();
    await read;

    expect(waiting(asks)).toEqual(['never-closed']);
    expect(told).toEqual([]);
  });

  it("an answer leaves its row before the server confirms it, a read landing meanwhile keeps it gone, and the server gets the user's choice", async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('send-report')));
    listing = [permissionAsk('send-report'), permissionAsk('never-closed')];

    const confirm = holdNext('reply');
    const answer = asks.reply('send-report', true);
    expect(waiting(asks)).toEqual([]);

    await asks.refreshOpen();
    expect(waiting(asks)).toEqual(['never-closed']);

    await confirm();
    await answer;
    expect(told).toEqual([]);
    expect(waiting(asks)).toEqual(['never-closed']);
    expect(requests.filter((r) => r.method === 'POST')).toEqual([
      { method: 'POST', path: '/api/asks/reply/send-report', body: { value: true } },
    ]);
  });

  it('an answer the server says came too late stays gone quietly, with no failure to report', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('answered-elsewhere')));
    listing = [permissionAsk('answered-elsewhere')];
    replyStatus = 404;

    await asks.reply('answered-elsewhere', false);
    await flushPromises();

    expect(told).toEqual([]);
    expect(waiting(asks)).toEqual([]);
    expect(opensRead()).toBe(0);
  });

  it('an answer that fails for another reason is reported, and its ask comes back only if the server still has it open', async () => {
    const { asks, frame } = await loadPage();
    frame(opened(permissionAsk('still-open')));
    frame(opened(permissionAsk('gone-on-server')));
    listing = [permissionAsk('still-open')];
    replyStatus = 500;

    await Promise.all([asks.reply('still-open', true), asks.reply('gone-on-server', true)]);
    await flushPromises();

    expect(told).toEqual(['Server error', 'Server error']);
    expect(waiting(asks)).toEqual(['still-open']);
  });
});

describe('asks store — a sign-in that waits on a code', () => {
  it('a UniFi code ask opened over the socket waits for its code, which goes to the server exactly as typed, leading zero and all, and its row goes', async () => {
    const { asks, frame } = await loadPage();

    frame(opened(codeAsk('unifi-sign-in')));
    expect(waiting(asks)).toEqual(['unifi-sign-in']);
    expect(asks.offScreen.map((a) => a.ask_id)).toEqual(['unifi-sign-in']);

    await asks.reply('unifi-sign-in', { code: '048217' });

    expect(told).toEqual([]);
    expect(requests.filter((r) => r.method === 'POST')).toEqual([
      { method: 'POST', path: '/api/asks/reply/unifi-sign-in', body: { value: { code: '048217' } } },
    ]);
    expect(waiting(asks)).toEqual([]);
  });

  it("a code the server refuses is reported in the server's words, never with the code, and its row comes back for another try; one already answered elsewhere stays gone, quietly", async () => {
    const { asks, frame } = await loadPage();
    frame(opened(codeAsk('wrong-code')));
    listing = [codeAsk('wrong-code')];
    replyStatus = 400;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await asks.reply('wrong-code', { code: '111111' });
    await flushPromises();

    expect(told).toEqual(['Server error']);
    expect(JSON.stringify([told, warn.mock.calls])).not.toContain('111111');
    expect(waiting(asks)).toEqual(['wrong-code']);

    frame(opened(codeAsk('answered-elsewhere')));
    listing = [codeAsk('wrong-code'), codeAsk('answered-elsewhere')];
    replyStatus = 404;
    const reads = opensRead();
    warn.mockClear();

    await asks.reply('answered-elsewhere', { code: '048217' });
    await flushPromises();

    expect(told).toEqual(['Server error']);
    expect(waiting(asks)).toEqual(['wrong-code']);
    expect(opensRead()).toBe(reads);
    expect(warn).not.toHaveBeenCalled();
  });

  it('only a UniFi sign-in is taken; a code ask for a service its row has no words for is dropped loudly', async () => {
    const { asks, frame } = await loadPage();
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});

    frame(opened(codeAsk('other-sign-in', 'other')));
    frame(opened(codeAsk('unifi-sign-in')));

    expect(waiting(asks)).toEqual(['unifi-sign-in']);
    expect(error.mock.calls).toEqual([['[asks] unreadable ask dropped:', 'other-sign-in']]);
  });
});
