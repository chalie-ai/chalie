// @vitest-environment happy-dom
/**
 * An open subagent transcript across a dropped connection — feature spec.
 *
 * While a delegate transcript is open the panel only learns of new rows from
 * its frames, and a dropped socket loses them for good: no replay, and the live
 * pills are cleared on disconnect. So when the connection comes back the
 * session store must read the open transcript again, once, by its channel, and
 * show whatever the server says it is now. Three edges matter:
 *   - nothing is read when no transcript is open (or the person has left it);
 *   - a transcript the backend no longer keeps (an empty block) stays
 *     "Transcript expired" instead of being painted as an empty turn, whether
 *     the nudge is a reconnect or a stray frame;
 *   - a failed or mismatched read leaves the panel as it was.
 *
 * The same open transcript is also where a running subagent is stopped. Its
 * turn number collides with the chat's own turn numbers, so the stop must be
 * addressed by the subagent's channel alone, must undo nothing, and must leave
 * a chat turn with the same number running.
 *
 * Real throughout: the session store (its real `init()` wiring), the real
 * drift dispatcher, ThreadPanel, the TurnView it registers as its surface, the
 * turnDom surface registry, the REST wrappers and the shared ApiClient. Two
 * boundaries are faked: `fetch` (the backend, answering thread reads and
 * stops) and the WebSocket singleton (so the connect, disconnect and push
 * callbacks the store registers can be fired the way a real socket would fire
 * them).
 *
 * The session store, turnDom and the dispatcher carry module-level state, so
 * every test re-imports a fresh module graph via `vi.resetModules()`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import type { WsPushEvent } from '@chalie/shared';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../api/conversation';

const { fakeWs, wsCallbacks } = vi.hoisted(() => {
  const wsCallbacks: { onConnect: () => void; onDisconnect: () => void; onDrift: (data: unknown) => void } = {
    onConnect: () => { /* replaced by session.init() */ },
    onDisconnect: () => { /* replaced by session.init() */ },
    onDrift: () => { /* replaced by session.init() */ },
  };
  return {
    wsCallbacks,
    fakeWs: {
      send: (): void => { /* not under test */ },
      onConnect: (cb: () => void): void => { wsCallbacks.onConnect = cb; },
      onDisconnect: (cb: () => void): void => { wsCallbacks.onDisconnect = cb; },
      onDrift: (cb: (data: unknown) => void): void => { wsCallbacks.onDrift = cb; },
      onAny: (): void => { /* not under test */ },
      connect: (): void => { /* not under test */ },
      ensureAlive: (): void => { /* not under test */ },
    },
  };
});

vi.mock('@chalie/shared', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@chalie/shared')>()),
  getWebSocket: () => fakeWs,
}));

const SEARCH = { channel: 'delegate:web_search', turn_id: 12 };

/** One `GET /api/threads/<id>` the backend answered: which turn, addressed by what. */
interface Read {
  turnId: number;
  channel: string | null;
  type: string | null;
}

/** What the backend answers a thread read with; throwing answers HTTP 500. */
type Serve = (read: Read) => unknown;

const reads: Read[] = [];
let serve: Serve = () => {
  throw new Error('no block served');
};

const delegateReads = (): Read[] => reads.filter((r) => r.channel !== null);

/** Every stop (`DELETE`) the backend was sent, as the path and query it named. */
const stops: string[] = [];
/** How the backend answers a stop; a test can hold it open to look mid-flight. */
let answerStop: () => Promise<Response> = async () => json({ success: true });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The backend at the network boundary: thread reads, stops, the dock's
 *  thinking-level read, and the pending-permission listing the store re-reads
 *  on every connect. */
async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (init?.method === 'DELETE') {
    stops.push(url.pathname + url.search);
    return answerStop();
  }
  if (url.pathname.startsWith('/api/threads/thinking-level/')) {
    return json({ success: true, result: { level: 'auto' } });
  }
  if (url.pathname === '/api/policies/pending') {
    return json({ success: true, result: [], pagination: { page: 1, limit: 50, total: 0 } });
  }
  const match = /^\/api\/threads\/(\d+)$/.exec(url.pathname);
  if (!match) return json({ error: 'unrouted' }, 404);
  const read: Read = {
    turnId: Number(match[1]),
    channel: url.searchParams.get('channel'),
    type: url.searchParams.get('type'),
  };
  reads.push(read);
  try {
    return json({ success: true, result: await serve(read) });
  } catch {
    return json({ error: 'backend failure' }, 500);
  }
}

function msg(id: string, role: ConversationMessage['role'], content: string, turnId: number): ConversationMessage {
  return { id, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: turnId };
}

function delegateBlock(
  ref: { channel: string; turn_id: number },
  messages: ConversationMessage[],
  working = false,
): DelegateTurnBlock {
  return {
    turn_id: ref.turn_id, gist: null, preview: messages[0]?.content ?? '', last_activity_at: null,
    working, duration_ms: 0, messages, type: null, channel: ref.channel,
  };
}

function userThread(turnId: number, messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: turnId, gist: 'Museum planning', preview: messages[0]?.content ?? '', last_activity_at: null,
    working: false, duration_ms: 0, messages, type: 'user',
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

const TASK = msg('120', 'user', 'Look up the museum opening hours', 12);
const ANSWER = msg('121', 'assistant', 'The museum opens at nine.', 12);
const OTHER_RUN = [
  msg('130', 'user', 'Refactor the config parser', 12),
  msg('131', 'assistant', 'Refactor finished.', 12),
];
const TRIP = [msg('770', 'user', 'Plan the museum trip', 77), msg('771', 'assistant', 'Happy to help plan it.', 77)];

/** A transcript that is still running shows the live working anchor; a settled
 *  one does not — the one DOM signature of working versus settled. */
const showsWorking = (panel: VueWrapper): boolean => panel.element.querySelector('.act-cycle') !== null;

const updatedFrame = (ref: { channel: string; turn_id: number }): WsPushEvent =>
  ({ status: 'updated', turn_id: ref.turn_id, channel: ref.channel }) as unknown as WsPushEvent;

/** The server's cancel confirmation for a subagent turn: addressed by its
 *  channel, with no type, as every delegate frame reaches the wire. */
const cancelledFrame = (ref: { channel: string; turn_id: number }): WsPushEvent =>
  ({ state: 'cancelled', turn_id: ref.turn_id, started_at: '2026-01-01T00:00:00Z', channel: ref.channel }) as unknown as WsPushEvent;

/** The accessible names of every stop control rendered under `root`. */
const stopControls = (root: Element): string[] =>
  [...root.querySelectorAll('button')]
    .map((b) => b.getAttribute('aria-label') ?? '')
    .filter((name) => /stop/i.test(name));

/** The subagent stop control in the open panel, found by its accessible name. */
const subagentStop = (panel: VueWrapper): HTMLButtonElement | null =>
  (panel.element as Element).querySelector<HTMLButtonElement>('button[aria-label="Stop subagent"]');

let wrapper: VueWrapper | null = null;
const cleanups: Array<() => void> = [];

/** A fresh app: the real store wired by `init()`, and the real panel mounted. */
async function freshApp() {
  vi.resetModules();
  setActivePinia(createPinia());
  const { useSessionStore } = await import('./session');
  const { default: ThreadPanel } = await import('../components/conversation/ThreadPanel.vue');
  const session = useSessionStore();
  session.init();
  wrapper = mount(ThreadPanel, { attachTo: document.body });
  return { session, panel: wrapper };
}

/** The socket drops and comes back — the way the real service would fire it. */
async function dropAndRestoreSocket(): Promise<void> {
  wsCallbacks.onDisconnect();
  await flushPromises();
  wsCallbacks.onConnect();
  await flushPromises();
}

beforeEach(() => {
  reads.length = 0;
  stops.length = 0;
  answerStop = async () => json({ success: true });
  serve = () => {
    throw new Error('no block served');
  };
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup();
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('reconnecting with a subagent transcript open', () => {
  it('re-reads the open transcript once, by its channel, and shows it settled when the server says it settled', async () => {
    serve = () => delegateBlock(SEARCH, [TASK], true);
    const { session, panel } = await freshApp();
    session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(showsWorking(panel)).toBe(true);
    expect(panel.text()).not.toContain('The museum opens at nine.');
    reads.length = 0;

    // The subagent finished while the socket was down: its settling frame died with it.
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER], false);
    await dropAndRestoreSocket();

    expect(reads).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(panel.text()).toContain('The museum opens at nine.');
    expect(showsWorking(panel)).toBe(false);
    expect(session.panelDelegate).toEqual(SEARCH);
  });

  it('does not roll the transcript back when the reconnect read arrives after a newer frame already landed', async () => {
    serve = () => delegateBlock(SEARCH, [TASK], true);
    const { session, panel } = await freshApp();
    session.openDelegatePanel(SEARCH);
    await flushPromises();
    reads.length = 0;

    // The reconnect read is slow and was answered from before the answer existed;
    // a frame for the same turn overtakes it with the settled block.
    const slowRead = deferred<DelegateTurnBlock>();
    serve = () => (reads.length === 1 ? slowRead.promise : delegateBlock(SEARCH, [TASK, ANSWER], false));
    wsCallbacks.onConnect();
    await flushPromises();
    wsCallbacks.onDrift(updatedFrame(SEARCH));
    await flushPromises();
    expect(panel.text()).toContain('The museum opens at nine.');

    slowRead.resolve(delegateBlock(SEARCH, [TASK], true));
    await flushPromises();

    expect(delegateReads()).toHaveLength(2);
    expect(panel.text()).toContain('The museum opens at nine.');
    expect(showsWorking(panel)).toBe(false);
  });

  it.each([
    ['no panel is open', async (): Promise<void> => { /* nothing opened */ }],
    ['a typed thread is open', async (): Promise<void> => {
      serve = () => userThread(77, TRIP);
      const { useSessionStore } = await import('./session');
      useSessionStore().openThreadPanel(77, 'user');
      await flushPromises();
    }],
    ['the person has left the subagent transcript', async (): Promise<void> => {
      serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
      const { useSessionStore } = await import('./session');
      const session = useSessionStore();
      session.openDelegatePanel(SEARCH);
      await flushPromises();
      session.closeThreadPanel();
      await flushPromises();
    }],
  ])('makes no subagent read when %s', async (_name, arrange) => {
    await freshApp();
    await arrange();
    reads.length = 0;

    await dropAndRestoreSocket();

    expect(delegateReads()).toEqual([]);
  });
});

describe('an expired subagent transcript', () => {
  async function openExpired() {
    serve = () => delegateBlock(SEARCH, []);
    const app = await freshApp();
    app.session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(app.panel.text()).toContain('Transcript expired');
    reads.length = 0;
    return app;
  }

  it.each([
    ['a reconnect', async (): Promise<void> => { await dropAndRestoreSocket(); }],
    ['a stray frame for the turn', async (): Promise<void> => {
      wsCallbacks.onDrift(updatedFrame(SEARCH));
      await flushPromises();
    }],
  ])('stays expired after %s finds the backend still keeps nothing for it', async (_name, nudge) => {
    const { panel } = await openExpired();

    await nudge();

    // The read really happened; its empty answer just is not drawn as a turn.
    expect(delegateReads()).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(panel.text()).toContain('Transcript expired');
    expect(panel.element.querySelector('[data-turn-id]')).toBeNull();
  });

  it('fills in the moment a frame finds the subagent\'s rows on the backend', async () => {
    const { panel } = await openExpired();

    serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
    wsCallbacks.onDrift(updatedFrame(SEARCH));
    await flushPromises();

    expect(panel.text()).not.toContain('Transcript expired');
    expect(panel.text()).toContain('The museum opens at nine.');
  });
});

describe('a reconnect read that goes wrong', () => {
  it.each([
    ['the backend fails the read', (): unknown => { throw new Error('backend down'); }],
    ['the backend answers on another subagent\'s channel', (): unknown =>
      delegateBlock({ channel: 'delegate:code_agent', turn_id: 12 }, OTHER_RUN)],
  ])('leaves the open transcript as it was when %s', async (_name, failure) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => { /* a failed read is logged; not asserted */ });
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
    const { session, panel } = await freshApp();
    session.openDelegatePanel(SEARCH);
    await flushPromises();
    reads.length = 0;

    serve = failure;
    await dropAndRestoreSocket();

    expect(delegateReads()).toHaveLength(1);
    expect(warn).toHaveBeenCalled();
    expect(panel.text()).toContain('The museum opens at nine.');
    expect(panel.text()).not.toContain('Refactor');
    expect(session.panelDelegate).toEqual(SEARCH);
  });
});

describe('stopping a running subagent from its open transcript', () => {
  const SEARCH_STOP = '/api/threads/12?channel=delegate%3Aweb_search';
  const FETCH_CALL = { tool_name: 'fetch_url', id: 41, turn_id: 12, channel: 'delegate:web_search', summary: 'museum.example/hours' };

  /** The panel open on the running web_search subagent, one live pill showing. */
  async function openRunning() {
    serve = () => delegateBlock(SEARCH, [TASK], true);
    const app = await freshApp();
    app.session.openDelegatePanel(SEARCH);
    await flushPromises();
    wsCallbacks.onDrift({ ...FETCH_CALL, state: 'started' } as unknown as WsPushEvent);
    await flushPromises();
    expect(app.panel.text()).toContain('museum.example/hours');
    reads.length = 0;
    return app;
  }

  /** Every "turn interrupted" hand-back the store announces to a reply box. */
  function recordHandBacks(): unknown[] {
    const handBacks: unknown[] = [];
    const listener = (e: Event): void => { handBacks.push((e as CustomEvent).detail); };
    document.addEventListener('session:turn-interrupted', listener);
    cleanups.push(() => document.removeEventListener('session:turn-interrupted', listener));
    return handBacks;
  }

  function holdStopOpen(): () => void {
    const stopReturns = deferred<void>();
    answerStop = () => stopReturns.promise.then(() => json({ success: true }));
    return () => stopReturns.resolve();
  }

  it('sends one stop addressed by the subagent\'s channel and never by a type, and hands nothing back to a reply box', async () => {
    const { session } = await openRunning();
    const handBacks = recordHandBacks();
    serve = () => delegateBlock(SEARCH, [TASK], false);

    await session.requestStop(SEARCH);
    await flushPromises();

    expect(stops).toEqual([SEARCH_STOP]);
    expect(handBacks).toEqual([]);
  });

  it('drops the live pills the moment stop is pressed, and re-reads the transcript only once the stop has returned', async () => {
    const { session, panel } = await openRunning();
    const releaseStop = holdStopOpen();
    serve = () => delegateBlock(SEARCH, [TASK], false);

    const stopping = session.requestStop(SEARCH);
    await flushPromises();

    // Mid-flight: the pills are gone at once, but nothing has been re-read and
    // the panel still says the subagent is running until the server answers.
    expect(stops).toEqual([SEARCH_STOP]);
    expect(panel.text()).not.toContain('museum.example/hours');
    expect(delegateReads()).toEqual([]);
    expect(showsWorking(panel)).toBe(true);

    releaseStop();
    await stopping;
    await flushPromises();

    expect(delegateReads()).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(showsWorking(panel)).toBe(false);
  });

  it('settles the panel from the server\'s cancel confirmation after its stop control is clicked, and never offers a reply box', async () => {
    serve = () => delegateBlock(SEARCH, [TASK], true);
    const { session, panel } = await freshApp();
    const offersReplyBox = (): boolean => panel.find('textarea').exists();
    session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(showsWorking(panel)).toBe(true);
    expect(offersReplyBox()).toBe(false);

    const releaseStop = holdStopOpen();
    serve = () => delegateBlock(SEARCH, [TASK], false);
    const stop = subagentStop(panel);
    expect(stop).not.toBeNull();
    stop!.click();
    await flushPromises();
    expect(stops).toEqual([SEARCH_STOP]);
    expect(offersReplyBox()).toBe(false);

    wsCallbacks.onDrift(cancelledFrame(SEARCH));
    await flushPromises();
    expect(showsWorking(panel)).toBe(false);
    expect(stopControls(panel.element as Element)).toEqual([]);
    expect(offersReplyBox()).toBe(false);

    releaseStop();
    await flushPromises();
    expect(showsWorking(panel)).toBe(false);
    expect(stopControls(panel.element as Element)).toEqual([]);
    expect(offersReplyBox()).toBe(false);
    expect(panel.text()).toContain('Look up the museum opening hours');
    expect(session.panelDelegate).toEqual(SEARCH);
  });

  it('leaves a chat turn with the same number running, with its own stop control, when the subagent is stopped', async () => {
    serve = () => delegateBlock(SEARCH, [TASK], true);
    const { session, panel } = await freshApp();
    const turnDom = await import('../utils/turnDom');
    const { default: TurnView } = await import('../components/conversation/TurnView.vue');
    // The chat sits before the panel in the document, so anything that looks
    // the stopped turn up by its bare number finds the chat turn first.
    const chat = document.body.insertBefore(document.createElement('div'), document.body.firstChild);
    turnDom.registerSurface({ id: 'spine', type: 'user', container: chat, component: TurnView });
    cleanups.push(() => turnDom.clearSurfaceContainer(chat));
    const question = msg('125', 'user', 'What time does the museum open?', 12);
    turnDom.upsertTurnToSurfaces({ ...userThread(12, [question]), working: true }, 'user');
    turnDom.setTurnWorking(12, 'user', true);
    const handBacks = recordHandBacks();
    session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(stopControls(chat)).toEqual(['Stop and undo']);

    serve = () => delegateBlock(SEARCH, [TASK], false);
    subagentStop(panel)!.click();
    await flushPromises();
    wsCallbacks.onDrift(cancelledFrame(SEARCH));
    await flushPromises();

    expect(showsWorking(panel)).toBe(false);
    expect(stops).toEqual([SEARCH_STOP]);
    expect(handBacks).toEqual([]);
    expect(turnDom.isTurnWorking(12, 'user')).toBe(true);
    expect(stopControls(chat)).toEqual(['Stop and undo']);
    expect(chat.textContent).toContain('What time does the museum open?');
  });
});
