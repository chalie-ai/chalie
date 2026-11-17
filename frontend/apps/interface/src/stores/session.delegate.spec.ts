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
 * Which subagents still run rides the same lost frames. The Subagents dock
 * and the header's pink Subagents button list the agents table, read again on
 * every connect, so an agent that started or ended while the socket was down
 * shows as the table now says. The status mark on the call that started a
 * subtask follows the chat turn instead: whether the subtask runs comes from
 * its own frames once seen, else from the turn's last read; the drop throws
 * the frames' word away, and a settled turn has no lane for the reconnect to
 * reconcile. So every turn that had a subagent running when the socket
 * dropped is read again on reconnect, and the mark follows that read.
 *
 * Real throughout: the session store (its real `init()` wiring), the agents
 * store, the real drift dispatcher, ThreadPanel, the TurnView it registers as
 * its surface, the header and its Subagents dock, the turnDom surface
 * registry, the REST wrappers and the shared ApiClient. Two boundaries are
 * faked: `fetch` (the backend, answering thread reads, stops and the agents
 * listing) and the WebSocket singleton (so the connect, disconnect and push
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
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock, ToolCallChip } from '../api/conversation';
import type { Agent } from '../api/agents';

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
/** The backend's ack for a stop it carried out. */
const STOPPED = { success: true, result: { cancelled: true, reason: null } };
/** How the backend answers a stop; a test can hold it open to look mid-flight. */
let answerStop: () => Promise<Response> = async () => json(STOPPED);

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The running agents the backend lists; a test sets them before a connect. */
let servedAgents: Agent[] = [];
/** While true, the agents listing answers HTTP 500. */
let agentsDown = false;

/** The backend at the network boundary: thread reads, stops, the dock's
 *  thinking-level read, and the pending-permission listing, the header's open
 *  browser tabs and the running agents that the stores re-read on every
 *  connect. */
async function fakeBackend(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (init?.method === 'DELETE') {
    stops.push(url.pathname + url.search);
    return answerStop();
  }
  if (url.pathname === '/api/agents/all') {
    if (agentsDown) return json({ error: 'backend failure' }, 500);
    const limit = Number(url.searchParams.get('limit'));
    return json({ success: true, result: servedAgents, pagination: { page: 1, limit, total: servedAgents.length } });
  }
  if (url.pathname.startsWith('/api/threads/thinking-level/')) {
    return json({ success: true, result: { level: 'auto' } });
  }
  if (url.pathname === '/api/policies/pending') {
    return json({ success: true, result: [], pagination: { page: 1, limit: 50, total: 0 } });
  }
  if (url.pathname === '/api/browser/tabs') {
    return json({ success: true, result: { tabs: [] } });
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
  return { id, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: turnId };
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
const showsWorking = (panel: VueWrapper): boolean => panel.element.querySelector('.activity') !== null;

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
  answerStop = async () => json(STOPPED);
  servedAgents = [];
  agentsDown = false;
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
    answerStop = () => stopReturns.promise.then(() => json(STOPPED));
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

  it('keeps the live pills while the stop is in flight, and drops them and re-reads the transcript only once the stop has returned', async () => {
    const { session, panel } = await openRunning();
    const releaseStop = holdStopOpen();
    serve = () => delegateBlock(SEARCH, [TASK], false);

    const stopping = session.requestStop(SEARCH);
    await flushPromises();

    // Mid-flight: nothing changes until the server answers — the subagent may
    // still be running if the stop fails.
    expect(stops).toEqual([SEARCH_STOP]);
    expect(panel.text()).toContain('museum.example/hours');
    expect(delegateReads()).toEqual([]);
    expect(showsWorking(panel)).toBe(true);

    releaseStop();
    await stopping;
    await flushPromises();

    expect(panel.text()).not.toContain('museum.example/hours');
    expect(delegateReads()).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(showsWorking(panel)).toBe(false);
  });

  it.each([
    ['the server refuses it', () => Promise.resolve(json({ success: false, error: 'Invalid channel' }, 400))],
    ['the server fails', () => Promise.resolve(json({ error: 'backend failure' }, 500))],
    ['the network drops it', () => Promise.reject(new TypeError('Failed to fetch'))],
    ['the answer is not a stop ack', () => Promise.resolve(json({ success: true }))],
  ])('says the stop failed and leaves the subagent running, stoppable again, when %s', async (_name, failure) => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => { /* asserted below */ });
    // The backend still serves the subagent as running when the failed stop re-reads it.
    const { session, panel } = await openRunning();
    answerStop = failure;

    subagentStop(panel)!.click();
    await flushPromises();

    expect(stops).toEqual([SEARCH_STOP]);
    expect(session.errorMessage).toBe("Couldn't stop the subagent. Try again.");
    expect(logged).toHaveBeenCalled();
    expect(panel.text()).toContain('museum.example/hours');
    // The one read is the re-read that found it still running.
    expect(delegateReads()).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(showsWorking(panel)).toBe(true);

    // The retry goes through, and the failure it answered no longer shows.
    answerStop = async () => json(STOPPED);
    serve = () => delegateBlock(SEARCH, [TASK], false);
    subagentStop(panel)!.click();
    await flushPromises();

    expect(stops).toEqual([SEARCH_STOP, SEARCH_STOP]);
    expect(session.errorMessage).toBeNull();
    expect(showsWorking(panel)).toBe(false);
  });

  it.each([
    ['the network drops the answer', () => Promise.reject(new TypeError('Failed to fetch'))],
    ['a gateway times out on the answer', () => Promise.resolve(json({ error: 'gateway timeout' }, 504))],
  ])('takes the stop as done, with no failure shown, when %s after the subagent was stopped', async (_name, lost) => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => { /* asserted below */ });
    const { session, panel } = await openRunning();
    const answer = deferred<void>();
    answerStop = () => answer.promise.then(lost);

    subagentStop(panel)!.click();
    await flushPromises();
    // The server stopped it and said so over the socket before answering.
    serve = () => ({ ...delegateBlock(SEARCH, [TASK], false), cancelled: true });
    wsCallbacks.onDrift(cancelledFrame(SEARCH));
    await flushPromises();
    answer.resolve();
    await flushPromises();

    expect(stops).toEqual([SEARCH_STOP]);
    expect(logged).toHaveBeenCalled();
    expect(session.errorMessage).toBeNull();
    expect(panel.text()).not.toContain('museum.example/hours');
    expect(panel.text()).toContain('This subagent was stopped.');
    expect(showsWorking(panel)).toBe(false);
    expect(stopControls(panel.element as Element)).toEqual([]);
  });

  it('takes a stop of a subagent that had already finished quietly, and shows its final transcript', async () => {
    const { session, panel } = await openRunning();
    answerStop = async () => json({ success: true, result: { cancelled: null, reason: 'no_active_turn' } });
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER], false);

    await session.requestStop(SEARCH);
    await flushPromises();

    expect(session.errorMessage).toBeNull();
    expect(panel.text()).toContain('The museum opens at nine.');
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

describe('a chat turn\'s running subtask across a dropped connection', () => {
  const SUBTASK = { channel: 'delegate:subtask', turn_id: 4 };
  const PLAN_TURN = 901;

  /** Subtask #4's agents row while it runs: its agent number is not its turn
   *  number. */
  const SUBTASK_AGENT: Agent = {
    agent_id: 17, spawned_by_channel: 'user', spawned_in_turn: PLAN_TURN, type: 'subtask',
    prompt: 'Compare the museum opening hours', status: 'running', response_id: null,
    turn_id: SUBTASK.turn_id, channel: SUBTASK.channel, summary: 'compare museum hours',
  };

  /** How subtask #4's turn stood when the backend read it, as its chip says. */
  type SubtaskAsRead = 'working' | 'completed' | 'crashed' | 'cancelled';

  /** The settled chat turn that started subtask #4, its chip's run state as
   *  the backend reads it. */
  function planTurn(subtask: SubtaskAsRead): ConversationTurnBlock {
    const start: ToolCallChip = {
      id: 91, tool_name: 'subtask', params: null, summary: 'compare museum hours', state: 'done', ended_at: null,
      delegate: {
        ...SUBTASK, working: subtask === 'working', crashed: subtask === 'crashed', cancelled: subtask === 'cancelled',
      },
    };
    return userThread(PLAN_TURN, [
      msg('9010', 'user', 'Plan my museum day', PLAN_TURN),
      { ...msg('9011', 'assistant', 'I started a subtask for that.', PLAN_TURN), tool_calls: [start], settled: true },
    ]);
  }

  /** The backend answers a read of the chat turn alone; any other read fails. */
  function serveChatTurn(subtask: SubtaskAsRead): Serve {
    return (read) => {
      if (read.turnId !== PLAN_TURN || read.type !== 'user' || read.channel !== null) throw new Error('not this turn');
      return planTurn(subtask);
    };
  }

  /** The subtask's own execution frame: addressed by its channel, no type. */
  const subtaskFrame = (state: 'working' | 'completed'): WsPushEvent =>
    ({
      id: SUBTASK.turn_id, channel: SUBTASK.channel, type: null, turn_id: SUBTASK.turn_id,
      started_at: '2026-01-01T00:00:01Z', ended_at: state === 'working' ? null : '2026-01-01T00:00:09Z',
      cancel_requested: false, state, stop_reason: null,
    }) as unknown as WsPushEvent;

  /** A fresh app — the header, its Subagents dock, the page's first connect
   *  (which reads the agents the backend lists), and the chat showing the
   *  planning turn as the backend first read it — the subtask's working frame
   *  already in. */
  async function chatWithRunningSubtask(asRead: SubtaskAsRead): Promise<HTMLElement> {
    vi.resetModules();
    setActivePinia(createPinia());
    const { useSessionStore } = await import('./session');
    const turnDom = await import('../utils/turnDom');
    const { default: TurnView } = await import('../components/conversation/TurnView.vue');
    const { default: PresenceBar } = await import('../components/layout/PresenceBar.vue');
    const { default: SubagentDock } = await import('../components/overlays/SubagentDock.vue');
    useSessionStore().init();
    const header = mount(PresenceBar, { attachTo: document.body });
    const dock = mount(SubagentDock, { attachTo: document.body });
    const chat = document.body.appendChild(document.createElement('div'));
    turnDom.registerSurface({ id: 'spine', type: 'user', container: chat, component: TurnView });
    cleanups.push(() => {
      turnDom.clearSurfaceContainer(chat);
      dock.unmount();
      header.unmount();
    });
    wsCallbacks.onConnect();
    await flushPromises();
    turnDom.upsertTurnToSurfaces(planTurn(asRead), 'user');
    wsCallbacks.onDrift(subtaskFrame('working'));
    await flushPromises();
    return chat;
  }

  /** The titles the Subagents dock lists, in order. */
  const listed = (): string[] =>
    [...document.querySelectorAll('aside[aria-label="Subagents"] .drawer-row__label')].map((el) => el.textContent!.trim());

  /** Whether the header's Subagents button shows its pink running cue. */
  const headerIsPink = (): boolean =>
    document.querySelector('#subagentDockBtn')!.classList.contains('has-activity');

  /** What the status mark on the call that started subtask #4 reads. */
  const callMark = (chat: HTMLElement): string | null =>
    chat.querySelector('.calls .call .call__mark')?.getAttribute('aria-label') ?? null;

  it('the first connect lists the running subtask in the dock from the agents table and lights the header', async () => {
    servedAgents = [SUBTASK_AGENT];
    await chatWithRunningSubtask('working');

    expect(listed()).toEqual(['compare museum hours']);
    expect(headerIsPink()).toBe(true);
  });

  it('drops the subtask from the dock and the header\'s pink on reconnect when it finished while the socket was down', async () => {
    servedAgents = [SUBTASK_AGENT];
    const chat = await chatWithRunningSubtask('working');
    expect(listed()).toEqual(['compare museum hours']);

    // Its `agent_changed` frame died with the socket; the table now lists it
    // no more.
    servedAgents = [];
    serve = serveChatTurn('completed');
    await dropAndRestoreSocket();

    expect(listed()).toEqual([]);
    expect(headerIsPink()).toBe(false);
    expect(chat.textContent).toContain('I started a subtask for that.');
  });

  it('lists an agent that started while the socket was down on reconnect, though no turn on screen started it', async () => {
    await chatWithRunningSubtask('completed');
    expect(listed()).toEqual([]);
    expect(headerIsPink()).toBe(false);

    servedAgents = [{
      ...SUBTASK_AGENT, agent_id: 18, spawned_by_channel: 'schedule', spawned_in_turn: 55,
      type: 'web_search', channel: 'delegate:web_search', turn_id: 9, summary: 'morning news',
    }];
    serve = serveChatTurn('completed');
    await dropAndRestoreSocket();

    expect(listed()).toEqual(['morning news']);
    expect(headerIsPink()).toBe(true);
  });

  it('keeps the dock as it stood when the agents read fails on reconnect, and corrects it on the next', async () => {
    servedAgents = [SUBTASK_AGENT];
    await chatWithRunningSubtask('working');
    vi.spyOn(console, 'warn').mockImplementation(() => { /* the failed read's warning */ });

    servedAgents = [];
    agentsDown = true;
    serve = serveChatTurn('working');
    await dropAndRestoreSocket();
    expect(listed()).toEqual(['compare museum hours']);
    expect(headerIsPink()).toBe(true);

    agentsDown = false;
    await dropAndRestoreSocket();
    expect(listed()).toEqual([]);
    expect(headerIsPink()).toBe(false);
  });

  it('its call\'s mark reads running again on reconnect when the re-read says it still runs, though the turn\'s first read had it done', async () => {
    // The first read is older than the subtask's own working frame, which wins.
    const chat = await chatWithRunningSubtask('completed');
    expect(callMark(chat)).toBe('running');

    serve = serveChatTurn('working');
    await dropAndRestoreSocket();

    expect(callMark(chat)).toBe('running');
  });

  it('reads the turn again on the next reconnect when the backend failed the first re-read', async () => {
    const chat = await chatWithRunningSubtask('completed');
    expect(callMark(chat)).toBe('running');
    vi.spyOn(console, 'warn').mockImplementation(() => { /* the failed re-read's warning */ });

    serve = () => {
      throw new Error('backend down');
    };
    await dropAndRestoreSocket();

    // The drop threw the working frame away and the read failed, so the mark
    // falls back to the turn's first read and nothing on screen says the
    // subtask runs for the second drop to find; only the turn kept from the
    // failed read brings it back.
    expect(callMark(chat)).toBe('succeeded');
    serve = serveChatTurn('working');
    await dropAndRestoreSocket();

    expect(callMark(chat)).toBe('running');
  });

  it.each([
    ['finished', 'succeeded', 'completed', null],
    ['crashed', 'failed', 'crashed', '1 failed'],
    ['was stopped', 'failed', 'cancelled', '1 failed'],
  ] as const)('a subtask that %s while the socket was down: its call\'s mark reads %s after reconnect, as the re-read says', async (_how, mark, asRead, failedCount) => {
    const chat = await chatWithRunningSubtask('working');
    expect(callMark(chat)).toBe('running');

    serve = serveChatTurn(asRead);
    await dropAndRestoreSocket();

    expect(callMark(chat)).toBe(mark);
    expect(chat.querySelector('.step__failed')?.textContent ?? null).toBe(failedCount);
  });
});
