// @vitest-environment happy-dom
/**
 * driftDispatcher — feature spec for delegate (subagent) turn frames.
 *
 * A delegate runs a child turn on channel `delegate:<name>`. Its turn_id is
 * allocated independently of the conversation's, so it COLLIDES with a user
 * turn's id: user turn 5 and a web_search delegate's turn 5 are different
 * turns. Its frames reach the wire with `type` stripped and the channel in its
 * place. The failure this spec guards is the quiet one — a delegate frame
 * resolving its identity from the DOM, finding the user turn with the same
 * number, and repainting, settling or un-working THAT turn.
 *
 * Everything runs for real: the turnDom surface registry, the live act-trail,
 * the real TurnView as every surface's component (so the data-type /
 * data-channel stamping the dispatcher's lookups depend on is the production
 * one), and the real session store behind the pill clicks. Only the network
 * boundary is faked (`fetch`, answering the thread reads the real REST
 * wrappers and ApiClient make), and the session-hook seam is a recording registration, as in the sibling dispatcher specs (the real one is
 * wired by `session.init()`, which opens the WebSocket).
 *
 * Layout used by every test: a delegate surface for `delegate:web_search`, a
 * second one for `delegate:code_agent`, and a user spine, all rendering a
 * turn numbered 5. The delegate containers sit BEFORE the user's in document
 * order, so a lookup that forgets to require `data-type` finds a delegate copy
 * first.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushPromises } from '@vue/test-utils';
import type { WsPushEvent } from '@chalie/shared';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../api/conversation';

/** The backend, at the network boundary: what it will answer a thread read
 *  with, and every read it was asked for. Everything between `dispatchDrift`
 *  and this `fetch` is production code. */
const backend = {
  user: null as ConversationTurnBlock | null,
  delegate: null as DelegateTurnBlock | null,
  userReads: [] as Array<{ turnId: number; type: string | null }>,
  delegateReads: [] as Array<{ turnId: number; channel: string }>,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  const match = /^\/api\/threads\/(\d+)$/.exec(url.pathname);
  if (!match) return json({ error: 'unrouted' }, 404);
  const turnId = Number(match[1]);
  const channel = url.searchParams.get('channel');
  if (channel) backend.delegateReads.push({ turnId, channel });
  else backend.userReads.push({ turnId, type: url.searchParams.get('type') });
  const block = channel ? backend.delegate : backend.user;
  return block ? json({ success: true, result: block }) : json({ error: 'nothing served' }, 500);
}

const TURN = 5;
const CH = 'delegate:web_search';
const OTHER_CH = 'delegate:code_agent';
const STARTED_AT = '2026-01-01T00:00:00Z';

function msg(id: string, role: ConversationMessage['role'], content: string): ConversationMessage {
  return { id, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', turn_id: TURN };
}

function userBlock(working: boolean, messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]?.content ?? '', last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

function delegateBlock(channel: string, working: boolean, messages: ConversationMessage[]): DelegateTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]?.content ?? '', last_activity_at: null,
    working, duration_ms: 0, messages, type: null, channel,
  };
}

const USER_QUESTION = msg('50', 'user', 'What time does the museum open?');
const TASK = msg('60', 'user', 'Look up the museum opening hours');
const OTHER_TASK = msg('70', 'user', 'Refactor the config parser');
const DELEGATE_PROGRESS = msg('62', 'assistant', 'Checking two sources for the hours.');
const DELEGATE_ANSWER = msg('63', 'assistant', 'The museum opens at nine.');

function frame(data: Record<string, unknown>): WsPushEvent {
  return data as unknown as WsPushEvent;
}

interface WorldOptions {
  /** Register the web_search delegate surface at all (default true). */
  panel?: boolean;
  /** Which delegate turn ids that surface follows (default: turn 5). */
  accepts?: (turnId: number) => boolean;
}

async function makeWorld(options: WorldOptions = {}) {
  const { panel = true, accepts = (id: number) => id === TURN } = options;
  vi.resetModules();
  backend.user = null;
  backend.delegate = null;
  backend.userReads = [];
  backend.delegateReads = [];
  vi.stubGlobal('fetch', fakeBackend);
  const { createPinia, setActivePinia } = await import('pinia');
  setActivePinia(createPinia());
  const turnDom = await import('./turnDom');
  const liveActTrail = await import('./liveActTrail');
  const { dispatchDrift, registerSessionHooks } = await import('./driftDispatcher');
  const { useSessionStore } = await import('../stores/session');
  const { default: TurnView } = await import('../components/conversation/TurnView.vue');

  const hookCalls = {
    released: [] as string[],
    finished: [] as string[],
    errors: [] as string[],
    drains: 0,
  };
  registerSessionHooks({
    releasePendingSend: (id, type) => { hookCalls.released.push(`${type}:${id}`); },
    getPanelThreadId: () => null,
    getPanelType: () => 'user',
    setErrorMessage: (message) => { hookCalls.errors.push(message); },
    finishTurn: async (id, type) => { hookCalls.finished.push(`${type}:${id}`); },
    drainQueues: () => { hookCalls.drains += 1; },
  });

  const container = () => document.body.appendChild(document.createElement('div'));
  const delegateEl = container();
  const otherDelegateEl = container();
  const userEl = container();

  turnDom.registerSurface({ id: 'spine', type: 'user', container: userEl, component: TurnView });
  if (panel) {
    turnDom.registerSurface({
      id: 'panel', channel: CH, container: delegateEl, component: TurnView,
      props: { canReply: false, fullThread: true }, accepts,
    });
  }
  turnDom.registerSurface({
    id: 'other-panel', channel: OTHER_CH, container: otherDelegateEl, component: TurnView,
    props: { canReply: false, fullThread: true }, accepts: (id) => id === TURN,
  });

  // The three same-numbered turns, all mid-flight.
  turnDom.upsertTurnToSurfaces(userBlock(true, [USER_QUESTION]), 'user');
  turnDom.setTurnWorking(TURN, 'user', true);
  turnDom.upsertDelegateTurn(delegateBlock(CH, true, [TASK]));
  turnDom.upsertDelegateTurn(delegateBlock(OTHER_CH, true, [OTHER_TASK]));

  const containers = [delegateEl, otherDelegateEl, userEl];
  const world = {
    turnDom,
    liveActTrail,
    hookCalls,
    delegateEl,
    otherDelegateEl,
    userEl,
    session: useSessionStore(),
    send: (data: Record<string, unknown>) => dispatchDrift(frame(data)),
    teardown: () => {
      // Unmount what the dispatcher rendered (stops the running-pill clocks)
      // and drop the module-level feeds so nothing outlives the test.
      for (const el of containers) turnDom.clearSurfaceContainer(el);
      for (const feed of ['user', CH, OTHER_CH]) liveActTrail.clearAll(feed);
      document.body.innerHTML = '';
    },
  };
  current = world;
  return world;
}

let current: { teardown: () => void } | null = null;

afterEach(() => {
  current?.teardown();
  current = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('a typed frame that also carries a channel keeps routing by type', () => {
  it('a user turn_execution frame with channel "user" refetches the user turn, releases its send hold and settles it; no delegate surface is touched', async () => {
    const w = await makeWorld();
    const delegateBefore = w.delegateEl.innerHTML;
    const otherBefore = w.otherDelegateEl.innerHTML;

    backend.user = userBlock(true, [USER_QUESTION, msg('52', 'assistant', 'Checking the museum site now.')]);
    w.send({ state: 'working', turn_id: TURN, started_at: STARTED_AT, type: 'user', channel: 'user' });
    await flushPromises();

    expect(backend.userReads).toEqual([{ turnId: TURN, type: 'user' }]);
    expect(backend.delegateReads).toEqual([]);
    expect(w.userEl.textContent).toContain('Checking the museum site now.');
    expect(w.hookCalls.released).toEqual([`user:${TURN}`]);

    backend.user = userBlock(false, [USER_QUESTION, msg('53', 'assistant', 'It opens at nine.')]);
    w.send({ state: 'completed', turn_id: TURN, started_at: STARTED_AT, type: 'user', channel: 'user' });
    await flushPromises();

    expect(w.userEl.textContent).toContain('It opens at nine.');
    expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(false);
    expect(w.turnDom.isTurnDone(TURN, 'user')).toBe(true);
    expect(w.hookCalls.finished).toEqual([`user:${TURN}`]);
    expect(w.delegateEl.innerHTML).toBe(delegateBefore);
    expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
  });
});

describe('delegate frames are addressed by channel, never by the same-numbered user turn', () => {
  it('a type-less delegate "updated" frame repaints only the matching channel surface; the user turn is untouched and keeps its own type', async () => {
    const w = await makeWorld();
    const userBefore = w.userEl.innerHTML;
    const otherBefore = w.otherDelegateEl.innerHTML;
    backend.delegate = delegateBlock(CH, false, [TASK, DELEGATE_ANSWER]);

    w.send({ status: 'updated', turn_id: TURN, channel: CH });
    await flushPromises();

    expect(backend.delegateReads).toEqual([{ turnId: TURN, channel: CH }]);
    expect(backend.userReads).toEqual([]);
    expect(w.delegateEl.textContent).toContain('The museum opens at nine.');
    expect(w.userEl.textContent).not.toContain('The museum opens at nine.');
    expect(w.userEl.innerHTML).toBe(userBefore);
    expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
    // The delegate copy sits first in the document, yet only the typed copy
    // can answer for the turn's type.
    expect(w.turnDom.findTurnType(TURN)).toBe('user');
  });

  it('delegate tool-call frames build the delegate turn\'s own live trail, not the same-numbered user turn\'s', async () => {
    const w = await makeWorld();
    const call = { tool_name: 'fetch_url', id: 41, turn_id: TURN, channel: CH, summary: 'museum.example/hours', transcript_row_id: 61 };

    w.send({ ...call, state: 'started' });
    await flushPromises();

    expect(w.delegateEl.querySelectorAll('.act-tool')).toHaveLength(1);
    expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(0);
    expect(w.otherDelegateEl.querySelectorAll('.act-tool')).toHaveLength(0);

    w.send({ ...call, state: 'done' });
    await flushPromises();

    expect(w.delegateEl.querySelectorAll('.act-tool--done')).toHaveLength(1);
    expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(0);
    // A tool-call frame is purely visual — it never fetches anything.
    expect(backend.userReads).toEqual([]);
    expect(backend.delegateReads).toEqual([]);
  });

  it('a delegate "working" frame refetches the delegate surface only and leaves the user turn\'s work flag and send hold alone', async () => {
    const w = await makeWorld();
    const userBefore = w.userEl.innerHTML;
    const otherBefore = w.otherDelegateEl.innerHTML;
    backend.delegate = delegateBlock(CH, true, [TASK, DELEGATE_PROGRESS]);

    w.send({ state: 'working', turn_id: TURN, started_at: STARTED_AT, channel: CH });
    await flushPromises();

    expect(backend.delegateReads).toEqual([{ turnId: TURN, channel: CH }]);
    expect(backend.userReads).toEqual([]);
    expect(w.delegateEl.textContent).toContain('Checking two sources for the hours.');
    expect(w.userEl.innerHTML).toBe(userBefore);
    expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
    expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(true);
    expect(w.hookCalls.released).toEqual([]);
  });

  it.each(['completed', 'crashed', 'cancelled'])(
    'a delegate %s frame settles only the delegate turn: no user work flag, done mark, toast, queue drain or live pill is touched',
    async (state) => {
      const w = await makeWorld();
      // The user turn has a live tool pill of its own, which must survive.
      w.send({ tool_name: 'web_search', id: 31, turn_id: TURN, type: 'user', state: 'started', transcript_row_id: 50 });
      await flushPromises();
      expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(1);
      const otherBefore = w.otherDelegateEl.innerHTML;
      backend.delegate = delegateBlock(CH, false, [TASK, DELEGATE_ANSWER]);

      w.send({ state, turn_id: TURN, started_at: STARTED_AT, channel: CH });
      await flushPromises();

      expect(backend.delegateReads).toEqual([{ turnId: TURN, channel: CH }]);
      expect(backend.userReads).toEqual([]);
      expect(w.delegateEl.textContent).toContain('The museum opens at nine.');
      expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(1);
      expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
      expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(true);
      expect(w.turnDom.isTurnDone(TURN, 'user')).toBe(false);
      expect(w.hookCalls).toEqual({ released: [], finished: [], errors: [], drains: 0 });
    },
  );
});

describe('a delegate frame nobody is following does no REST work', () => {
  it.each([
    ['no surface is registered for its channel', { panel: false }],
    ['the surface for its channel follows a different turn', { accepts: (id: number) => id === 99 }],
  ])('issues no GET for updated / working / completed frames when %s', async (_label, options) => {
    const w = await makeWorld(options);
    const userBefore = w.userEl.innerHTML;
    const otherBefore = w.otherDelegateEl.innerHTML;
    w.send({ tool_name: 'fetch_url', id: 41, turn_id: TURN, channel: CH, state: 'started', transcript_row_id: 61 });

    w.send({ status: 'updated', turn_id: TURN, channel: CH });
    w.send({ state: 'working', turn_id: TURN, started_at: STARTED_AT, channel: CH });
    w.send({ state: 'completed', turn_id: TURN, started_at: STARTED_AT, channel: CH });
    await flushPromises();

    expect(backend.delegateReads).toEqual([]);
    expect(backend.userReads).toEqual([]);
    expect(w.userEl.innerHTML).toBe(userBefore);
    expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
    // The settle still drops the delegate's live pills even with nothing to
    // repaint, so a panel opened on it later does not resurrect a stale
    // running pill.
    expect(w.liveActTrail.liveTrailsFor(CH, TURN)).toEqual([]);
  });

  it('refuses to render a block that comes back from another channel than the one it was read on, and says so', async () => {
    const w = await makeWorld();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => { /* asserted below */ });
    const delegateBefore = w.delegateEl.innerHTML;
    const otherBefore = w.otherDelegateEl.innerHTML;
    backend.delegate = delegateBlock(OTHER_CH, false, [OTHER_TASK, msg('71', 'assistant', 'Content from the wrong channel.')]);

    w.send({ status: 'updated', turn_id: TURN, channel: CH });
    await flushPromises();

    expect(warn).toHaveBeenCalled();
    expect(document.body.textContent).not.toContain('Content from the wrong channel.');
    expect(w.delegateEl.innerHTML).toBe(delegateBefore);
    expect(w.otherDelegateEl.innerHTML).toBe(otherBefore);
  });
});

describe('a delegate call\'s live pill on its parent turn', () => {
  it('gains its delegate link from the re-emitted "started" frame without being duplicated, and keeps the same link once "done"', async () => {
    const w = await makeWorld();
    const ref = { channel: CH, turn_id: 12 };
    const call = { tool_name: 'web_search', id: 31, turn_id: TURN, type: 'user', channel: 'user', summary: 'museum hours', transcript_row_id: 50 };

    w.send({ ...call, state: 'started' });
    await flushPromises();
    expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(1);
    expect(w.userEl.querySelectorAll('button.act-tool')).toHaveLength(0);

    // The backend re-sends `started` once the child turn exists.
    w.send({ ...call, state: 'started', delegate: ref });
    await flushPromises();
    expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(1);
    expect(w.userEl.querySelectorAll('button.act-tool')).toHaveLength(1);

    w.send({ ...call, state: 'done' });
    await flushPromises();
    expect(w.userEl.querySelectorAll('.act-tool')).toHaveLength(1);
    expect(w.userEl.querySelectorAll('button.act-tool.act-tool--done')).toHaveLength(1);

    w.userEl.querySelector<HTMLButtonElement>('button.act-tool')!.click();
    expect(w.session.panelDelegate).toEqual(ref);
    // The parent's pill never leaks onto a delegate turn that shares its id.
    expect(w.delegateEl.querySelectorAll('.act-tool')).toHaveLength(0);
  });
});
