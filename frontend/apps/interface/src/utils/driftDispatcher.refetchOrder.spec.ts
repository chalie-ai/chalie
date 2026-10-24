// @vitest-environment happy-dom
/**
 * driftDispatcher — feature spec for refetch ORDER: a turn read that answers
 * after a later one must not repaint over it.
 *
 * Every frame that changes a turn sends a fresh read of its whole block, and
 * nothing makes those reads answer in the order they were sent. A turn that
 * settles gains no new rows, so the read sent on an earlier `updated` or
 * `working` frame carries the SAME version as the settle's read, and the
 * version guard (which drops only a strictly older block) lets it through. If
 * it answers last it repaints the settled turn as still running: the live
 * activity line comes back, the closing footer goes, and the turn is marked
 * working again, so the send gate stays shut. Nothing arrives later to correct
 * it.
 *
 * Split into its own file for the same reason as the `.typeResolution` and
 * `.delegate` siblings: `driftDispatcher.spec.ts` mocks `./turnDom` wholesale,
 * and this can only be seen on the real rendered turn.
 *
 * Everything runs for real: the turnDom surface registry, the live act-trail,
 * the cancel reconcile, and TurnView as every surface's component. Only the
 * network is faked, at `fetch`, and it HOLDS every read until the test
 * answers it, so each test sets the order the answers land in.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushPromises } from '@vue/test-utils';
import type { WsPushEvent } from '@chalie/shared';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../api/conversation';

/** Every thread read sent, in send order, each held until `answer` is called. */
const reads: Array<{ url: URL; answer: (block: unknown) => void }> = [];

function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  return new Promise((resolve) => {
    reads.push({
      url,
      answer: (block) => resolve(new Response(
        JSON.stringify({ success: true, result: block }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )),
    });
  });
}

const TURN = 7;
const CH = 'delegate:web_search';
const STARTED_AT = '2026-01-01T00:00:00Z';

function msg(id: string, role: ConversationMessage['role'], content: string, settled = false): ConversationMessage {
  return {
    id, role, content, timestamp: '2026-01-01 00:00:00', day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: TURN,
    ...(settled ? { settled: true } : {}),
  };
}

function userBlock(working: boolean, messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]?.content ?? '', last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

function delegateBlock(working: boolean, messages: ConversationMessage[]): DelegateTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]?.content ?? '', last_activity_at: null,
    working, duration_ms: 0, messages, type: null, channel: CH,
  };
}

const QUESTION = msg('70', 'user', 'When does the museum open?');
const TASK = msg('80', 'user', 'Look up the museum opening hours');

function frame(data: Record<string, unknown>): WsPushEvent {
  return data as unknown as WsPushEvent;
}

let current: { teardown: () => void } | null = null;

/** A spine following user turn 7 and a subagent panel following delegate
 *  turn 7, both mid-flight, as the page shows them before any frame below. */
async function makeWorld() {
  vi.resetModules();
  reads.length = 0;
  vi.stubGlobal('fetch', fakeBackend);
  const { createPinia, setActivePinia } = await import('pinia');
  setActivePinia(createPinia());
  const turnDom = await import('./turnDom');
  const liveActTrail = await import('./liveActTrail');
  const { dispatchDrift, registerSessionHooks } = await import('./driftDispatcher');
  const { default: TurnView } = await import('../components/conversation/TurnView.vue');
  registerSessionHooks({
    releasePendingSend: () => { /* not under test */ },
    getPanelThreadId: () => null,
    getPanelType: () => 'user',
    setErrorMessage: () => { /* not under test */ },
    finishTurn: async () => { /* not under test */ },
    drainQueues: () => { /* not under test */ },
  });

  const spine = document.body.appendChild(document.createElement('div'));
  const panel = document.body.appendChild(document.createElement('div'));
  turnDom.registerSurface({ id: 'spine', type: 'user', container: spine, component: TurnView });
  turnDom.registerSurface({
    id: 'panel', channel: CH, container: panel, component: TurnView,
    props: { canReply: false, fullThread: true }, accepts: (id) => id === TURN,
  });
  turnDom.upsertTurnToSurfaces(userBlock(true, [QUESTION]), 'user');
  turnDom.setTurnWorking(TURN, 'user', true);
  turnDom.upsertDelegateTurn(delegateBlock(true, [TASK]));

  const world = {
    turnDom,
    spine,
    panel,
    send: (data: Record<string, unknown>) => dispatchDrift(frame(data)),
    teardown: () => {
      for (const el of [spine, panel]) turnDom.clearSurfaceContainer(el);
      for (const feed of ['user', CH]) liveActTrail.clearAll(feed);
      document.body.innerHTML = '';
    },
  };
  current = world;
  return world;
}

/** Answer the held read at `index`, then let the upsert it triggers land. */
async function answer(index: number, block: unknown): Promise<void> {
  reads[index]!.answer(block);
  await flushPromises();
}

/** The live activity line (TurnView draws it only while the rendered block
 *  says working). */
const thinking = (el: HTMLElement) => el.querySelector('.activity');
/** The closing footer's timestamp (a trace-only footer has none). */
const closingFooter = (el: HTMLElement) => el.querySelector('.speech-form__timestamp');

afterEach(() => {
  current?.teardown();
  current = null;
  vi.unstubAllGlobals();
});

describe('a refetch that answers after the settle cannot repaint the turn as running', () => {
  it('completed: the "updated" read answering last with working:true at the same version leaves the turn settled', async () => {
    const w = await makeWorld();
    // The same reply row, read before and after the execution closed on it.
    const replyInFlight = msg('71', 'assistant', 'It opens at nine.');
    const reply = msg('71', 'assistant', 'It opens at nine.', true);
    expect(thinking(w.spine)).not.toBeNull();

    // The reply row lands; its `updated` read is sent and the server answers
    // before the execution closes on that row, but the answer is slow.
    w.send({ status: 'updated', turn_id: TURN, type: 'user' });
    w.send({ state: 'completed', turn_id: TURN, started_at: STARTED_AT, type: 'user' });
    await flushPromises();
    expect(reads.map((r) => r.url.pathname)).toEqual([`/api/threads/${TURN}`, `/api/threads/${TURN}`]);

    // The settle's read answers first, then the earlier one.
    await answer(1, userBlock(false, [QUESTION, reply]));
    expect(thinking(w.spine)).toBeNull();
    await answer(0, userBlock(true, [QUESTION, replyInFlight]));

    expect(w.spine.textContent).toContain('It opens at nine.');
    expect(thinking(w.spine)).toBeNull();
    expect(closingFooter(w.spine)).not.toBeNull();
    expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(false);
  });

  it('cancelled: the "updated" read answering after the cancel reconcile leaves the interim reply closed and the turn idle', async () => {
    const w = await makeWorld();
    const interim = msg('71', 'assistant', 'Let me check the museum site.');

    w.send({ status: 'updated', turn_id: TURN, type: 'user' });
    w.send({ state: 'cancelled', turn_id: TURN, started_at: STARTED_AT, type: 'user' });
    await flushPromises();
    expect(reads).toHaveLength(2);

    // The reconcile's read: the cancelled turn keeps the reply it already
    // wrote, which now closes the exchange.
    await answer(1, userBlock(false, [QUESTION, interim]));
    expect(closingFooter(w.spine)).not.toBeNull();
    await answer(0, userBlock(true, [QUESTION, interim]));

    expect(w.spine.textContent).toContain('Let me check the museum site.');
    expect(thinking(w.spine)).toBeNull();
    expect(closingFooter(w.spine)).not.toBeNull();
    expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(false);
  });

  it('cancelled before any reply: the "working" read answering after the reconcile removed the turn does not bring it back', async () => {
    const w = await makeWorld();

    w.send({ state: 'working', turn_id: TURN, started_at: STARTED_AT, type: 'user' });
    w.send({ state: 'cancelled', turn_id: TURN, started_at: STARTED_AT, type: 'user' });
    await flushPromises();
    expect(reads).toHaveLength(2);

    // The server strips the reply-less question, so the turn is gone.
    await answer(1, userBlock(false, []));
    expect(w.turnDom.getTurnEl(TURN, 'user', w.spine)).toBeNull();
    await answer(0, userBlock(true, [QUESTION]));

    expect(w.turnDom.getTurnEl(TURN, 'user', w.spine)).toBeNull();
    expect(w.spine.textContent).not.toContain('When does the museum open?');
    expect(w.turnDom.isTurnWorking(TURN, 'user')).toBe(false);
  });

  it('delegate: the subagent panel\'s "updated" read answering after its settle leaves the panel settled', async () => {
    const w = await makeWorld();
    const answerRow = msg('81', 'assistant', 'The museum opens at nine.');
    expect(thinking(w.panel)).not.toBeNull();

    w.send({ status: 'updated', turn_id: TURN, channel: CH });
    w.send({ state: 'completed', turn_id: TURN, started_at: STARTED_AT, channel: CH });
    await flushPromises();
    expect(reads.map((r) => r.url.searchParams.get('channel'))).toEqual([CH, CH]);

    await answer(1, delegateBlock(false, [TASK, answerRow]));
    expect(thinking(w.panel)).toBeNull();
    await answer(0, delegateBlock(true, [TASK, answerRow]));

    expect(w.panel.textContent).toContain('The museum opens at nine.');
    expect(thinking(w.panel)).toBeNull();
    expect(closingFooter(w.panel)).not.toBeNull();
  });
});
