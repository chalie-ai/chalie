// @vitest-environment happy-dom
/**
 * ThreadPanel in delegate mode — feature spec.
 *
 * Clicking a delegate pill opens the panel on that subagent's turn:
 * `session.openDelegatePanel(ref)` sets `panelDelegate`, and the panel reads
 * `GET /api/threads/<id>?channel=<channel>`. The transcript is read-only (no
 * reply box), is captioned as a task from Chalie, may legitimately be empty
 * (the backend keeps a delegate's transcript only for a while), and must never
 * paint a block that came back from some other channel.
 *
 * Real throughout: ThreadPanel, the real TurnView it registers as its surface,
 * the real turnDom surface registry, the real Pinia session store, the real
 * REST wrappers and shared ApiClient, and the real InputDock where the
 * typed-thread contrast needs it. The only fake is the backend itself: `fetch`
 * is the network boundary, answered here with the block each test serves.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import ThreadPanel from './ThreadPanel.vue';
import { useSessionStore } from '../../stores/session';
import { upsertDelegateTurn } from '../../utils/turnDom';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../../api/conversation';

const SEARCH = { channel: 'delegate:web_search', turn_id: 12 };
const AGENT = { channel: 'delegate:code_agent', turn_id: 12 };

/** One `GET /api/threads/<id>` the panel sent: which turn, addressed by what. */
interface Read {
  turnId: number;
  channel: string | null;
  type: string | null;
}

/** Answers a thread read with the block to serve; throwing answers HTTP 500. */
type Serve = (read: Read) => unknown;

const reads: Read[] = [];
let serve: Serve = () => {
  throw new Error('no block served');
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** The backend, at the network boundary: the thread read and the dock's
 *  thinking-level read, nothing else. */
async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname.startsWith('/api/threads/thinking-level/')) {
    return json({ success: true, result: { level: 'auto' } });
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

function delegateBlock(ref: { channel: string; turn_id: number }, messages: ConversationMessage[], working = false): DelegateTurnBlock {
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
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const TASK = msg('120', 'user', 'Look up the museum opening hours', 12);
const ANSWER = msg('121', 'assistant', 'The museum opens at nine.', 12);
const REFACTOR = [msg('130', 'user', 'Refactor the config parser', 12), msg('131', 'assistant', 'Refactor finished.', 12)];
const TRIP = [msg('770', 'user', 'Plan the museum trip', 77), msg('771', 'assistant', 'Happy to help plan it.', 77)];

const BACK = 'button[aria-label="Back to conversation"]';

let wrapper: VueWrapper | null = null;

function openPanel(): { panel: VueWrapper; session: ReturnType<typeof useSessionStore> } {
  wrapper = mount(ThreadPanel, { attachTo: document.body });
  return { panel: wrapper, session: useSessionStore() };
}

beforeEach(() => {
  setActivePinia(createPinia());
  reads.length = 0;
  serve = () => {
    throw new Error('no block served');
  };
  vi.stubGlobal('fetch', fakeBackend);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('opening a delegate transcript', () => {
  it('reads the transcript by its channel, headed by the subagent\'s name, with the task captioned and no reply box', async () => {
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();

    expect(reads).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(panel.find('header').text()).toContain('web_search');
    expect(panel.text()).toContain('Task from Chalie');
    expect(panel.text()).toContain('Look up the museum opening hours');
    expect(panel.text()).toContain('The museum opens at nine.');
    expect(panel.text()).not.toContain('Transcript expired');
    // Read-only: nothing to type into.
    expect(panel.find('textarea').exists()).toBe(false);
    // The task caption belongs to the first user row only, not to Chalie's answer.
    expect(panel.findAll('.msg-row--chalie').some((r) => r.text().includes('Task from Chalie'))).toBe(false);
  });

  it('says "Transcript expired" for an empty block, still names the subagent, offers no reply box, and fills in when the subagent\'s rows arrive afterwards', async () => {
    serve = () => delegateBlock(SEARCH, []);
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();

    expect(panel.text()).toContain('Transcript expired');
    expect(panel.find('header').text()).toContain('web_search');
    expect(panel.find('textarea').exists()).toBe(false);
    expect(panel.element.querySelector('[data-turn-id]')).toBeNull();
    // An expiry is a normal outcome, not a load failure.
    expect(session.errorMessage).toBeFalsy();

    // The live path: a delegate frame refetches the turn and upserts it to the
    // surface the panel registered.
    upsertDelegateTurn(delegateBlock(SEARCH, [TASK, ANSWER]));
    await flushPromises();

    expect(panel.text()).not.toContain('Transcript expired');
    expect(panel.text()).toContain('The museum opens at nine.');
  });

  it('refuses to render a block the backend answers on a different channel than the one opened, and reports the load failure instead of an expiry', async () => {
    // The panel warns about the mismatch; that it does is asserted below, the
    // text is not.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => { /* asserted below */ });
    serve = () => delegateBlock(AGENT, REFACTOR);
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();

    expect(reads).toEqual([{ turnId: 12, channel: 'delegate:web_search', type: null }]);
    expect(warn).toHaveBeenCalled();
    expect(panel.text()).not.toContain('Refactor');
    expect(panel.element.querySelector('[data-turn-id]')).toBeNull();
    expect(panel.text()).not.toContain('Transcript expired');
    expect(session.errorMessage).toBeTruthy();
    // Nothing hung on a spinner either.
    expect(panel.find('output').exists()).toBe(false);
  });

  it('stops the spinner and reports a load failure when the backend fails the read', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => { /* a failed read is logged; not asserted */ });
    serve = () => {
      throw new Error('backend down');
    };
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();

    expect(reads).toHaveLength(1);
    expect(session.errorMessage).toBeTruthy();
    expect(panel.find('output').exists()).toBe(false);
    expect(panel.element.querySelector('[data-turn-id]')).toBeNull();
    expect(panel.text()).not.toContain('Transcript expired');
  });

  it('puts keyboard focus on the back button when it opens, and returns it to the pill that opened it when it closes', async () => {
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
    const pill = document.createElement('button');
    document.body.append(pill);
    pill.focus();
    expect(document.activeElement).toBe(pill);
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(document.activeElement).toBe(panel.find(BACK).element);

    await panel.find(BACK).trigger('click');
    await flushPromises();
    expect(session.panelDelegate).toBeNull();
    expect(document.activeElement).toBe(pill);
  });
});

describe('moving between transcripts', () => {
  it.each([
    ['a transcript with rows', (late: ReturnType<typeof deferred<unknown>>) => late.resolve(delegateBlock(SEARCH, [TASK, ANSWER]))],
    ['an empty transcript', (late: ReturnType<typeof deferred<unknown>>) => late.resolve(delegateBlock(SEARCH, []))],
    ['a failed read', (late: ReturnType<typeof deferred<unknown>>) => late.reject(new Error('backend down'))],
  ])('%s that arrives late for a transcript the person has already left never touches the one they opened next', async (_label, settleLate) => {
    vi.spyOn(console, 'warn').mockImplementation(() => { /* a stale failed read is logged; not asserted */ });
    const slow = deferred<unknown>();
    serve = (read) => (read.channel === SEARCH.channel ? slow.promise : delegateBlock(AGENT, REFACTOR));
    const { panel, session } = openPanel();

    session.openDelegatePanel(SEARCH);
    await flushPromises();
    // Nested click: a second subagent's pill replaces the first.
    session.openDelegatePanel(AGENT);
    await flushPromises();
    expect(panel.find('header').text()).toContain('code_agent');
    expect(panel.text()).toContain('Refactor finished.');

    settleLate(slow);
    await flushPromises();

    expect(panel.find('header').text()).toContain('code_agent');
    expect(panel.text()).toContain('Refactor finished.');
    expect(panel.text()).not.toContain('The museum opens at nine.');
    expect(panel.text()).not.toContain('Transcript expired');
    expect(session.errorMessage).toBeFalsy();
    expect(panel.element.querySelectorAll('[data-turn-id]')).toHaveLength(1);
  });

  it('another run of the same subagent never leaks into the transcript that is open', async () => {
    serve = () => delegateBlock(SEARCH, [TASK, ANSWER]);
    const { panel, session } = openPanel();
    session.openDelegatePanel(SEARCH);
    await flushPromises();

    // The subagent runs again on the same channel (a new turn id) while the
    // panel is still on the first run.
    upsertDelegateTurn(delegateBlock(
      { channel: SEARCH.channel, turn_id: 13 },
      [msg('140', 'user', 'Look up the gallery hours', 13), msg('141', 'assistant', 'The gallery opens at ten.', 13)],
    ));
    await flushPromises();

    expect(panel.text()).toContain('The museum opens at nine.');
    expect(panel.text()).not.toContain('The gallery opens at ten.');
    expect(panel.element.querySelectorAll('[data-turn-id]')).toHaveLength(1);
  });

  it('a typed thread keeps its reply box and heading; opening a delegate over it drops the reply box and the thread\'s rows, and the back button closes the panel', async () => {
    serve = (read) => (read.channel ? delegateBlock(SEARCH, [TASK, ANSWER]) : userThread(77, TRIP));
    const { panel, session } = openPanel();

    session.openThreadPanel(77, 'user');
    await flushPromises();
    expect(reads).toEqual([{ turnId: 77, channel: null, type: 'user' }]);
    expect(panel.find('textarea').exists()).toBe(true);
    expect(panel.find('header').text()).toContain('Museum planning');
    expect(panel.text()).toContain('Happy to help plan it.');
    expect(panel.text()).not.toContain('Task from Chalie');

    session.openDelegatePanel(SEARCH);
    await flushPromises();
    expect(reads[1]).toEqual({ turnId: 12, channel: 'delegate:web_search', type: null });
    expect(panel.find('textarea').exists()).toBe(false);
    expect(panel.find('header').text()).toContain('web_search');
    expect(panel.text()).toContain('The museum opens at nine.');
    expect(panel.text()).not.toContain('Happy to help plan it.');

    await panel.find(BACK).trigger('click');
    await flushPromises();
    expect(session.panelDelegate).toBeNull();
    expect(session.panelThreadId).toBeNull();
    expect(panel.find('[role="dialog"]').exists()).toBe(false);
  });
});
