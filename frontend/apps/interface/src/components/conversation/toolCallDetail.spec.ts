// @vitest-environment happy-dom
/**
 * Opened tool-call rows — feature spec for what a call shows when its row is
 * opened.
 *
 * Every call row is a button; opening it reads that one call from the backend
 * and shows, under the row, only what matters for its kind of call: the
 * command a shell call ran, a line diff of what a file edit replaced, what a
 * browser action acted on and the pages it reached, what a delegate was asked
 * and the way into its transcript, an email's fields in a fixed order, and
 * for anything else every argument it was called with. A running call that
 * settles while open is read again; an open row stays open, with what it
 * read, when its step settles into its stored row.
 *
 * Exchanges reach the page as they do in the app: a settled exchange through
 * the real surface registry, a running one through live frames into the real
 * dispatcher and act-trail, all onto the real TurnView, with the real session
 * store behind "Open transcript". Every module is fresh per test, so no
 * opened row or read call outlives it. Only the network boundary is stood in
 * for: `fetch`, answering the thread read and each call's own read
 * (GET /api/tool-calls/<id>), able to hold a read until the test releases it
 * or to fail it; plus the session hooks `session.init()` would register, as
 * in the sibling dispatcher specs.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { nextTick } from 'vue';
import { flushPromises } from '@vue/test-utils';
import type { WsPushEvent } from '@chalie/shared';
import type { ConversationMessage, ConversationTurnBlock, ToolCallChip, ToolCallDetail } from '../../api/conversation';

const TURN = 61;
const STARTED_AT = '2026-01-01T09:00:00Z';

type Params = Record<string, unknown>;

/** The backend at the network boundary: the block a thread read returns, each
 *  call as its own read returns it, which reads fail, whether reads are held
 *  for the test to release, and every call read it was asked for. */
const backend = {
  thread: null as ConversationTurnBlock | null,
  calls: new Map<number, ToolCallDetail>(),
  failing: new Set<number>(),
  holding: false,
  held: [] as Array<{ id: number; release: () => void }>,
  reads: [] as number[],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** A call read is answered with the call as it stood when it was asked for. */
function answerCallRead(id: number): Response {
  if (backend.failing.has(id)) return json({ error: 'Could not read the tool call' }, 500);
  const call = backend.calls.get(id);
  return call ? json({ success: true, result: call }) : json({ error: 'Tool call not found' }, 404);
}

async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  const callRead = /^\/api\/tool-calls\/(\d+)$/.exec(url.pathname);
  if (callRead) {
    const id = Number(callRead[1]);
    backend.reads.push(id);
    const answer = answerCallRead(id);
    if (!backend.holding) return answer;
    return new Promise((resolve) => backend.held.push({ id, release: () => resolve(answer) }));
  }
  if (url.pathname === `/api/threads/${TURN}` && backend.thread) return json({ success: true, result: backend.thread });
  return json({ error: 'unrouted' }, 404);
}

function releaseHeld(): void {
  for (const h of backend.held.splice(0)) h.release();
}

/** A call as the backend reads it back: settled unless `over` says otherwise,
 *  with the stored result envelope the model read. */
function call(id: number, tool_name: string, summary: string, params: Params, over: Partial<ToolCallDetail> = {}): ToolCallDetail {
  const state = over.state ?? 'done';
  return {
    id, tool_name, params, summary, state, delegate: null, pages: [],
    created_at: '2026-01-01T09:00:01Z',
    ended_at: state === 'started' ? null : '2026-01-01T09:00:02Z',
    result: state === 'started' ? null : `[${tool_name}(status=success)]\n{"ok":true}\n[end:${tool_name}]`,
    ...over,
  };
}

/** Serves each call's own read, and returns the chips its stored row carries. */
function serve(...calls: ToolCallDetail[]): ToolCallChip[] {
  return calls.map((c) => {
    backend.calls.set(c.id, c);
    return { id: c.id, tool_name: c.tool_name, params: c.params, summary: c.summary, state: c.state, ended_at: c.ended_at, delegate: c.delegate };
  });
}

function row(id: number, role: ConversationMessage['role'], content: string, extra: Partial<ConversationMessage> = {}): ConversationMessage {
  return {
    id: String(id), role, content, timestamp: '2026-01-01 09:00:00', day: '2026-01-01',
    created_at: STARTED_AT, turn_id: TURN, ...extra,
  };
}

function block(messages: ConversationMessage[], working: boolean): ConversationTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

const QUESTION = row(600, 'user', 'Sort out my Saturday trip to Gozo');
const ANSWER = row(602, 'assistant', 'Your trip is sorted.', { settled: true, created_at: '2026-01-01T09:00:30Z' });

function toolFrame(c: ToolCallDetail): WsPushEvent {
  return { tool_name: c.tool_name, id: c.id, turn_id: TURN, type: 'user', state: c.state, summary: c.summary, params: c.params } as unknown as WsPushEvent;
}

let teardown: (() => void) | null = null;

/** A user spine, as the app registers it — the dispatcher, act-trail, surface
 *  registry, session store and TurnView all fresh modules. */
async function makeWorld() {
  vi.resetModules();
  backend.thread = null;
  backend.calls = new Map();
  backend.failing = new Set();
  backend.holding = false;
  backend.held = [];
  backend.reads = [];
  vi.stubGlobal('fetch', fakeBackend);
  const pinia = await import('pinia');
  pinia.setActivePinia(pinia.createPinia());
  const turnDom = await import('../../utils/turnDom');
  const liveActTrail = await import('../../utils/liveActTrail');
  const { dispatchDrift, registerSessionHooks } = await import('../../utils/driftDispatcher');
  const { useSessionStore } = await import('../../stores/session');
  const { default: View } = await import('./TurnView.vue');
  registerSessionHooks({
    releasePendingSend: () => {},
    getPanelThreadId: () => null,
    getPanelType: () => 'user',
    setErrorMessage: () => {},
    finishTurn: async () => {},
    drainQueues: () => {},
  });
  const el = document.body.appendChild(document.createElement('div'));
  turnDom.registerSurface({ id: 'spine', type: 'user', container: el, component: View });
  teardown = () => {
    turnDom.clearSurfaceContainer(el);
    liveActTrail.clearAll('user');
    el.remove();
  };
  return {
    el,
    session: useSessionStore(),
    /** A settled exchange whose one step made `calls`, its steps opened. */
    settled: async (...calls: ToolCallDetail[]) => {
      turnDom.upsertTurnToSurfaces(block([QUESTION, row(601, 'assistant', '', { tool_calls: serve(...calls) }), ANSWER], false), 'user');
      await flushPromises();
      await openSteps(el);
    },
    /** The exchange running on its question. */
    working: async () => {
      turnDom.upsertTurnToSurfaces(block([QUESTION], true), 'user');
      turnDom.setTurnWorking(TURN, 'user', true);
      await flushPromises();
    },
    send: async (data: WsPushEvent | Record<string, unknown>) => {
      dispatchDrift(data as WsPushEvent);
      await flushPromises();
    },
  };
}

afterEach(() => {
  teardown?.();
  teardown = null;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Opens every folded step's call list by its summary toggle. */
async function openSteps(root: Element): Promise<void> {
  for (const s of root.querySelectorAll('.step__summary')) {
    const toggle = s.closest('button')!;
    if (toggle.getAttribute('aria-expanded') === 'false') toggle.click();
  }
  await nextTick();
}

/** The call row that reads `summary`. */
function rowNamed(root: Element, summary: string): HTMLElement {
  const found = [...root.querySelectorAll<HTMLElement>('.call')].filter((r) => r.querySelector('.call__summary')?.textContent === summary);
  expect(found).toHaveLength(1);
  return found[0]!;
}

/** What a row opened: the element its `aria-controls` names, or null. */
const controlled = (r: HTMLElement): HTMLElement | null => document.getElementById(r.getAttribute('aria-controls') ?? '');

function detailOf(r: HTMLElement): HTMLElement {
  const d = controlled(r);
  expect(d).not.toBeNull();
  return d!;
}

/** Opens a row as a person would, lets its read land, and returns what it opened. */
async function open(r: HTMLElement): Promise<HTMLElement> {
  expect(r.closest('[inert]')).toBeNull();
  r.click();
  await flushPromises();
  expect(r.getAttribute('aria-expanded')).toBe('true');
  return detailOf(r);
}

const heading = (d: Element) => d.querySelector('.call-detail__heading')?.textContent ?? null;
const fields = (d: Element) => [...d.querySelectorAll('dl > div')].map((f) => [f.querySelector('dt')!.textContent, f.querySelector('dd')!.textContent]);
const pages = (d: Element) => [...d.querySelectorAll('ol > li')].map((li) => [...li.querySelectorAll('p')].map((p) => p.textContent));
const code = (d: Element) => d.querySelector('pre')?.textContent ?? null;
const transcriptButtons = (d: Element) => [...d.querySelectorAll<HTMLButtonElement>('button')].filter((b) => b.textContent?.includes('Open transcript'));

/** Each diff line as [what it is, its mark, its text]. */
function diff(d: Element): string[][] {
  return [...d.querySelectorAll('.diff-line')].map((line) => {
    const op = [...line.classList].find((c) => c.startsWith('diff-line--'))!.slice('diff-line--'.length);
    const mark = line.querySelector('.diff-line__mark')!.textContent!;
    return [op, mark, line.textContent!.slice(mark.length)];
  });
}

describe('what an opened call shows, by its kind', () => {
  it('a command shows what it ran; a file edit its path and a line diff of what it replaced; a new file every line added, an overwritten one its plain contents; a read its source and the lines it read', async () => {
    const w = await makeWorld();
    await w.settled(
      call(6101, 'bash', 'Running the lease tests', { command: 'pytest -q tests/test_lease.py' }, {
        result: '[bash(status=success, exit_code=0)]\n{"stdout":"3 passed","exit_code":0}\n[end:bash]',
      }),
      call(6102, 'edit_file', 'Capping the deposit', {
        path: 'lease/deposit.py',
        search: 'def cap(rent):\n    return rent * 2\n',
        replace: 'def cap(rent):\n    return rent * 1.5\n',
      }),
      call(6103, 'write_file', 'Writing the trip notes', { path: 'notes/trip.md', contents: '# Trip\nFerry at 9\n' }, {
        result: '[write_file(status=success)]\n{"path":"notes/trip.md","bytes":18,"created":true}\n[end:write_file]',
      }),
      call(6104, 'write_file', 'Rewriting the to-do list', { path: 'notes/todo.md', contents: 'Pack bags\nBook ferry' }, {
        result: '[write_file(status=success)]\n{"path":"notes/todo.md","bytes":20,"created":false}\n[end:write_file]',
      }),
      call(6105, 'read', 'Reading the lease', { source: 'lease-2026.pdf', start_line: 3, end_line: 9 }),
      call(6106, 'read', 'Reading the deposit rules', { source: 'deposit-rules.md', start_line: 3 }),
    );

    expect(code(await open(rowNamed(w.el, 'Running the lease tests')))).toBe('pytest -q tests/test_lease.py');

    const edit = await open(rowNamed(w.el, 'Capping the deposit'));
    expect(edit.textContent).toContain('lease/deposit.py');
    expect(diff(edit)).toEqual([
      ['same', ' ', 'def cap(rent):'],
      ['remove', '-', '    return rent * 2'],
      ['add', '+', '    return rent * 1.5'],
    ]);

    const created = await open(rowNamed(w.el, 'Writing the trip notes'));
    expect(created.textContent).toContain('notes/trip.md');
    expect(diff(created)).toEqual([['add', '+', '# Trip'], ['add', '+', 'Ferry at 9']]);
    expect(code(created)).toBeNull();

    const overwritten = await open(rowNamed(w.el, 'Rewriting the to-do list'));
    expect(overwritten.textContent).toContain('notes/todo.md');
    expect(diff(overwritten)).toEqual([]);
    expect(code(overwritten)).toBe('Pack bags\nBook ferry');

    const range = await open(rowNamed(w.el, 'Reading the lease'));
    expect([...range.querySelectorAll('p')].map((p) => p.textContent)).toEqual(['lease-2026.pdf', 'lines 3–9']);
    const from = await open(rowNamed(w.el, 'Reading the deposit rules'));
    expect([...from.querySelectorAll('p')].map((p) => p.textContent)).toEqual(['deposit-rules.md', 'from line 3']);
  });

  it('a browser call is headed by its action and shows what that action acted on — the address it opened, the text it searched for, what it clicked, why it handed over, the field and value it filled, the section it read — and the pages it reached; a scroll shows its action alone', async () => {
    const w = await makeWorld();
    const timetable = { title: 'Gozo ferry timetable', url: 'https://ferries.example/gozo' };
    await w.settled(
      call(6201, 'browser', 'Opening the timetable', { action: 'open', url: 'https://ferries.example/gozo' }, { pages: [timetable] }),
      call(6202, 'browser', 'Finding Saturday', { action: 'find', query: 'Saturday' }, { pages: [timetable] }),
      call(6203, 'browser', 'Clicking Book now', { action: 'click', target: 'Book now' }),
      call(6204, 'browser', 'Handing over the payment', { action: 'handoff', reason: 'The booking page needs your card details' }),
      call(6205, 'browser', 'Entering the passengers', { action: 'fill', target: '#passengers', value: '2' }),
      call(6206, 'browser', 'Reading the departures', { action: 'read', section: 'Saturday departures' }),
      call(6207, 'browser', 'Scrolling down', { action: 'scroll', direction: 'down' }),
    );

    const shown = async (summary: string) => {
      const d = await open(rowNamed(w.el, summary));
      return { heading: heading(d), fields: fields(d), pages: pages(d) };
    };
    expect(await shown('Opening the timetable')).toEqual({
      heading: 'open', fields: [['Url', 'https://ferries.example/gozo']], pages: [['Gozo ferry timetable', 'https://ferries.example/gozo']],
    });
    expect(await shown('Finding Saturday')).toEqual({
      heading: 'find', fields: [['Query', 'Saturday']], pages: [['Gozo ferry timetable', 'https://ferries.example/gozo']],
    });
    expect(await shown('Clicking Book now')).toEqual({ heading: 'click', fields: [['Target', 'Book now']], pages: [] });
    expect(await shown('Handing over the payment')).toEqual({
      heading: 'handoff', fields: [['Reason', 'The booking page needs your card details']], pages: [],
    });
    expect(await shown('Entering the passengers')).toEqual({ heading: 'fill', fields: [['Target', '#passengers'], ['Value', '2']], pages: [] });
    expect(await shown('Reading the departures')).toEqual({ heading: 'read', fields: [['Section', 'Saturday departures']], pages: [] });
    expect(await shown('Scrolling down')).toEqual({ heading: 'scroll', fields: [], pages: [] });
  });

  it('a delegate shows what it was asked — a browsing one also the pages it visited, in order — and "Open transcript" opens its transcript; a delegate call without a transcript has no way in', async () => {
    const w = await makeWorld();
    const browsing = { channel: 'delegate:web_browse', turn_id: 21 };
    const searching = { channel: 'delegate:web_search', turn_id: 22 };
    const subtask = { channel: 'delegate:subtask', turn_id: 23 };
    await w.settled(
      call(6301, 'web_browse', 'Comparing ferry fares', { instructions: 'Find the cheapest Gozo crossing' }, {
        delegate: browsing,
        pages: [
          { title: 'Ferry fares', url: 'https://ferries.example/fares' },
          { title: 'Fast ferry', url: 'https://fastferry.example/' },
        ],
      }),
      call(6302, 'web_search', 'Searching for ferry strikes', { instructions: 'Search for Gozo ferry strikes this week' }, { delegate: searching }),
      call(6303, 'subtask', 'Messaging the booking subtask', { action: 'message', subtask: 'booking', message: 'Also check the return times' }, { delegate: subtask }),
      call(6304, 'code_agent', 'Fixing the fare parser', { instructions: 'Refactor the fare parser' }),
    );

    const browse = await open(rowNamed(w.el, 'Comparing ferry fares'));
    expect(fields(browse)).toEqual([['Instructions', 'Find the cheapest Gozo crossing']]);
    expect(pages(browse)).toEqual([
      ['Ferry fares', 'https://ferries.example/fares'],
      ['Fast ferry', 'https://fastferry.example/'],
    ]);
    const search = await open(rowNamed(w.el, 'Searching for ferry strikes'));
    expect(fields(search)).toEqual([['Instructions', 'Search for Gozo ferry strikes this week']]);
    const message = await open(rowNamed(w.el, 'Messaging the booking subtask'));
    expect(fields(message)).toEqual([['Message', 'Also check the return times']]);
    const unlinked = await open(rowNamed(w.el, 'Fixing the fare parser'));
    expect(fields(unlinked)).toEqual([['Instructions', 'Refactor the fare parser']]);
    expect(transcriptButtons(unlinked)).toHaveLength(0);

    for (const [d, ref] of [[browse, browsing], [search, searching], [message, subtask]] as const) {
      const buttons = transcriptButtons(d);
      expect(buttons).toHaveLength(1);
      buttons[0]!.click();
      expect(w.session.panelDelegate).toEqual(ref);
    }
  });

  it('an email call is headed by its action and lists what it was given in a fixed order, skipping the empty; any other call is headed by its action and shows every other argument, a nested one as JSON', async () => {
    const w = await makeWorld();
    await w.settled(
      call(6401, 'email', 'Sending the ferry details', {
        action: 'send', attachments: ['/home/user/ticket.pdf'], body: 'See you Saturday.', subject: 'Ferry on Saturday', cc: '', to: 'sam@example.com', date_from: null,
      }),
      call(6402, '_mcp_tracker_get_issue', 'Reading issue 7', { action: 'get', repo: 'acme/ferries', number: 7, filters: { labels: ['bug'], state: 'open' } }),
      call(6403, 'calendar', 'Adding the ferry', { action: 'create_event', title: 'Ferry to Gozo', start: '2026-01-03T09:00' }),
    );

    const email = await open(rowNamed(w.el, 'Sending the ferry details'));
    expect(heading(email)).toBe('send');
    expect(fields(email)).toEqual([
      ['To', 'sam@example.com'],
      ['Subject', 'Ferry on Saturday'],
      ['Body', 'See you Saturday.'],
      ['Attachments', '/home/user/ticket.pdf'],
    ]);

    const mcp = await open(rowNamed(w.el, 'Reading issue 7'));
    expect(heading(mcp)).toBe('get');
    expect(fields(mcp)).toEqual([
      ['Repo', 'acme/ferries'],
      ['Number', '7'],
      ['Filters', JSON.stringify({ labels: ['bug'], state: 'open' }, null, 2)],
    ]);
    expect([...mcp.querySelectorAll('dd')].map((dd) => dd.classList.contains('call-detail__code'))).toEqual([false, false, true]);

    const builtIn = await open(rowNamed(w.el, 'Adding the ferry'));
    expect(heading(builtIn)).toBe('create_event');
    expect(fields(builtIn)).toEqual([['Title', 'Ferry to Gozo'], ['Start', '2026-01-03T09:00']]);
  });
});

describe('opening and reading a call', () => {
  it('each row is a button that opens its call under it and closes it again, saying which it is and naming what it opened; a settled call is read once however often it is opened, and each call on its own', async () => {
    const w = await makeWorld();
    await w.settled(
      call(6501, 'bash', 'Running the tests', { command: 'pytest -q' }),
      call(6502, 'read', 'Reading the lease', { source: 'lease-2026.pdf' }),
    );
    const tests = rowNamed(w.el, 'Running the tests');
    const lease = rowNamed(w.el, 'Reading the lease');
    expect([tests.tagName, lease.tagName]).toEqual(['BUTTON', 'BUTTON']);
    expect(tests.getAttribute('aria-expanded')).toBe('false');
    expect(controlled(tests)).toBeNull();

    const first = await open(tests);
    expect(code(first)).toBe('pytest -q');
    tests.click();
    await nextTick();
    expect(tests.getAttribute('aria-expanded')).toBe('false');
    expect(controlled(tests)).toBeNull();
    expect(code(await open(tests))).toBe('pytest -q');
    expect(backend.reads).toEqual([6501]);

    expect((await open(lease)).textContent).toContain('lease-2026.pdf');
    expect(lease.getAttribute('aria-controls')).not.toBe(tests.getAttribute('aria-controls'));
    expect(backend.reads).toEqual([6501, 6502]);
  });

  it('a running call opened while it runs keeps what it read on screen while its settled read loads, even when its step settles into its stored row meanwhile, and stays open there; the settled call is read once more, and no more, and shows when that read lands', async () => {
    const w = await makeWorld();
    await w.working();
    const params = { action: 'open', url: 'https://ferries.example/gozo' };
    const running = call(9301, 'browser', 'Opening the timetable', params, { state: 'started' });
    const done = call(9301, 'browser', 'Opening the timetable', params, {
      pages: [{ title: 'Gozo ferry timetable', url: 'https://ferries.example/gozo' }],
    });
    serve(running);
    await w.send(toolFrame(running));
    await openSteps(w.el);

    const live = rowNamed(w.el.querySelector('.activity')!, 'Opening the timetable');
    const whileRunning = await open(live);
    expect(heading(whileRunning)).toBe('open');
    expect(fields(whileRunning)).toEqual([['Url', 'https://ferries.example/gozo']]);
    expect(pages(whileRunning)).toEqual([]);
    expect(backend.reads).toEqual([9301]);

    // It settles while open: read again, and until that read lands the read
    // made while it ran stays on screen.
    backend.holding = true;
    serve(done);
    await w.send(toolFrame(done));
    expect(backend.reads).toEqual([9301, 9301]);
    const settling = detailOf(rowNamed(w.el.querySelector('.activity')!, 'Opening the timetable'));
    expect(settling.textContent).not.toContain('Loading…');
    expect(heading(settling)).toBe('open');
    expect(fields(settling)).toEqual([['Url', 'https://ferries.example/gozo']]);

    // Before that read lands, the refetch returns the row that made the call:
    // the call leaves the running block for that row's own block, still open,
    // still showing what it read, and waits on the same read.
    backend.thread = block([QUESTION, row(611, 'assistant', '', { tool_calls: serve(done) })], true);
    await w.send({ status: 'updated', turn_id: TURN, type: 'user' });
    expect(w.el.querySelector('.activity .call')).toBeNull();
    const stored = rowNamed(w.el.querySelector('.step:not(.activity)')!, 'Opening the timetable');
    expect(stored.getAttribute('aria-expanded')).toBe('true');
    const moved = detailOf(stored);
    expect(moved.textContent).not.toContain('Loading…');
    expect(heading(moved)).toBe('open');
    expect(fields(moved)).toEqual([['Url', 'https://ferries.example/gozo']]);
    expect(pages(moved)).toEqual([]);
    expect(backend.reads).toEqual([9301, 9301]);

    releaseHeld();
    backend.holding = false;
    await flushPromises();
    expect(pages(detailOf(stored))).toEqual([['Gozo ferry timetable', 'https://ferries.example/gozo']]);
    expect(backend.reads).toEqual([9301, 9301]);
  });

  it('a call being read says "Loading…"; a read that fails says so, names the call in the console, and is tried afresh when the row is opened again', async () => {
    const w = await makeWorld();
    await w.settled(
      call(6701, 'bash', 'Running the tests', { command: 'pytest -q' }),
      call(6702, 'bash', 'Checking the deposit', { command: 'python check_deposit.py' }),
    );

    backend.holding = true;
    const tests = rowNamed(w.el, 'Running the tests');
    tests.click();
    await flushPromises();
    expect(detailOf(tests).textContent?.trim()).toBe('Loading…');
    releaseHeld();
    backend.holding = false;
    await flushPromises();
    expect(code(detailOf(tests))).toBe('pytest -q');
    expect(detailOf(tests).textContent).not.toContain('Loading…');

    backend.failing.add(6702);
    const errors = vi.spyOn(console, 'error').mockImplementation(() => { /* asserted below */ });
    const deposit = rowNamed(w.el, 'Checking the deposit');
    const failed = await open(deposit);
    expect(failed.textContent?.trim()).toBe("Couldn't load this call.");
    expect(errors).toHaveBeenCalledWith(expect.stringContaining('6702'), expect.anything());

    backend.failing.delete(6702);
    deposit.click();
    await nextTick();
    const retried = await open(deposit);
    expect(code(retried)).toBe('python check_deposit.py');
    expect(retried.textContent).not.toContain("Couldn't load this call.");
    expect(backend.reads.filter((id) => id === 6702)).toHaveLength(2);
  });

  it('a read made while a call ran that answers after the settled call\'s read never replaces it, on screen or when the row is opened again', async () => {
    const w = await makeWorld();
    await w.working();
    const params = { action: 'open', url: 'https://ferries.example/gozo' };
    const running = call(9401, 'browser', 'Opening the timetable', params, { state: 'started' });
    const done = call(9401, 'browser', 'Opening the timetable', params, {
      pages: [{ title: 'Gozo ferry timetable', url: 'https://ferries.example/gozo' }],
    });
    serve(running);
    await w.send(toolFrame(running));
    await openSteps(w.el);

    // Opened while running, its read is slow; the call settles and is read
    // again before the first read answers.
    backend.holding = true;
    const live = rowNamed(w.el, 'Opening the timetable');
    live.click();
    await flushPromises();
    serve(done);
    await w.send(toolFrame(done));
    const [whileRunning, settled] = backend.held.splice(0);
    expect([whileRunning!.id, settled!.id]).toEqual([9401, 9401]);
    backend.holding = false;

    settled!.release();
    await flushPromises();
    const timetable = [['Gozo ferry timetable', 'https://ferries.example/gozo']];
    expect(pages(detailOf(live))).toEqual(timetable);
    whileRunning!.release();
    await flushPromises();
    expect(pages(detailOf(live))).toEqual(timetable);

    live.click();
    await nextTick();
    live.click();
    await nextTick();
    expect(pages(detailOf(live))).toEqual(timetable);
    await flushPromises();
    expect(pages(detailOf(live))).toEqual(timetable);
    expect(backend.reads).toEqual([9401, 9401]);
  });
});
