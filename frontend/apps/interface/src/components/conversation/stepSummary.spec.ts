// @vitest-environment happy-dom
/**
 * Step summaries — feature spec for what a step's calls read as.
 *
 * A step's calls fold under one muted phrase of what they did — "Wrote 2 files
 * and read 1 file" — counted by the arguments each call acted on, with how
 * many failed beside it, over one row per call: the kind's icon, the call's
 * summary, then whether it succeeded. The spec mounts real exchanges through
 * the real TurnView (ThinkingAndCalls, ToolCallList, ChalieBubble,
 * ActivityLine, Pinia) and asserts what a person reads:
 *
 *   - what a row said stands before its block, the running row's too;
 *   - each step's phrase, counted per kind by distinct target (or by call when
 *     the kind has none), MCP tools as one kind, an unknown tool by its name;
 *   - a failure is counted on the folded head;
 *   - plan calls are the plan card's: a step that only updated the plan and
 *     did not think draws no block and no row, while the card stands and the
 *     exchange's duration still counts that step's time;
 *   - each row's icon is its kind's, every row is a button that opens it, and
 *     the status mark closes the row.
 *
 * The last two tests run the running step end to end: live frames go through
 * the real dispatcher into the real act-trail and onto a TurnView the real
 * surface registry mounted, and the turn settles through a real refetch. Only
 * the network boundary is stood in for (`fetch`, answering the thread read),
 * plus the session hooks `session.init()` would register, as in the sibling
 * dispatcher specs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import type { WsPushEvent } from '@chalie/shared';
import TurnView from './TurnView.vue';
import { clearAll, startLiveTool } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock, PlanTask, ToolCallChip } from '../../api/conversation';

const TURN = 41;

type Params = Record<string, unknown> | null;

function chip(
  id: number,
  tool_name: string,
  params: Params,
  summary: string,
  state: ToolCallChip['state'] = 'done',
  delegate: ToolCallChip['delegate'] = null,
): ToolCallChip {
  return { id, tool_name, params, summary, state, ended_at: null, delegate };
}

function row(
  turnId: number,
  id: number,
  role: ConversationMessage['role'],
  content: string,
  extra: Partial<ConversationMessage> = {},
): ConversationMessage {
  return {
    id: String(id), role, content, timestamp: '2026-01-01 09:00:00', day: '2026-01-01',
    created_at: '2026-01-01T09:00:00Z', turn_id: turnId, ...extra,
  };
}

function block(turnId: number, messages: ConversationMessage[], working: boolean): ConversationTurnBlock {
  return {
    turn_id: turnId, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

/** A stored step of turn 41: no text, only its calls. */
const step = (id: number, calls: ToolCallChip[]) => row(TURN, id, 'assistant', '', { tool_calls: calls });

let mounted: VueWrapper[] = [];

function mountTurn(b: ConversationTurnBlock): VueWrapper {
  const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });
  mounted.push(wrapper);
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
});

afterEach(() => {
  // Which call lists are open outlives a mount, so fold any a test opened.
  for (const w of mounted) {
    for (const s of w.findAll('.step__summary')) {
      const toggle = s.element.closest('button')!;
      if (toggle.getAttribute('aria-expanded') === 'true') toggle.click();
    }
    w.unmount();
  }
  mounted = [];
  clearAll(ConfigType.USER);
  document.body.innerHTML = '';
});

describe('a step reads as what it said, then what its calls did', () => {
  it('a row that said something reads its words before its block — a stored row and the running one alike', () => {
    startLiveTool(ConfigType.USER, TURN, 4020, 'move', 'Moving drafts/a.md', { current_path: 'drafts/a.md' });
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 400, 'user', 'Tidy my notes folder'),
      row(TURN, 401, 'assistant', 'Found three drafts.', {
        tool_calls: [chip(4010, 'search_files', { pattern: 'draft' }, 'Finding drafts')],
      }),
      row(TURN, 402, 'assistant', 'Moving the drafts first.', {
        tool_calls: [chip(4020, 'move', { current_path: 'drafts/a.md' }, 'Moving drafts/a.md', 'started')],
      }),
    ], true));

    const order = wrapper.findAll('.msg-row--chalie').map((r) =>
      r.findAll('.speech-form__text, .step').map((p) => (p.classes('step') ? `${p.classes('activity') ? 'live' : 'step'}:${p.find('.step__summary').text()}` : p.text())),
    );
    expect(order).toEqual([
      ['Found three drafts.', 'step:Searched files'],
      ['Moving the drafts first.', 'live:Moved 1 item'],
    ]);
  });

  it('names each step\'s calls by kind, counting the distinct things each kind acted on, or its calls when a kind names none', () => {
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 409, 'user', 'Sort out the project notes and the shopping list'),
      step(410, [
        chip(4100, 'write_file', { path: 'notes/a.md' }, 'Writing notes/a.md'),
        chip(4101, 'write_file', { path: 'notes/b.md' }, 'Writing notes/b.md'),
        chip(4102, 'read', { source: 'notes/c.md' }, 'Reading notes/c.md'),
      ]),
      step(411, [
        chip(4110, 'write_file', { path: 'todo.md' }, 'Writing todo.md'),
        chip(4111, 'write_file', { path: 'todo.md' }, 'Fixing a typo in todo.md'),
      ]),
      step(412, [
        chip(4120, 'bash', { command: 'ls notes' }, 'Listing notes'),
        chip(4121, 'bash', { command: 'ls notes' }, 'Listing notes again'),
        chip(4122, 'bash', { command: 'ls notes' }, 'Listing notes once more'),
      ]),
      step(413, [
        chip(4130, 'browser', { action: 'open', url: 'https://shop.example/a' }, 'Opening the shop'),
        chip(4131, 'browser', { action: 'open', url: 'https://shop.example/b' }, 'Opening the basket'),
        chip(4132, 'browser', { action: 'click', target: '#checkout' }, 'Clicking checkout'),
      ]),
      step(414, [
        chip(4140, 'list', { action: 'add', list: 'groceries', items: ['milk', 'eggs'] }, 'Adding milk and eggs'),
        chip(4141, 'list', { action: 'add', list: 'groceries', items: ['eggs', 'bread'] }, 'Adding eggs and bread'),
      ]),
      step(415, [
        chip(4150, 'bash', { command: 'make' }, 'Building'),
        chip(4151, 'write_file', { path: 'out.txt' }, 'Writing out.txt'),
        chip(4152, 'read', { source: 'in.txt' }, 'Reading in.txt'),
      ]),
      step(416, [
        chip(4160, 'bash', { command: 'make test' }, 'Testing'),
        chip(4161, 'write_file', { path: 'report.txt' }, 'Writing report.txt'),
        chip(4162, 'read', { source: 'report.txt' }, 'Reading report.txt'),
        chip(4163, 'web_fetch', { url: 'https://docs.example' }, 'Fetching the docs'),
      ]),
      step(417, [
        chip(4170, '_mcp_github_search_issues', { query: 'crash' }, 'Searching issues'),
        chip(4171, '_mcp_notion_get_page', { page: 'Roadmap' }, 'Reading the roadmap page'),
      ]),
      step(418, [chip(4180, 'frobnicate', {}, '')]),
      row(TURN, 4190, 'assistant', 'All sorted.', { settled: true }),
    ], false));

    expect(wrapper.findAll('.step__summary').map((s) => s.text())).toEqual([
      'Wrote 2 files and read 1 file',
      'Wrote 1 file',
      'Ran 3 commands',
      'Opened 2 pages and clicked 1 element',
      'Added 3 items',
      // Three kinds are each named; a fourth folds into "1 more".
      'Ran 1 command, wrote 1 file and read 1 file',
      'Ran 1 command, wrote 1 file, read 1 file and 1 more',
      'Used 2 MCP tools',
      'Used frobnicate',
    ]);
  });

  it('counts a failed call on the folded head, where it reads without opening the list; a step with none says nothing of failure', () => {
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 419, 'user', 'Compare the two leases'),
      step(420, [
        chip(4200, 'read', { source: 'lease-2025.pdf' }, 'Reading the 2025 lease'),
        chip(4201, 'read', { source: 'lease-2026.pdf' }, 'Reading the 2026 lease', 'error'),
      ]),
      step(421, [chip(4210, 'read', { source: 'lease-2024.pdf' }, 'Reading the 2024 lease')]),
      row(TURN, 422, 'assistant', 'The 2026 lease would not open.', { settled: true }),
    ], false));

    const [failing, clean] = wrapper.findAll('.step');
    const failed = failing!.find('.step__failed');
    expect(failed.text()).toBe('1 failed');
    expect(failing!.find('.step__summary').element.closest('button')!.getAttribute('aria-expanded')).toBe('false');
    expect(failed.element.closest('[inert]')).toBeNull();
    expect(clean!.find('.step__failed').exists()).toBe(false);
  });

  it('leaves plan calls to the plan card: a step that only updated the plan and did not think draws no block, and a running step shows only its other calls', () => {
    const plan: PlanTask[] = [
      { number: 1, title: 'Check the ferry', task: 'Find ferry times', status: 'in_progress' },
      { number: 2, title: 'Book', task: 'Book the crossing', status: 'todo' },
    ];
    startLiveTool(ConfigType.USER, TURN, 4310, 'web_search', 'Searching ferry times', null);
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 429, 'user', 'Get me to Gozo on Saturday', { plan }),
      step(430, [
        chip(4300, 'plan', { action: 'create' }, 'Setting up the plan'),
        chip(4301, 'plan', { action: 'update' }, 'Starting the ferry check'),
      ]),
      step(431, [
        chip(4311, 'plan', { action: 'update' }, 'Marking the ferry check started'),
        chip(4310, 'web_search', null, 'Searching ferry times', 'started'),
      ]),
    ], true));

    expect(wrapper.find('.plan-card').exists()).toBe(true);
    expect(wrapper.findAll('.step:not(.activity)')).toHaveLength(0);
    const live = wrapper.find('.activity');
    expect(live.find('.step__summary').text()).toBe('Delegated 1 web search');
    expect(live.findAll('.call__summary').map((c) => c.text())).toEqual(['Searching ferry times']);
    expect(wrapper.text()).not.toMatch(/Setting up the plan|Starting the ferry check|Marking the ferry check/);
  });

  it('a settled step that only updated the plan, without thinking or text, takes no row and leaves the lead to the answer, while the exchange\'s duration still counts its time', () => {
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 449, 'user', 'Plan my Saturday'),
      { ...step(450, [chip(4500, 'plan', { action: 'create' }, 'Setting up the plan')]), created_at: '2026-01-01T09:00:10Z' },
      row(TURN, 451, 'assistant', 'Your Saturday is planned.', { settled: true, created_at: '2026-01-01T09:00:30Z' }),
    ], false));

    // The question, the answer, and the footer that closes it — nothing for
    // the plan step, and the answer stands as Chalie's first row.
    const rows = wrapper.findAll('.msg-row');
    expect(rows.map((r) => r.text().includes('Your Saturday is planned.') ? 'answer' : r.classes().find((c) => /^msg-row--(user|footer)$/.test(c)))).toEqual([
      'msg-row--user', 'answer', 'msg-row--footer',
    ]);
    expect(rows[1]!.classes()).toContain('msg-row--lead');
    expect(wrapper.text()).not.toContain('Setting up the plan');
    // From the question at 09:00:00 to the answer at 09:00:30.
    expect(wrapper.find('.speech-form__meta').text()).toMatch(/· 30s/);
  });

  it('opens each call row on its kind\'s icon and closes it on its outcome; every row, a delegate\'s too, is a button that opens it', () => {
    const wrapper = mountTurn(block(TURN, [
      row(TURN, 439, 'user', 'Find the bug report and fix it'),
      step(440, [
        chip(4400, 'write_file', { path: 'fix.py' }, 'Writing fix.py'),
        chip(4401, 'read', { source: 'missing.log' }, 'Reading missing.log', 'error'),
        chip(4402, 'bash', { command: 'pytest' }, 'Running the tests'),
        chip(4403, 'browser', { action: 'open', url: 'https://bugs.example/7' }, 'Opening the report'),
        chip(4404, '_mcp_github_get_issue', { number: 7 }, 'Reading issue 7'),
        chip(4405, 'frobnicate', {}, 'Frobnicating'),
        chip(4406, 'web_search', null, 'Searching the error', 'done', { channel: 'delegate:web_search', turn_id: 77 }),
      ]),
      row(TURN, 441, 'assistant', 'Fixed.', { settled: true }),
    ], false));

    const rows = wrapper.findAll('.call').map(({ element: c }) => {
      const icon = c.firstElementChild!;
      const mark = c.lastElementChild!;
      return [
        c.tagName,
        icon.classList.contains('call__icon') ? [...icon.classList].find((k) => /^lucide-(?!icon$).*(?<!-icon)$/.test(k)) : 'no icon first',
        c.querySelector('.call__summary')!.textContent,
        mark.classList.contains('call__mark') ? `${mark.getAttribute('role')}:${mark.getAttribute('aria-label')}` : 'no mark last',
      ].join(' ');
    });
    expect(rows).toEqual([
      'BUTTON lucide-file-plus Writing fix.py img:succeeded',
      'BUTTON lucide-file-text Reading missing.log img:failed',
      'BUTTON lucide-square-terminal Running the tests img:succeeded',
      'BUTTON lucide-globe Opening the report img:succeeded',
      'BUTTON lucide-plug Reading issue 7 img:succeeded',
      'BUTTON lucide-wrench Frobnicating img:succeeded',
      'BUTTON lucide-bot Searching the error img:succeeded',
    ]);
  });
});

// ── The running step, end to end through the dispatcher ─────────────────────

const LIVE_TURN = 9;
const STARTED_AT = '2026-01-01T09:00:00Z';
const QUESTION = row(LIVE_TURN, 90, 'user', 'Check my lease against the deposit rules');

/** The backend at the network boundary: the block a thread read returns. */
const backend = { user: null as ConversationTurnBlock | null };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  if (url.pathname !== `/api/threads/${LIVE_TURN}` || !backend.user) return json({ error: 'unrouted' }, 404);
  return json({ success: true, result: backend.user });
}

function toolFrame(id: number, tool_name: string, state: string, summary: string, params?: Params): WsPushEvent {
  return { tool_name, id, turn_id: LIVE_TURN, type: 'user', state, summary, ...(params === undefined ? {} : { params }) } as unknown as WsPushEvent;
}

let teardown: (() => void) | null = null;

/** A user spine, as the app registers it, showing turn 9 working on its
 *  question — the dispatcher, act-trail, surface registry and TurnView all
 *  fresh modules, so nothing a test opened outlives it. */
async function makeWorld() {
  vi.resetModules();
  backend.user = null;
  vi.stubGlobal('fetch', fakeBackend);
  const pinia = await import('pinia');
  pinia.setActivePinia(pinia.createPinia());
  const turnDom = await import('../../utils/turnDom');
  const liveActTrail = await import('../../utils/liveActTrail');
  const { dispatchDrift, registerSessionHooks } = await import('../../utils/driftDispatcher');
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
  turnDom.upsertTurnToSurfaces(block(LIVE_TURN, [QUESTION], true), 'user');
  turnDom.setTurnWorking(LIVE_TURN, 'user', true);
  teardown = () => {
    turnDom.clearSurfaceContainer(el);
    liveActTrail.clearAll('user');
  };
  return {
    el,
    liveActTrail,
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
});

const summaryToggle = (s: Element) => s.querySelector('.step__summary')!.closest('button')!;
const callsBody = (s: Element) => s.querySelector('.calls')!.closest('.trace-body')!;
const callRows = (s: Element) => [...s.querySelectorAll('.call__summary')].map((c) => c.textContent);

describe('the running step, fed by live frames', () => {
  it('counts its calls by the arguments their frames carried, and a call whose frame carried none on its own', async () => {
    const w = await makeWorld();

    await w.send(toolFrame(9101, 'write_file', 'started', 'Writing notes/plan.md', { path: 'notes/plan.md' }));
    await w.send(toolFrame(9102, 'write_file', 'started', 'Saving notes/plan.md again', { path: 'notes/plan.md' }));
    await w.send(toolFrame(9103, 'write_file', 'started', 'Writing notes/todo.md', { path: 'notes/todo.md' }));
    await w.send(toolFrame(9104, 'write_file', 'started', 'Writing a scratch file'));

    expect(w.liveActTrail.liveTrailsFor('user', LIVE_TURN).flatMap((t) => t.pills.map((p) => p.params))).toEqual([
      { path: 'notes/plan.md' }, { path: 'notes/plan.md' }, { path: 'notes/todo.md' }, null,
    ]);
    expect(w.el.querySelector('.activity .step__summary')!.textContent).toBe('Wrote 3 files');
  });

  it('keeps its call list open, once opened, as more calls arrive and as the step settles into its stored row', async () => {
    const w = await makeWorld();
    const live = () => w.el.querySelector('.activity')!;

    await w.send(toolFrame(9201, 'read', 'started', 'Reading the lease', { source: 'lease.pdf' }));
    expect(summaryToggle(live()).getAttribute('aria-expanded')).toBe('false');
    summaryToggle(live()).click();
    await flushPromises();
    expect(summaryToggle(live()).getAttribute('aria-expanded')).toBe('true');

    await w.send(toolFrame(9202, 'web_fetch', 'started', 'Fetching the deposit rules', { url: 'https://rules.example/deposit' }));
    expect(summaryToggle(live()).getAttribute('aria-expanded')).toBe('true');
    expect(callsBody(live()).hasAttribute('inert')).toBe(false);
    expect(callRows(live())).toEqual(['Reading the lease', 'Fetching the deposit rules']);

    // Both finish, and the refetch returns the row that made them: the calls
    // leave the running block for that row's own block, a new one.
    await w.send(toolFrame(9201, 'read', 'done', 'Reading the lease', { source: 'lease.pdf' }));
    await w.send(toolFrame(9202, 'web_fetch', 'done', 'Fetching the deposit rules', { url: 'https://rules.example/deposit' }));
    const stepRow = row(LIVE_TURN, 91, 'assistant', '', {
      tool_calls: [
        chip(9201, 'read', { source: 'lease.pdf' }, 'Reading the lease'),
        chip(9202, 'web_fetch', { url: 'https://rules.example/deposit' }, 'Fetching the deposit rules'),
      ],
    });
    backend.user = block(LIVE_TURN, [QUESTION, stepRow], true);
    await w.send({ status: 'updated', turn_id: LIVE_TURN, type: 'user' });

    expect(live().querySelector('.step__summary')).toBeNull();
    const stored = () => w.el.querySelector('.step:not(.activity)')!;
    expect(summaryToggle(stored()).getAttribute('aria-expanded')).toBe('true');
    expect(callsBody(stored()).hasAttribute('inert')).toBe(false);
    expect(callRows(stored())).toEqual(['Reading the lease', 'Fetching the deposit rules']);

    // The turn settles with its answer; the step's list is still open.
    backend.user = block(LIVE_TURN, [
      QUESTION, stepRow, row(LIVE_TURN, 92, 'assistant', 'The deposit clause is within the rules.', { settled: true }),
    ], false);
    await w.send({ state: 'completed', turn_id: LIVE_TURN, started_at: STARTED_AT, type: 'user' });

    expect(w.el.querySelector('.activity')).toBeNull();
    expect(w.el.textContent).toContain('The deposit clause is within the rules.');
    expect(summaryToggle(stored()).getAttribute('aria-expanded')).toBe('true');
    expect(callsBody(stored()).hasAttribute('inert')).toBe(false);
  });
});
