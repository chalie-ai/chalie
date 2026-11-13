// @vitest-environment happy-dom
/**
 * Delegate tool calls — feature spec.
 *
 * A tool call that ran as a subagent carries `delegate: {channel, turn_id}`.
 * Every call row is a real <button> that opens, under it, what the call did;
 * a delegate's row carries the Bot glyph, and its opened detail ends on an
 * "Open transcript" button that opens that subagent's transcript in the
 * thread panel. A call without a delegate opens to its detail with no way
 * into a transcript. The places a call is drawn are all covered, each through
 * the real component the conversation uses for it:
 *   - live, while the turn works          (the running block's calls)
 *   - settled, in its row's block         (a finished exchange)
 *   - settled mid-stream, in a block      (a working turn's stored rows)
 *
 * Real components, real Pinia session store, real live act-trail feed, all
 * fresh modules per test so no opened row or read call outlives it. Only the
 * network boundary is stood in for: `fetch`, answering each call's own read
 * (GET /api/tool-calls/<id>). A click is asserted by what it does to the
 * session (`panelDelegate`), the thing the ThreadPanel reads, not by spying on
 * a method. The Bot icon is identified by rendering lucide's own Bot and
 * comparing glyphs, so a restyled class name does not break the spec but a
 * different icon does.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { nextTick } from 'vue';
import { flushPromises, mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { Bot } from '@lucide/vue';
import { ConfigType } from '@chalie/shared';
import type { DelegateRef } from '@chalie/shared';
import type { LiveToolPill } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock, ToolCallDetail } from '../../api/conversation';

const REF = { channel: 'delegate:web_search', turn_id: 12 };
const TURN = 301;

/** The backend at the network boundary: each call as its own read returns it. */
const served = new Map<number, ToolCallDetail>();

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

async function fakeBackend(input: RequestInfo | URL): Promise<Response> {
  const url = new URL(String(input), 'http://chalie.test');
  const match = /^\/api\/tool-calls\/(\d+)$/.exec(url.pathname);
  const call = match ? served.get(Number(match[1])) : undefined;
  return call ? json({ success: true, result: call }) : json({ error: 'Tool call not found' }, 404);
}

/** Serves a call's read: what it was asked, the stored result the model read
 *  back (empty while it runs) and the delegate turn it spawned, if any. */
function serve(id: number, tool_name: string, params: Record<string, unknown>, state: ToolCallDetail['state'], delegate: DelegateRef | null): void {
  served.set(id, {
    id, tool_name, params, summary: '', state, delegate, pages: [],
    created_at: '2026-01-01T00:00:00Z',
    ended_at: state === 'started' ? null : '2026-01-01T00:00:01Z',
    result: state === 'started' ? '' : `[${tool_name}(status=success)]\n{"answer":"Opens at nine."}\n[end:${tool_name}]`,
  });
}

// The modules under test, imported fresh for each test.
let TurnView: typeof import('./TurnView.vue').default;
let ActivityLine: typeof import('./ActivityLine.vue').default;
let useSessionStore: typeof import('../../stores/session').useSessionStore;
let trail: typeof import('../../utils/liveActTrail');

let mounted: VueWrapper[] = [];

function track<T extends VueWrapper>(wrapper: T): T {
  mounted.push(wrapper);
  return wrapper;
}

beforeEach(async () => {
  // Which rows are open, and each call as read, are module state: fresh
  // modules give every test its own.
  vi.resetModules();
  served.clear();
  vi.stubGlobal('fetch', fakeBackend);
  const pinia = await import('pinia');
  pinia.setActivePinia(pinia.createPinia());
  ({ default: TurnView } = await import('./TurnView.vue'));
  ({ default: ActivityLine } = await import('./ActivityLine.vue'));
  ({ useSessionStore } = await import('../../stores/session'));
  trail = await import('../../utils/liveActTrail');
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  trail.clearAll(ConfigType.USER);
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

/** The inner glyph markup lucide draws for the Bot icon. */
const BOT_GLYPH = mount(Bot).element.innerHTML;

function hasBotIcon(el: Element): boolean {
  return [...el.querySelectorAll('svg')].some((svg) => svg.innerHTML === BOT_GLYPH);
}

/** Every <button> in the wrapper that carries the Bot glyph — the delegate rows. */
function delegateButtons(wrapper: VueWrapper): HTMLButtonElement[] {
  return [...wrapper.element.querySelectorAll('button')].filter(hasBotIcon);
}

/** A row button is named by what it shows (its visible text) — never by a
 *  label that replaces the visible text. */
function expectNamedByContent(button: HTMLElement, visible: string): void {
  expect(button.tagName).toBe('BUTTON');
  expect(button.getAttribute('type')).toBe('button');
  expect(button.textContent ?? '').toContain(visible);
  const label = button.getAttribute('aria-label');
  if (label != null) expect(label).toContain(visible);
}

/** Every <button> under `root` whose visible text mentions `label`. */
function buttonsMentioning(root: { element: Element } | Element, label: string): HTMLButtonElement[] {
  const el = root instanceof Element ? root : root.element;
  return [...el.querySelectorAll('button')].filter((b) => (b.textContent ?? '').includes(label));
}

/** The way into a transcript, wherever it is drawn. */
const transcriptButtons = (root: { element: Element } | Element) => buttonsMentioning(root, 'Open transcript');

/** Each step's summary toggle — the button that folds and unfolds its calls. */
function summaryToggles(wrapper: VueWrapper): HTMLButtonElement[] {
  return [...wrapper.element.querySelectorAll('.step__summary')].map((s) => s.closest('button')!);
}

/** Opens every step's call list by its summary toggle, as a person would. */
async function openCalls(wrapper: VueWrapper): Promise<void> {
  for (const t of summaryToggles(wrapper)) if (t.getAttribute('aria-expanded') === 'false') t.click();
  await nextTick();
}

/** A person's click: it can only land on what is reachable, never inside a
 *  folded (inert) list. */
function press(el: HTMLElement): void {
  expect(el.closest('[inert]')).toBeNull();
  el.click();
}

/** What a row opened: the element its `aria-controls` names. */
function detailOf(row: HTMLElement): HTMLElement {
  const id = row.getAttribute('aria-controls');
  const root = row.getRootNode() as ParentNode;
  const detail = [...root.querySelectorAll<HTMLElement>('[id]')].find((el) => el.id === id);
  expect(detail).toBeDefined();
  return detail!;
}

/** Opens a row as a person would and waits for its read to land. */
async function openRow(row: HTMLElement): Promise<HTMLElement> {
  press(row);
  await flushPromises();
  expect(row.getAttribute('aria-expanded')).toBe('true');
  return detailOf(row);
}

function livePill(over: Partial<LiveToolPill>): LiveToolPill {
  return {
    id: '31', name: 'web_search', params: null, summary: 'museum hours',
    ok: false, resolved: false, delegate: null, ...over,
  };
}

/** The running block's call rows, in the order they ran. */
function liveCalls(wrapper: VueWrapper): HTMLElement[] {
  return [...(wrapper.element as Element).querySelectorAll<HTMLElement>('.activity .call')];
}

type ToolCalls = NonNullable<ConversationMessage['tool_calls']>;

const DELEGATE_CALL: ToolCalls[number] = {
  id: 51, tool_name: 'web_search', params: null, summary: 'museum hours', state: 'done', ended_at: null, delegate: REF,
};
const PLAIN_CALL: ToolCalls[number] = {
  id: 52, tool_name: 'calendar', params: null, summary: 'todays agenda', state: 'done', ended_at: null, delegate: null,
};

/** The reads of the two stored calls above. */
function serveStoredCalls(): void {
  serve(51, 'web_search', { instructions: 'Find the museum opening hours' }, 'done', REF);
  serve(52, 'calendar', { range: 'today' }, 'done', null);
}

/** The assistant row carrying `toolCalls` — a row whose work was tools. */
function callsMessage(toolCalls: ToolCalls): ConversationMessage {
  return {
    id: '3011', role: 'assistant', content: '', timestamp: '2026-01-01 00:00:01',
    day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: TURN, tool_calls: toolCalls,
  };
}

/** The settled assistant answer that closes the exchange: the same shape as
 *  `callsMessage`'s row, with real content and its own id. */
const ANSWER: ConversationMessage = {
  id: '3012', role: 'assistant', content: 'The museum opens at nine.',
  timestamp: '2026-01-01 00:00:02', day: '2026-01-01', created_at: '2026-01-01T00:00:02Z',
  turn_id: TURN, settled: true,
};

function turnWith(working: boolean, toolCalls: ToolCalls, answer?: ConversationMessage): ConversationTurnBlock {
  const messages: ConversationMessage[] = [
    { id: '3010', role: 'user', content: 'When does the museum open?', timestamp: '2026-01-01 00:00:00', day: '2026-01-01', created_at: '2026-01-01T00:00:00Z', turn_id: TURN },
    { ...callsMessage(toolCalls), content: 'Let me look that up.' },
    ...(answer ? [answer] : []),
  ];
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

describe('live calls while the turn works', () => {
  it('a delegate call on the running line is a row button named by its visible content, with the Bot icon, running or finished; opened, it shows "Open transcript", which opens that subagent\'s transcript', async () => {
    serve(31, 'web_search', { instructions: 'Find the museum opening hours' }, 'started', REF);
    serve(32, 'web_search', { instructions: 'Check the museum holiday hours' }, 'done', { channel: 'delegate:web_search', turn_id: 13 });
    serve(33, 'web_search', { instructions: 'Find the ticket prices' }, 'started', { channel: 'delegate:web_search', turn_id: 14 });
    const wrapper = track(mount(ActivityLine, {
      props: {
        pills: [
          livePill({ id: '31', delegate: REF }),
          livePill({ id: '32', name: 'web_search', resolved: true, ok: true, delegate: { channel: 'delegate:web_search', turn_id: 13 } }),
          // A call with no summary line is still a delegate row.
          livePill({ id: '33', name: 'web_search', summary: '', delegate: { channel: 'delegate:web_search', turn_id: 14 } }),
        ],
      },
    }));
    const session = useSessionStore();

    const rows = liveCalls(wrapper);
    expect(rows).toHaveLength(3);
    expect(delegateButtons(wrapper)).toEqual(rows);
    // One sentence per row: 31 and 32 read by their summary, 33 — with no
    // summary — by its name. Nothing is open yet, so there is no way into a
    // transcript.
    expectNamedByContent(rows[0]!, 'museum hours');
    expectNamedByContent(rows[1]!, 'museum hours');
    expectNamedByContent(rows[2]!, 'web_search');
    expect(rows.map((r) => r.getAttribute('aria-expanded'))).toEqual(['false', 'false', 'false']);
    expect(transcriptButtons(wrapper)).toHaveLength(0);

    await openCalls(wrapper);
    const [first] = transcriptButtons(await openRow(rows[0]!));
    press(first!);
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();

    // The finished row opens its own, different, transcript; so does the row
    // with no summary.
    const [second] = transcriptButtons(await openRow(rows[1]!));
    press(second!);
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 13 });
    const [third] = transcriptButtons(await openRow(rows[2]!));
    press(third!);
    expect(session.panelDelegate).toEqual({ channel: 'delegate:web_search', turn_id: 14 });
  });

  it('a live call without a delegate is a row button that opens to what it did, with no Bot icon and no way into a transcript', async () => {
    serve(40, 'calendar', { range: 'today' }, 'started', null);
    serve(41, 'weather', { city: 'Valletta' }, 'error', null);
    const wrapper = track(mount(ActivityLine, {
      props: {
        pills: [
          livePill({ id: '40', name: 'calendar', summary: 'todays agenda' }),
          livePill({ id: '41', name: 'weather', summary: '', resolved: true, ok: false }),
        ],
      },
    }));
    const session = useSessionStore();

    // One row per call, in order: 40 by its summary, 41 — with no summary — by
    // its name. Each is a button, without the Bot glyph.
    const rows = liveCalls(wrapper);
    expect(rows.map((r) => r.textContent?.trim())).toEqual(['todays agenda', 'weather']);
    expectNamedByContent(rows[0]!, 'todays agenda');
    expectNamedByContent(rows[1]!, 'weather');
    expect(delegateButtons(wrapper)).toHaveLength(0);

    // Opened, each shows what it was asked, and nothing opens a transcript.
    await openCalls(wrapper);
    expect((await openRow(rows[0]!)).textContent).toContain('today');
    expect((await openRow(rows[1]!)).textContent).toContain('Valletta');
    expect(transcriptButtons(wrapper)).toHaveLength(0);
    expect(session.panelDelegate).toBeNull();
  });

  it('the live feed of a working turn gives an open delegate row its "Open transcript" once its link lands, and keeps it through completion', async () => {
    // Each call's read, as the backend answers while it runs: no child turn yet.
    serve(31, 'web_search', { instructions: 'Find the museum opening hours' }, 'started', null);
    serve(40, 'calendar', { range: 'today' }, 'started', null);
    trail.startLiveTool(ConfigType.USER, TURN, 31, 'web_search', 'museum hours');
    trail.startLiveTool(ConfigType.USER, TURN, 40, 'calendar', 'todays agenda');
    const wrapper = track(mount(TurnView, {
      props: { block: { ...turnWith(true, []), messages: turnWith(true, []).messages.slice(0, 1) }, type: ConfigType.USER },
    }));

    await openCalls(wrapper);
    const [delegateRow, plainRow] = liveCalls(wrapper);
    const delegateDetail = await openRow(delegateRow!);
    const plainDetail = await openRow(plainRow!);
    expect(transcriptButtons(wrapper)).toHaveLength(0);

    // The link lands on the live call: its open detail gains the way in; the
    // plain call's does not.
    trail.setLiveToolDelegate(ConfigType.USER, TURN, 31, REF);
    await nextTick();
    expect(transcriptButtons(wrapper)).toHaveLength(1);
    expect(transcriptButtons(delegateDetail)).toHaveLength(1);
    expect(transcriptButtons(plainDetail)).toHaveLength(0);

    // Both finish; the delegate's settled read names its child turn.
    serve(31, 'web_search', { instructions: 'Find the museum opening hours' }, 'done', REF);
    serve(40, 'calendar', { range: 'today' }, 'done', null);
    trail.finishLiveTool(ConfigType.USER, TURN, 31, true);
    trail.finishLiveTool(ConfigType.USER, TURN, 40, true);
    await flushPromises();
    const [pill] = transcriptButtons(wrapper);
    expect(transcriptButtons(wrapper)).toHaveLength(1);
    expect(detailOf(liveCalls(wrapper)[0]!).contains(pill!)).toBe(true);
    press(pill!);
    expect(useSessionStore().panelDelegate).toEqual(REF);
  });
});

describe('settled step blocks', () => {
  it('a delegate call is a row button with the Bot icon whose opened detail opens its transcript, beside a plain call\'s row button whose detail does not', async () => {
    serveStoredCalls();
    // The settled exchange's work stands above its answer as a block: the
    // row that carried the calls, then the settled answer that closes it.
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL, PLAIN_CALL], ANSWER), type: ConfigType.USER },
    }));
    const session = useSessionStore();

    const rows = [...(wrapper.element as Element).querySelectorAll<HTMLElement>('.call')];
    expect(rows).toHaveLength(2);
    // One sentence per row, both buttons; only the delegate's carries the Bot.
    expectNamedByContent(rows[0]!, 'museum hours');
    expectNamedByContent(rows[1]!, 'todays agenda');
    expect(delegateButtons(wrapper)).toEqual([rows[0]]);
    expect(transcriptButtons(wrapper)).toHaveLength(0);

    await openCalls(wrapper);
    const delegateDetail = await openRow(rows[0]!);
    const plainDetail = await openRow(rows[1]!);
    expect(delegateDetail.textContent).toContain('Find the museum opening hours');
    expect(transcriptButtons(plainDetail)).toHaveLength(0);
    const [open] = transcriptButtons(delegateDetail);
    expect(transcriptButtons(wrapper)).toEqual([open]);
    press(open!);
    expect(session.panelDelegate).toEqual(REF);
    expect(session.panelThreadId).toBeNull();
  });

  it('a block\'s delegate button is out of the tab order while its calls are folded, and in it once they are opened', async () => {
    serveStoredCalls();
    const wrapper = track(mount(TurnView, {
      props: { block: turnWith(false, [DELEGATE_CALL], ANSWER), type: ConfigType.USER },
    }));
    const [button] = delegateButtons(wrapper);
    expect(button).toBeDefined();
    // Folded by default: the button sits in an inert list a person cannot reach.
    expect(button!.closest('[inert]')).not.toBeNull();

    await openCalls(wrapper);
    expect(summaryToggles(wrapper).map((t) => t.getAttribute('aria-expanded'))).toEqual(['true']);
    expect(button!.closest('[inert]')).toBeNull();
  });
});

describe('settled calls on a working turn (its step blocks)', () => {
  it('a step row that only called tools draws both its calls as row buttons in its block; only the delegate\'s opened detail opens its transcript', async () => {
    serveStoredCalls();
    // A working turn whose provider call asked for tools and said nothing: its
    // block is that call's work alone, above the running block.
    const turn = turnWith(true, [DELEGATE_CALL, PLAIN_CALL]);
    turn.messages[1] = { ...turn.messages[1]!, content: '' };
    const wrapper = track(mount(TurnView, { props: { block: turn, type: ConfigType.USER } }));
    const session = useSessionStore();
    expect(wrapper.find('[data-transcript-row-id="3011"] .speech-form__text').exists()).toBe(false);
    const rows = [...(wrapper.element as Element).querySelectorAll<HTMLElement>('.step:not(.activity) .call')];
    expect(rows).toHaveLength(2);
    for (const r of rows) expect(r.closest('[inert]')).not.toBeNull();
    await openCalls(wrapper);
    for (const r of rows) expect(r.closest('[inert]')).toBeNull();
    expectNamedByContent(rows[0]!, 'museum hours');
    expectNamedByContent(rows[1]!, 'todays agenda');
    expect(delegateButtons(wrapper)).toEqual([rows[0]]);

    expect(transcriptButtons(await openRow(rows[1]!))).toHaveLength(0);
    const [open] = transcriptButtons(await openRow(rows[0]!));
    expect(transcriptButtons(wrapper)).toEqual([open]);
    press(open!);
    expect(session.panelDelegate).toEqual(REF);
  });
});
