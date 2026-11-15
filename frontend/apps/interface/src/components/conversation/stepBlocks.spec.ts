// @vitest-environment happy-dom
/**
 * Step blocks — feature spec on a real-shaped exchange.
 *
 * The payload is the shape the API returns for a researched turn: the user row
 * carries the framework's turn-zero calls (recall, plan); six assistant rows
 * each carry their own reasoning (`thinking`) and tool calls with an empty
 * `content`; one settled row closes the exchange with the answer's HTML and
 * its own reasoning. The spec mounts that through the real TurnView — real
 * ThinkingAndCalls, ToolCallList, ChalieBubble, BubbleFooter, ActivityLine,
 * Pinia and event bus — and asserts what a person reads and can press:
 *
 *   - what a row said comes first, then its block: one muted head line —
 *     "Thought for Xs" only when the row thought, then what its calls did
 *     ("Delegated 3 web searches") — over one row per call; the user row's
 *     turn-zero calls are its memory seed and draw no block; plan calls are the
 *     plan card's, never a block's; a thought with nothing to unfold has no
 *     chevron;
 *   - a block's thinking and its calls are each folded and out of the tab order
 *     until pressed: the thought opens onto the row's reasoning, the summary
 *     onto its calls;
 *   - the same block stands live and settled, its thinking open across the
 *     settle;
 *   - the running block names its calls with no clock on them, is the running
 *     row's own block once a refetch returns that row, and a permission ask is
 *     named and answered on its head, beside Stop;
 *   - speech is offered for the settled answer only — never for a step's work
 *     or an interim row's prose, which reads in its own row, outside the fold.
 *
 * The REST transport the permission answer goes out on is the one stand-in:
 * the spec reads the decision the block sent off it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import { on } from '../../composables/useEventBus';
import { usePermissionsStore } from '../../stores/permissions';
import { useVoiceTranscriptsStore } from '../../stores/voiceTranscripts';
import { clearAll, finishLiveTool, startLiveTool } from '../../utils/liveActTrail';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';
import type { PendingPermission } from '../../api/policies';

const { respondMock } = vi.hoisted(() => ({ respondMock: vi.fn() }));
vi.mock('../../api/policies', () => ({
  policies: {
    pending: () => Promise.resolve([]),
    respond: (payload: unknown) => respondMock(payload),
  },
}));

const TURN = 3;
const SPEAK_BUTTON = 'button.speech-form__act-btn--speak';

type ToolCalls = NonNullable<ConversationMessage['tool_calls']>;

function call(id: number, tool_name: string, summary: string): ToolCalls[number] {
  return { id, tool_name, params: null, summary, state: 'done', ended_at: null, delegate: null };
}

/** One stored assistant row of turn 3: no text, its own reasoning and calls. */
function stepRow(
  id: number,
  thinking: ConversationMessage['thinking'],
  toolCalls: ToolCalls,
  extra: Partial<ConversationMessage> = {},
): ConversationMessage {
  return {
    id: String(id), role: 'assistant', content: '', timestamp: '2026-01-01 10:00:00', day: '2026-01-01',
    created_at: '2026-01-01T10:00:30Z', turn_id: TURN, settled: false,
    thinking, tool_calls: toolCalls, ...extra,
  };
}

const USER_ROW: ConversationMessage = {
  id: '29', role: 'user', content: 'Compare Malta, Spain and Albania for a quiet winter month.',
  timestamp: '2026-01-01 10:00:00', day: '2026-01-01', created_at: '2026-01-01T10:00:00Z', turn_id: TURN,
  tool_calls: [
    call(1, 'recall', ''),
    call(2, 'recall', 'Checking memory for Malta/Spain/Albania context'),
    call(3, 'plan', 'Setting up research plan…'),
  ],
};

const STEP_ROWS: ConversationMessage[] = [
  stepRow(30, { traces: ['Recall is done; load search next.'], duration_ms: 9000, tokens: 310 }, [
    call(10, 'plan', 'Closing recall task'),
    call(11, 'load_tools', 'Loading web search tools'),
  ]),
  stepRow(31, { traces: ['Start with Malta.'], duration_ms: 4200, tokens: 120 }, [
    call(12, 'web_search', 'Searching Malta winter rentals'),
  ]),
  stepRow(32, { traces: ['Spain next.'], duration_ms: 3100, tokens: 98 }, [
    call(13, 'web_search', 'Searching Spain winter rentals'),
  ]),
  stepRow(33, { traces: ['Albania is the unknown.', 'Three angles: rent, weather, cost.'], duration_ms: 10810, tokens: 402 }, [
    call(14, 'web_search', 'Searching Albania winter rentals'),
    call(15, 'web_search', 'Albania winter weather'),
    call(16, 'web_search', 'Albania cost of living'),
  ]),
  stepRow(56, { traces: ['Enough material.'], duration_ms: 6400, tokens: 150 }, [
    call(17, 'read', 'Reading the Albania cost page'),
    call(18, 'plan', 'Starting final synthesis of the answer'),
  ]),
  stepRow(57, { traces: ['Table first.'], duration_ms: 2400, tokens: 60 }, [
    call(19, 'plan', 'Drafting the comparison'),
  ]),
];

const ANSWER_ROW: ConversationMessage = {
  id: '58', role: 'assistant', content: '<p>Malta is the easiest quiet winter.</p>',
  timestamp: '2026-01-01 10:02:48', day: '2026-01-01', created_at: '2026-01-01T10:02:48Z', turn_id: TURN,
  settled: true, thinking: { traces: ['Rank by effort.'], duration_ms: 33023, tokens: 900 },
};

/** Each step row's block as `workText` reads it. The plan calls (10, 18, 19)
 *  are in none of them: row 57 called only the plan, so its block is its
 *  thought alone. */
const STEP_LABELS = [
  'Thought for 9s · Used load_tools › Loading web search tools',
  'Thought for 4s · Delegated 1 web search › Searching Malta winter rentals',
  'Thought for 3s · Delegated 1 web search › Searching Spain winter rentals',
  'Thought for 11s · Delegated 3 web searches › Searching Albania winter rentals, Albania winter weather, Albania cost of living',
  'Thought for 6s · Read 1 file › Reading the Albania cost page',
  'Thought for 2s',
];

/** Row 33's block: the fourth, the user row's seed drawing none. */
const ROW_33 = 3;

function turn(messages: ConversationMessage[], working: boolean): ConversationTurnBlock {
  return {
    turn_id: TURN, gist: null, preview: messages[0]!.content, last_activity_at: null,
    working, duration_ms: 0, messages, type: 'user',
  };
}

const settledTurn = (): ConversationTurnBlock => turn([USER_ROW, ...STEP_ROWS, ANSWER_ROW], false);
const workingTurn = (): ConversationTurnBlock => turn([USER_ROW, ...STEP_ROWS], true);

let mounted: VueWrapper[] = [];

function mountTurn(block: ConversationTurnBlock): VueWrapper {
  const wrapper = mount(TurnView, { props: { block, type: ConfigType.USER, fullThread: false } });
  mounted.push(wrapper);
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
  respondMock.mockReset();
  respondMock.mockResolvedValue(undefined);
});

/** Each block's summary toggle — the button that folds and unfolds its calls. */
const summaryToggle = (block: ReturnType<VueWrapper['find']>) => block.find('.step__summary').element.closest('button')!;

afterEach(() => {
  // Which call lists are open outlives a mount (it is how a list stays open
  // as its step settles), so fold every list a test opened before unmounting.
  for (const w of mounted) {
    for (const s of w.findAll('.step__summary')) {
      const toggle = s.element.closest('button')!;
      if (toggle.getAttribute('aria-expanded') === 'true') toggle.click();
    }
    w.unmount();
  }
  mounted = [];
  clearAll(ConfigType.USER);
  vi.useRealTimers();
  document.body.innerHTML = '';
});

/** What a block's work reads as: its head line — the thought, then what its
 *  calls did, then how many failed — then its calls' rows joined with ", ". */
function workText(block: ReturnType<VueWrapper['find']>): string {
  const head = ['.step__label', '.step__summary', '.step__failed']
    .map((sel) => block.find(sel))
    .filter((el) => el.exists())
    .map((el) => el.text())
    .join(' · ');
  const calls = block.findAll('.call__summary').map((c) => c.text()).join(', ');
  return [head, calls].filter(Boolean).join(' › ');
}

/** The rendered exchange read top to bottom, one entry per row. A row of
 *  Chalie's reads its parts in the order they stand on the page — what it
 *  said, its block (`step:` and its work, or `live` for the running one) —
 *  joined with " | "; a row of prose alone reads `bubble:` and its text. */
function layout(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.msg-row').map((r) => {
    if (r.find('.speech-form__timestamp').exists()) return 'footer';
    if (r.find('.user-message').exists()) return 'user';
    if (!r.find('.speech-form--chalie').exists()) return 'other';
    const parts = r.findAll('.speech-form__text, .step');
    if (!parts.some((p) => p.classes('step'))) return parts.map((p) => `bubble:${p.text()}`).join(' | ');
    return parts.map((p) => {
      if (!p.classes('step')) return p.text();
      return p.classes('activity') ? 'live' : `step:${workText(p)}`;
    }).join(' | ');
  });
}

/** Every stored row's work, top to bottom — the running block's aside. */
const steps = (wrapper: VueWrapper) => wrapper.findAll('.step:not(.activity)');

describe('a researched exchange reads as one block per row of work', () => {
  it('draws no block for the user row\'s seed calls, then each row\'s block, then the answer — its thought under it — and the footer', () => {
    const wrapper = mountTurn(settledTurn());

    expect(layout(wrapper)).toEqual([
      'user',
      ...STEP_LABELS.map((l) => `step:${l}`),
      'Malta is the easiest quiet winter. | step:Thought for 33s',
      'footer',
    ]);
  });

  it('keeps every block\'s thinking and calls folded and out of the tab order until pressed; the thought opens onto the row\'s reasoning, the summary onto its calls, and a second press folds each again', async () => {
    const wrapper = mountTurn(settledTurn());
    const folds = () => wrapper.findAll('.step:not(.activity) button.fold');

    // Seven thoughts, and a summary for each of the five rows with calls to show.
    expect(folds().map((f) => f.attributes('aria-expanded'))).toEqual(Array(12).fill('false'));
    expect(wrapper.findAll('.step:not(.activity) .trace-body').every((b) => b.attributes('inert') !== undefined)).toBe(true);

    const block = steps(wrapper)[ROW_33]!;
    const thinking = block.find('.trace-body');
    const callsBody = () => block.find('.calls').element.closest('.trace-body')!;
    const thoughtFold = block.find('button.fold');
    const toggle = summaryToggle(block);
    // The calls are on the page, but folded where no one can reach them.
    expect(callsBody().hasAttribute('inert')).toBe(true);

    await thoughtFold.trigger('click');

    expect(thoughtFold.attributes('aria-expanded')).toBe('true');
    expect(thinking.classes()).toContain('trace-body--open');
    expect(thinking.attributes('inert')).toBeUndefined();
    expect(thinking.findAll('pre.step__trace').map((t) => t.text())).toEqual([
      'Albania is the unknown.',
      'Three angles: rent, weather, cost.',
    ]);
    expect(thinking.find('.call').exists()).toBe(false);
    // Opening the thought leaves the calls folded, and only the pressed block opened.
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(callsBody().hasAttribute('inert')).toBe(true);
    expect(folds().filter((f) => f.attributes('aria-expanded') === 'true')).toHaveLength(1);

    await thoughtFold.trigger('click');

    expect(thoughtFold.attributes('aria-expanded')).toBe('false');
    expect(thinking.classes()).not.toContain('trace-body--open');
    expect(thinking.attributes('inert')).toBeDefined();

    toggle.click();
    await wrapper.vm.$nextTick();

    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(callsBody().classList).toContain('trace-body--open');
    expect(callsBody().hasAttribute('inert')).toBe(false);
    expect(block.findAll('.call__summary').map((c) => c.text())).toEqual([
      'Searching Albania winter rentals',
      'Albania winter weather',
      'Albania cost of living',
    ]);
    expect(thoughtFold.attributes('aria-expanded')).toBe('false');
    expect(folds().filter((f) => f.attributes('aria-expanded') === 'true')).toHaveLength(1);

    toggle.click();
    await wrapper.vm.$nextTick();

    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(callsBody().classList).not.toContain('trace-body--open');
    expect(callsBody().hasAttribute('inert')).toBe(true);
  });
});

describe('a block says only what its row did', () => {
  /** Thin work: a timed thought that left no trace, untimed reasoning beside a
   *  call with a blank summary, and a call with no thinking at all. */
  const thinTurn = (): ConversationTurnBlock => turn([
    USER_ROW,
    stepRow(40, { traces: [], duration_ms: 5000, tokens: 0 }, []),
    stepRow(41, { traces: ['Check the dates first.'], duration_ms: 0, tokens: 12 }, [call(20, 'web_search', '  ')]),
    stepRow(42, undefined, [call(21, 'read', 'Reading the visa page')]),
    ANSWER_ROW,
  ], false);

  it('reads "Thought" for reasoning that was never timed, names a call by its tool when its summary is blank, and shows a row that never thought as its calls alone', () => {
    expect(layout(mountTurn(thinTurn()))).toEqual([
      'user',
      'step:Thought for 5s',
      'step:Thought · Delegated 1 web search › web_search',
      'step:Read 1 file › Reading the visa page',
      'Malta is the easiest quiet winter. | step:Thought for 33s',
      'footer',
    ]);
  });

  it('gives a thought with nothing to unfold no chevron and no fold to open, a thought that left reasoning both, and a row that never thought its summary alone on the head', () => {
    const [bare, reasoned, unthought] = steps(mountTurn(thinTurn()));

    expect(bare!.find('.step__label').text()).toBe('Thought for 5s');
    expect(bare!.find('.fold-chev').exists()).toBe(false);
    expect(bare!.find('.trace-body').exists()).toBe(false);
    expect(bare!.find('[aria-expanded]').exists()).toBe(false);

    expect(reasoned!.find('button.fold .fold-chev').exists()).toBe(true);
    expect(reasoned!.find('.trace-body').text()).toContain('Check the dates first.');

    // No thought: no thought label and nothing to separate from it — the head
    // is what the calls did, which unfolds onto them.
    expect(unthought!.find('.step__label').exists()).toBe(false);
    expect(unthought!.find('.step__sep').exists()).toBe(false);
    expect(unthought!.findAll('.step__head button.fold').map((b) => b.find('.step__summary').exists())).toEqual([true]);
    expect(unthought!.find('.step__summary').text()).toBe('Read 1 file');
    expect(unthought!.find('.call__summary').text()).toBe('Reading the visa page');
  });
});

describe('one step block, live and settled', () => {
  it('keeps the same open block across the settle — nothing remounts, only the answer\'s block and the footer arrive', async () => {
    const wrapper = mountTurn(workingTurn());
    const block = steps(wrapper)[ROW_33]!;
    await block.find('button.fold').trigger('click');
    const element = block.element;

    await wrapper.setProps({ block: settledTurn() });

    const after = steps(wrapper)[ROW_33]!;
    expect(after.element).toBe(element);
    expect(after.find('button.fold').attributes('aria-expanded')).toBe('true');
    expect(wrapper.find('.activity').exists()).toBe(false);
    expect(layout(wrapper).slice(-2)).toEqual(['Malta is the easiest quiet winter. | step:Thought for 33s', 'footer']);
  });
});

describe('the running block', () => {
  /** A pending ask parked on a turn, as the backend lists it. */
  function ask(request_id: string, turn_id: number, action_id: string, summary: string): PendingPermission {
    return { request_id, action_id, summary, asked_at: new Date().toISOString(), origin: { type: 'user', turn_id, forked: false } };
  }

  /** The block's own controls, by what they say (the fold is its thinking). */
  const controls = (live: ReturnType<VueWrapper['find']>): string[] =>
    live.findAll('button:not(.fold)').map((b) => b.text());

  it('names its running and finished calls by what they do, with no clock on any of them', async () => {
    vi.useFakeTimers();
    startLiveTool(ConfigType.USER, TURN, 501, 'web_search', 'Searching Albania winter rentals');
    startLiveTool(ConfigType.USER, TURN, 502, 'read', 'Reading the cost page');
    finishLiveTool(ConfigType.USER, TURN, 502, true);
    const wrapper = mountTurn(workingTurn());
    const live = wrapper.find('.activity');
    summaryToggle(live).click();
    await wrapper.vm.$nextTick();
    const atStart = live.text();

    vi.advanceTimersByTime(12_000);
    await wrapper.vm.$nextTick();

    expect(workText(live)).toBe('Thinking… · Delegated 1 web search and read 1 file › Searching Albania winter rentals, Reading the cost page');
    expect(live.findAll('.call').map((c) => [c.find('.call__summary').text(), c.find('.call__mark').attributes('aria-label')])).toEqual([
      ['Searching Albania winter rentals', 'running'],
      ['Reading the cost page', 'succeeded'],
    ]);
    // Twelve seconds in, nothing on the block has moved, and no call row counts anything.
    expect(live.text()).toBe(atStart);
    for (const c of live.findAll('.call')) expect(c.text()).not.toMatch(/\d/);
  });

  it('becomes the running row\'s own block once a refetch returns that row — its reasoning and live call in one block, the same block as before', async () => {
    // A plan update runs beside a search; the plan is the plan card's, never the block's.
    startLiveTool(ConfigType.USER, TURN, 19, 'plan', 'Drafting the comparison');
    startLiveTool(ConfigType.USER, TURN, 59, 'web_search', 'Checking the exchange rate');
    const before = STEP_ROWS.slice(0, -1);
    const wrapper = mountTurn(turn([USER_ROW, ...before], true));
    const element = wrapper.find('.activity').element;
    expect(workText(wrapper.find('.activity'))).toBe('Thinking… · Delegated 1 web search › Checking the exchange rate');

    // The refetch returns the running row: its reasoning, its calls still running.
    const running = stepRow(57, { traces: ['Table first.'], duration_ms: 2400, tokens: 60 }, [
      { ...call(19, 'plan', 'Drafting the comparison'), state: 'started' },
      { ...call(59, 'web_search', 'Checking the exchange rate'), state: 'started' },
    ]);
    await wrapper.setProps({ block: turn([USER_ROW, ...before, running], true) });

    expect(layout(wrapper)).toEqual(['user', ...STEP_LABELS.slice(0, -1).map((l) => `step:${l}`), 'live']);
    expect(wrapper.find('.activity').element).toBe(element);
    expect(workText(wrapper.find('.activity'))).toBe('Thought for 2s · Delegated 1 web search › Checking the exchange rate');
    expect(wrapper.find('.activity').text()).not.toContain('Drafting the comparison');
  });

  it('parked on its own turn\'s ask, names it and waits for you beside Stop — Allow and Deny each send their own answer and hand the head back', async () => {
    const permissions = usePermissionsStore();
    // An ask parked on another turn is never this block's.
    permissions.enqueue(ask('r0', TURN + 1, 'pim', 'Read the inbox'));
    permissions.enqueue(ask('r1', TURN, 'search', 'Search the web for Albania winter rentals'));
    const wrapper = mountTurn(workingTurn());
    const live = () => wrapper.find('.activity');

    expect(live().find('.step__label').text()).toBe('Search the web for Albania winter rentals');
    expect(live().text()).toContain('waiting for you');
    expect(live().find('.activity__ask').text()).toBe('Web Search');
    expect(controls(live())).toEqual(['Deny', 'Allow', 'Stop']);

    await live().find('.activity__answer--allow').trigger('click');

    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(respondMock).toHaveBeenLastCalledWith({ request_id: 'r1', approved: true });
    expect(live().find('.step__label').text()).toBe('Thinking…');
    expect(live().text()).not.toContain('waiting for you');
    expect(controls(live())).toEqual(['Stop']);

    permissions.enqueue(ask('r2', TURN, 'search', 'Search the web for Spain winter rentals'));
    await wrapper.vm.$nextTick();
    await live().find('.activity__answer--deny').trigger('click');

    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(respondMock).toHaveBeenLastCalledWith({ request_id: 'r2', approved: false });
    expect(controls(live())).toEqual(['Stop']);
    // The other turn's ask was never answered from here.
    expect(permissions.queue.map((r) => r.request_id)).toEqual(['r0']);
  });
});

describe('speech is offered for the settled answer only', () => {
  const spoken: number[] = [];
  let unbind: () => void = () => {};

  beforeEach(() => {
    spoken.length = 0;
    unbind = on('chalie:speak-message', ({ transcriptId }) => spoken.push(transcriptId));
  });

  afterEach(() => unbind());

  it('puts one speaker button on the page, in the settled answer\'s footer, however many thoughts are open and interim replies show — and it speaks that answer', async () => {
    useVoiceTranscriptsStore().record(58, 'ready');
    const withProse = STEP_ROWS.map((r) => ({ ...r, content: `<p>Working on ${r.id}.</p>` }));
    const wrapper = mountTurn(turn([USER_ROW, ...withProse, ANSWER_ROW], false));
    for (const fold of steps(wrapper)) await fold.find('button.fold').trigger('click');

    // Every interim reply reads in its own row above its work, outside the fold.
    expect(layout(wrapper)).toEqual([
      'user',
      ...STEP_ROWS.map((r, i) => `Working on ${r.id}. | step:${STEP_LABELS[i]}`),
      'Malta is the easiest quiet winter. | step:Thought for 33s',
      'footer',
    ]);
    expect(wrapper.findAll('.step .speech-form__text')).toHaveLength(0);
    expect(wrapper.findAll(SPEAK_BUTTON)).toHaveLength(1);
    const footer = wrapper.findAll('.msg-row').find((r) => r.find('.speech-form__timestamp').exists())!;
    expect(footer.findAll(SPEAK_BUTTON)).toHaveLength(1);

    await footer.find(SPEAK_BUTTON).trigger('click');

    expect(spoken).toEqual([58]);
  });
});
