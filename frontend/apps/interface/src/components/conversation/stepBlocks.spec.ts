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
 *   - one block per row of work, the settled answer's too: "Thought for Xs"
 *     only when the row thought, then one row per call, then what it said; the
 *     user row's turn-zero calls are its memory seed and draw no block; a
 *     thought with nothing to unfold has no chevron;
 *   - every block's thinking is folded and out of the tab order until pressed,
 *     and opens onto the row's reasoning; its calls are always in view;
 *   - the same block stands live and settled, its thinking open across the
 *     settle;
 *   - the running block names its calls with no clock on them, is the running
 *     row's own block once a refetch returns that row, and a permission ask is
 *     named and answered on its head, beside Stop;
 *   - speech is offered for the settled answer only — never for a step's work
 *     or an interim row's prose, which reads in its own block, outside the fold.
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
  return { id, tool_name, summary, state: 'done', ended_at: null, delegate: null };
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

const STEP_LABELS = [
  'Thought for 9s · Closing recall task, Loading web search tools',
  'Thought for 4s · Searching Malta winter rentals',
  'Thought for 3s · Searching Spain winter rentals',
  'Thought for 11s · Searching Albania winter rentals, Albania winter weather, Albania cost of living',
  'Thought for 6s · Reading the Albania cost page, Starting final synthesis of the answer',
  'Thought for 2s · Drafting the comparison',
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

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  clearAll(ConfigType.USER);
  vi.useRealTimers();
  document.body.innerHTML = '';
});

/** What a block's work reads as: its head, then its calls joined with ", ". */
function workText(block: ReturnType<VueWrapper['find']>): string {
  const head = block.find('.step__label');
  const calls = block.findAll('.call__summary').map((c) => c.text()).join(', ');
  return [head.exists() ? head.text() : '', calls].filter(Boolean).join(' · ');
}

/** The rendered exchange read top to bottom, one entry per row: a block of
 *  Chalie's as its work, then what it said. */
function layout(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.msg-row').map((r) => {
    if (r.find('.speech-form__timestamp').exists()) return 'footer';
    if (r.find('.user-message').exists()) return 'user';
    if (!r.find('.speech-form--chalie').exists()) return 'other';
    const work = r.find('.activity').exists() ? 'live' : r.find('.step').exists() ? `step:${workText(r)}` : '';
    const text = r.find('.speech-form__text');
    if (!text.exists()) return work;
    return work ? `${work} | ${text.text()}` : `bubble:${text.text()}`;
  });
}

/** Every stored row's work, top to bottom — the running block's aside. */
const steps = (wrapper: VueWrapper) => wrapper.findAll('.step:not(.activity)');

describe('a researched exchange reads as one block per row of work', () => {
  it('draws no block for the user row\'s seed calls, then each row\'s block, then the answer\'s — its thought over the answer — and the footer', () => {
    const wrapper = mountTurn(settledTurn());

    expect(layout(wrapper)).toEqual([
      'user',
      ...STEP_LABELS.map((l) => `step:${l}`),
      'step:Thought for 33s | Malta is the easiest quiet winter.',
      'footer',
    ]);
  });

  it('keeps every block\'s thinking folded and out of the tab order until pressed, its calls in view; pressed, it opens onto the row\'s reasoning, and a second press folds it again', async () => {
    const wrapper = mountTurn(settledTurn());
    const folds = () => wrapper.findAll('.step:not(.activity) button.fold');

    expect(folds().map((f) => f.attributes('aria-expanded'))).toEqual(Array(7).fill('false'));
    expect(wrapper.findAll('.step:not(.activity) .trace-body').every((b) => b.attributes('inert') !== undefined)).toBe(true);

    const block = steps(wrapper)[ROW_33]!;
    const body = block.find('.trace-body');
    // The calls read while the thinking is folded, outside the fold.
    expect(block.findAll('.call__summary').map((c) => c.text())).toEqual([
      'Searching Albania winter rentals',
      'Albania winter weather',
      'Albania cost of living',
    ]);
    expect(block.find('.calls').element.closest('[inert]')).toBeNull();

    await block.find('button.fold').trigger('click');

    expect(block.find('button.fold').attributes('aria-expanded')).toBe('true');
    expect(body.classes()).toContain('trace-body--open');
    expect(body.attributes('inert')).toBeUndefined();
    expect(body.findAll('pre.step__trace').map((t) => t.text())).toEqual([
      'Albania is the unknown.',
      'Three angles: rent, weather, cost.',
    ]);
    expect(body.find('.call').exists()).toBe(false);
    // Only the pressed block opened.
    expect(folds().filter((f) => f.attributes('aria-expanded') === 'true')).toHaveLength(1);

    await block.find('button.fold').trigger('click');

    expect(block.find('button.fold').attributes('aria-expanded')).toBe('false');
    expect(body.classes()).not.toContain('trace-body--open');
    expect(body.attributes('inert')).toBeDefined();
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
      'step:Thought · web_search',
      'step:Reading the visa page',
      'step:Thought for 33s | Malta is the easiest quiet winter.',
      'footer',
    ]);
  });

  it('gives a thought with nothing to unfold no chevron and no fold to open, a thought that left reasoning both, and a row that never thought no head', () => {
    const [bare, reasoned, unthought] = steps(mountTurn(thinTurn()));

    expect(bare!.find('.step__label').text()).toBe('Thought for 5s');
    expect(bare!.find('.fold-chev').exists()).toBe(false);
    expect(bare!.find('.trace-body').exists()).toBe(false);
    expect(bare!.find('[aria-expanded]').exists()).toBe(false);

    expect(reasoned!.find('.fold-chev').exists()).toBe(true);
    expect(reasoned!.find('.trace-body').text()).toContain('Check the dates first.');

    expect(unthought!.find('.step__head').exists()).toBe(false);
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
    expect(layout(wrapper).slice(-2)).toEqual(['step:Thought for 33s | Malta is the easiest quiet winter.', 'footer']);
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

    vi.advanceTimersByTime(12_000);
    await wrapper.vm.$nextTick();

    expect(workText(live)).toBe('Thinking… · Searching Albania winter rentals, Reading the cost page');
    expect(live.findAll('.call').map((c) => [c.find('.call__summary').text(), c.find('.call__mark').attributes('aria-label')])).toEqual([
      ['Searching Albania winter rentals', 'running'],
      ['Reading the cost page', 'succeeded'],
    ]);
    // Twelve seconds in, nothing on the block counts them.
    expect(live.text()).not.toMatch(/\d/);
  });

  it('becomes the running row\'s own block once a refetch returns that row — its reasoning and live call in one block, the same block as before', async () => {
    startLiveTool(ConfigType.USER, TURN, 19, 'plan', 'Drafting the comparison');
    const before = STEP_ROWS.slice(0, -1);
    const wrapper = mountTurn(turn([USER_ROW, ...before], true));
    const element = wrapper.find('.activity').element;
    expect(workText(wrapper.find('.activity'))).toBe('Thinking… · Drafting the comparison');

    // The refetch returns the running row: its reasoning, its call still running.
    const running = stepRow(57, { traces: ['Table first.'], duration_ms: 2400, tokens: 60 }, [
      { ...call(19, 'plan', 'Drafting the comparison'), state: 'started' },
    ]);
    await wrapper.setProps({ block: turn([USER_ROW, ...before, running], true) });

    expect(layout(wrapper)).toEqual(['user', ...STEP_LABELS.slice(0, -1).map((l) => `step:${l}`), 'live']);
    expect(wrapper.find('.activity').element).toBe(element);
    expect(workText(wrapper.find('.activity'))).toBe('Thought for 2s · Drafting the comparison');
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

    // Every interim reply reads in its own block under its work, outside the fold.
    expect(layout(wrapper)).toEqual([
      'user',
      ...STEP_ROWS.map((r, i) => `step:${STEP_LABELS[i]} | Working on ${r.id}.`),
      'step:Thought for 33s | Malta is the easiest quiet winter.',
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
