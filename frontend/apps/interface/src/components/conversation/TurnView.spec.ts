// @vitest-environment happy-dom
/**
 * TurnView — feature spec for the just-fixed live-act-trail visibility
 * contract (see TurnView.vue displayRows, ~line 79).
 *
 * The in-flight "thinking…" / live act-trail row must render ONLY when this
 * render is the authoritative live view of the turn:
 *   - working, NON-forked turn, spine   (fullThread=false) → renders
 *   - working, FORKED turn,     spine   (fullThread=false) → does NOT render
 *     (the thread pill in ConversationFeed carries the working indicator —
 *     duplicating it here was the regression this spec locks in)
 *   - working, FORKED turn,     thread panel (fullThread=true) → renders
 *
 * "Forked" is real production state: a block whose messages() contains a
 * `thread_message: true` row — TurnView derives `isForkedThread` directly
 * off the `block` prop (`computed(() => props.block.messages.some((m) =>
 * m.thread_message))`), no store/composable involved, so the block is just
 * mounted straight in, exactly as it's fetched off the wire.
 *
 * Real component tree throughout (TurnView → ActivityLine/BubbleFooter/
 * UserBubble/ChalieBubble), real Pinia, real `@chalie/shared` barrel — the
 * only thing given a DOM stand-in is happy-dom itself (this file opts in via
 * the docblock above; the suite's default `environment: 'node'` is
 * untouched elsewhere).
 *
 * The second half pins the exchange layout, in the order it happened: the
 * question, one block per assistant row that did work or spoke (the user row's
 * turn-zero calls are its memory seed and draw none) — its step block, then its
 * own prose, visible outside the fold — and a footer right under the row that
 * closes a segment — the timestamp, the segment's duration when it did work,
 * and actions. A still-working exchange keeps the same blocks and closes the
 * list with the live activity line.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock, PlanTask } from '../../api/conversation';
import {
  clearAll,
  clearLiveTurnsForToolCallsResolved,
  finishLiveTool,
  startLiveTool,
} from '../../utils/liveActTrail';

function msg(
  id: string,
  role: ConversationMessage['role'],
  content: string,
  turnId: number,
  threadMessage = false,
  toolCalls?: ConversationMessage['tool_calls'],
): ConversationMessage {
  return {
    id,
    role,
    content,
    timestamp: '2026-01-01 00:00:00',
    day: '2026-01-01',
    created_at: '2026-01-01T00:00:00Z',
    turn_id: turnId,
    ...(threadMessage ? { thread_message: true } : {}),
    ...(toolCalls ? { tool_calls: toolCalls } : {}),
  };
}

function block(turnId: number, messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: messages[0]?.content ?? '',
    last_activity_at: null,
    working: true,
    duration_ms: 0,
    messages,
    type: 'user',
  };
}

/** Any row, with the optional fields a stored row can carry. */
function row(
  id: string,
  role: ConversationMessage['role'],
  content: string,
  turnId: number,
  extra: Partial<ConversationMessage> = {},
): ConversationMessage {
  return { ...msg(id, role, content, turnId), ...extra };
}

type ToolCalls = NonNullable<ConversationMessage['tool_calls']>;

let nextCallId = 9000;

/** One finished call per name, as the backend sends a row's chips — each with
 *  its own tool_calls row id, none shared with a live pill. */
function calls(...names: string[]): ToolCalls {
  return names.map((tool_name) => ({
    id: nextCallId++, tool_name, summary: `ran ${tool_name}`, state: 'done' as const, ended_at: null, delegate: null,
  }));
}

/** A turn that is no longer working. */
function settledTurn(turnId: number, messages: ConversationMessage[]): ConversationTurnBlock {
  return { ...block(turnId, messages), working: false };
}

/** Three provider calls in one exchange: an interim reply, a silent step that
 *  also reasoned, and the settled answer — each with a call of its own. */
function threeCallTurn(): ConversationTurnBlock {
  const turnId = 210;
  return settledTurn(turnId, [
    row('2100', 'user', 'research the museum', turnId),
    row('2101', 'assistant', 'interim', turnId, { tool_calls: calls('web_search') }),
    row('2102', 'assistant', '', turnId, {
      tool_calls: calls('read'),
      thinking: { traces: ['which file has it?'], duration_ms: 2000, tokens: 12 },
    }),
    row('2103', 'assistant', 'final', turnId, { tool_calls: calls('memory_recall'), settled: true }),
  ]);
}

const COPY = 'button[aria-label="Copy message"]';
const SPEAK = 'button[aria-label="Read this message aloud"]';
const REPLY = 'button[aria-label="Reply in a thread"]';

/** The rendered turn read top to bottom, one entry per row: a user bubble, the
 *  live plan card, a footer (a timestamp and actions), or one of Chalie's step
 *  blocks — its work (`live` while it runs, else `step:<head · calls>`), then
 *  what it said, joined as `step:<work> | <text>`; a block that only spoke
 *  reads `bubble:<text>`. */
function layout(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.msg-row').map((r) => {
    if (r.classes('msg-row--plan')) return 'plan';
    if (r.find('.speech-form__timestamp').exists()) return 'footer';
    if (r.find('.user-message').exists()) return `user:${r.find('.user-message').attributes('data-user-text')}`;
    if (!r.find('.speech-form--chalie').exists()) return 'other';
    const work = r.find('.activity').exists() ? 'live' : r.find('.step').exists() ? `step:${workText(r)}` : '';
    const text = r.find('.speech-form__text');
    if (!text.exists()) return work;
    return work ? `${work} | ${text.text()}` : `bubble:${text.text()}`;
  });
}

/** A block's work as it reads: its "Thought for Xs" head, then its calls. */
function workText(block: ReturnType<VueWrapper['find']>): string {
  const head = block.find('.step__label');
  const calls = block.findAll('.call__summary').map((c) => c.text()).join(', ');
  return [head.exists() ? head.text() : '', calls].filter(Boolean).join(' · ');
}

/** The one sentence each tool-call row of a stored block shows (the summary,
 *  or the tool name when the summary is empty), one array per block, top to
 *  bottom. The running block is left out. */
function stepCalls(wrapper: VueWrapper): string[][] {
  return wrapper.findAll('.step:not(.activity)').map((s) => s.findAll('.call__summary').map((c) => c.text()));
}

/** The row that holds the exchange's timestamp, duration and actions. */
function footerOf(wrapper: VueWrapper): ReturnType<VueWrapper['find']> {
  return wrapper.findAll('.msg-row').find((r) => r.find('.speech-form__timestamp').exists())!;
}

beforeEach(() => {
  setActivePinia(createPinia());
});

// Live pills live in module state shared across mounts; reset it between tests.
afterEach(() => {
  clearAll(ConfigType.USER);
});

describe('live act-trail visibility — working turn on the spine', () => {
  it('renders the live-act row for a NON-forked working turn', () => {
    const turnId = 101;
    const b = block(turnId, [
      msg('1010', 'user', 'settle0 question', turnId),
      msg('1011', 'assistant', 'settle0 answer', turnId),
    ]);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.find('.activity').exists()).toBe(true);
    // settle0 rows always render regardless of the live-act guard.
    expect(wrapper.text()).toContain('settle0 question');
    expect(wrapper.text()).toContain('settle0 answer');
    // A non-forked block must NOT stamp data-forked on the root — this is
    // the Activity drawer's sole signal for which turns even qualify as
    // forked (see utils/threadActivity.ts's `[data-forked]` selector).
    expect(wrapper.attributes('data-forked')).toBeUndefined();
    // Working turns have nothing "complete" to timestamp — no footer yet.
    expect(wrapper.find('.speech-form__meta').exists()).toBe(false);
  });

  it('does NOT render the live-act row for a FORKED working turn (the thread pill owns that indicator)', () => {
    const turnId = 102;
    const b = block(turnId, [
      msg('1020', 'user', 'opener of the thread', turnId),
      msg('1021', 'assistant', 'settle0 reply', turnId),
      msg('1022', 'user', 'a reply inside the thread', turnId, true),
    ]);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.find('.activity').exists()).toBe(false);
    // Regression guard for the over-correction we reverted: settle0 rows
    // (non-thread_message) must still render inline on the spine even though
    // this turn is forked.
    expect(wrapper.text()).toContain('opener of the thread');
    expect(wrapper.text()).toContain('settle0 reply');
    // The thread continuation itself is spine-dropped (fullThread=false).
    expect(wrapper.text()).not.toContain('a reply inside the thread');
    // A forked block DOES stamp data-forked on the root.
    expect(wrapper.attributes('data-forked')).toBe('true');
    // The opener exchange is SETTLED (the thread continuation replied to it),
    // so its footer stays on the spine even while the fork works — exactly one
    // per turn_id, never flickering away when a continuation starts streaming.
    // Only a still-streaming exchange's own footer is withheld.
    expect(wrapper.findAll('.speech-form__meta')).toHaveLength(1);
  });

  it('a running call shows once — as its live pill while it runs, then as its stored chip once the pill clears', async () => {
    // One call, two sources: the live frame opens a pill keyed on the call's
    // tool_calls row id, and a refetch mid-call returns the same call as a
    // `started` chip carrying that id. An earlier finished call has no pill.
    const turnId = 104;
    const chip = (id: number, tool_name: string, state: ToolCalls[number]['state']): ToolCalls[number] => ({
      id, tool_name, summary: `ran ${tool_name}`, state, ended_at: null, delegate: null,
    });
    const rows = (running: ToolCalls[number]): ConversationMessage[] => [
      row('1040', 'user', 'find the museum hours', turnId),
      row('1041', 'assistant', 'let me look', turnId, { tool_calls: [chip(76, 'read', 'done')] }),
      row('1042', 'assistant', '', turnId, { tool_calls: [running] }),
    ];
    startLiveTool(ConfigType.USER, turnId, 77, 'web_search');

    const wrapper = mount(TurnView, {
      props: { block: block(turnId, rows(chip(77, 'web_search', 'started'))), type: ConfigType.USER, fullThread: false },
    });
    const chipNames = (): string[] => wrapper.findAll('.step:not(.activity) .call__summary').map((e) => e.text());
    const pillNames = (): string[] => wrapper.findAll('.activity .call__summary').map((e) => e.text());

    // While it runs: the live line alone shows it, named by its tool, and the
    // earlier step keeps its block — its stored line, then its interim reply.
    expect(pillNames()).toEqual(['web_search']);
    expect(wrapper.find('.activity .step__label').text()).toBe('Thinking…');
    expect(chipNames()).toEqual(['ran read']);
    expect(layout(wrapper)).toEqual(['user:find the museum hours', 'step:ran read | let me look', 'live']);

    // It finishes: the resolved pill still holds it until the refetch lands.
    finishLiveTool(ConfigType.USER, turnId, 77, true);
    await wrapper.vm.$nextTick();
    expect(pillNames()).toEqual(['web_search']);
    expect(chipNames()).toEqual(['ran read']);

    // The refetch lands the finished chip and drops the resolved pill: now the
    // stored step shows the call, once, and the live line moves on below it.
    clearLiveTurnsForToolCallsResolved(ConfigType.USER, turnId, true);
    await wrapper.setProps({ block: block(turnId, rows(chip(77, 'web_search', 'done'))) });
    expect(pillNames()).toEqual([]);
    expect(chipNames()).toEqual(['ran read', 'ran web_search']);
    expect(layout(wrapper)).toEqual([
      'user:find the museum hours', 'step:ran read | let me look', 'step:ran web_search', 'live',
    ]);
  });
});

describe('live act-trail visibility — working turn in the thread panel', () => {
  it('renders the live-act row for a FORKED working turn when fullThread=true, alongside its thread rows', () => {
    const turnId = 103;
    const b = block(turnId, [
      msg('1030', 'user', 'opener of the panel thread', turnId),
      msg('1031', 'assistant', 'settle0 panel reply', turnId),
      msg('1032', 'user', 'panel thread continuation', turnId, true),
    ]);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: true },
    });

    expect(wrapper.find('.activity').exists()).toBe(true);
    expect(wrapper.text()).toContain('opener of the panel thread');
    expect(wrapper.text()).toContain('settle0 panel reply');
    // fullThread renders the WHOLE thread, continuations included.
    expect(wrapper.text()).toContain('panel thread continuation');
    // Per-exchange footers: the SETTLED opener exchange keeps its footer even
    // while the thread works on the (still footerless) continuation — only the
    // in-progress exchange is suppressed. So exactly one footer, on the opener,
    // sitting after the settled reply and before the working continuation.
    expect(wrapper.findAll('.speech-form__meta')).toHaveLength(1);
    const order = wrapper.findAll('.msg-row').map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.text().includes('settle0 panel reply')
          ? 'settle0'
          : r.text().includes('panel thread continuation')
            ? 'continuation'
            : 'other',
    );
    expect(order.indexOf('footer')).toBeGreaterThan(order.indexOf('settle0'));
    expect(order.indexOf('footer')).toBeLessThan(order.indexOf('continuation'));
  });
});


// One exchange draws the question, one quiet step block per assistant row that
// did work (its thinking and calls folded behind it — the user row's seed
// calls draw none), each row's own prose as a bubble right after its line, and
// a footer under the row that closes it, everything in the order it happened.
// While the exchange still works, the same step blocks stand and the live
// activity line closes the list.
describe('tool-call placement — a block per row of work, the answer\'s too, then its footer, the live line last while working', () => {
  it('a call stored on the user\'s own row is the turn-zero memory seed: it draws no step block and shows nowhere in the exchange', () => {
    const turnId = 201;
    const b = settledTurn(turnId, [
      row('2010', 'user', 'user question with pre-turn tools', turnId, { tool_calls: calls('memory_recall') }),
      row('2011', 'assistant', 'assistant answer', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:user question with pre-turn tools',
      'bubble:assistant answer',
      'footer',
    ]);
    expect(stepCalls(wrapper)).toEqual([]);
    expect(wrapper.text()).not.toContain('ran memory_recall');
  });

  it('a working turn whose only row is the user\'s draws no line for its seed call: the live line alone follows the question, carrying only the running call', () => {
    const turnId = 203;
    const b = block(turnId, [
      row('2030', 'user', 'user question, still working', turnId, { tool_calls: calls('memory_recall') }),
    ]);
    startLiveTool(ConfigType.USER, turnId, 2039, 'web_search', 'checking the forecast');

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:user question, still working', 'live']);
    expect(wrapper.findAll('.activity .call__summary').map((c) => c.text())).toEqual(['checking the forecast']);
    expect(wrapper.text()).not.toContain('ran memory_recall');
  });

  it('a step row that only asked for tools draws its block with the step block alone', () => {
    const turnId = 211;
    const b = settledTurn(turnId, [
      row('2110', 'user', 'what is the weather', turnId),
      row('2111', 'assistant', '', turnId, { tool_calls: calls('web_search') }),
      row('2112', 'assistant', 'answer', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // The step row said nothing: its block is its line, with no text under it.
    expect(layout(wrapper)).toEqual(['user:what is the weather', 'step:ran web_search', 'bubble:answer', 'footer']);
    expect(wrapper.find('[data-transcript-row-id="2111"] .step').exists()).toBe(true);
    expect(wrapper.find('[data-transcript-row-id="2111"] .speech-form__text').exists()).toBe(false);
    expect(stepCalls(wrapper)).toEqual([['ran web_search']]);
  });

  it('a multi-call exchange draws each row as a block of its own — its step block, then its prose — and one footer that rolls nothing up', () => {
    const wrapper = mount(TurnView, { props: { block: threeCallTurn(), type: ConfigType.USER, fullThread: false } });

    // "Thought for" leads only the row that thought; the interim reply and the
    // answer each read under their own row's line, the same way.
    expect(layout(wrapper)).toEqual([
      'user:research the museum',
      'step:ran web_search | interim',
      'step:Thought for 2s · ran read',
      'step:ran memory_recall | final',
      'footer',
    ]);
    expect(stepCalls(wrapper)).toEqual([['ran web_search'], ['ran read'], ['ran memory_recall']]);
    expect(wrapper.findAll('.speech-form--chalie')).toHaveLength(3);
    const steps = wrapper.findAll('.step');
    // The reasoning folds behind the row that reasoned, and no row's prose
    // hides inside a fold: every text is on the page, outside any step block.
    expect(steps.map((s) => s.text().includes('which file has it?'))).toEqual([false, true, false]);
    expect(steps.some((s) => /interim|final/.test(s.text()))).toBe(false);
    const texts = wrapper.findAll('.speech-form__text').map((b) => b.element);
    expect(texts).toHaveLength(2);
    expect(texts.some((b) => b.closest('[inert]') || b.closest('.step'))).toBe(false);
    // One footer for the exchange, with nothing to unfold and no work tally.
    expect(wrapper.findAll('.speech-form__meta')).toHaveLength(1);
    const footer = footerOf(wrapper);
    expect(footer.find('button[aria-expanded]').exists()).toBe(false);
    expect(footer.text()).not.toMatch(/step|thought/i);
  });

  it('only the row that closes the exchange carries the timestamp and actions', () => {
    const wrapper = mount(TurnView, { props: { block: threeCallTurn(), type: ConfigType.USER, fullThread: false } });

    const rows = wrapper.findAll('.msg-row');
    const last = rows[rows.length - 1]!;
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
    expect(last.find('.speech-form__timestamp').exists()).toBe(true);
    expect(wrapper.findAll(COPY)).toHaveLength(1);
    expect(last.findAll(COPY)).toHaveLength(1);
    expect(wrapper.findAll(SPEAK)).toHaveLength(1);
    expect(last.findAll(SPEAK)).toHaveLength(1);
    expect(wrapper.findAll(REPLY)).toHaveLength(1);
    expect(last.findAll(REPLY)).toHaveLength(1);
  });

  it('the settled answer keeps its timestamp and actions while the turn is still working', () => {
    const turnId = 214;
    // The backend can mark the answer settled before the turn itself is done.
    const b = block(turnId, [
      row('2140', 'user', 'summarise it', turnId),
      row('2141', 'assistant', '', turnId, { tool_calls: calls('read') }),
      row('2142', 'assistant', 'answer', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:summarise it', 'step:ran read', 'bubble:answer', 'footer', 'live']);
    const footer = footerOf(wrapper);
    expect(footer.find('.speech-form__timestamp').exists()).toBe(true);
    expect(footer.findAll(SPEAK)).toHaveLength(1);
  });

  it('a running exchange has no timestamp or actions yet — its step blocks and interim reply so far, then the live line, which never repeats the reply', () => {
    const turnId = 218;
    const b = block(turnId, [
      row('2180', 'user', 'find the file', turnId),
      row('2181', 'assistant', 'let me check', turnId, { tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:find the file', 'step:ran read | let me check', 'live']);
    expect(stepCalls(wrapper)).toEqual([['ran read']]);
    expect(wrapper.find('.activity').text()).not.toContain('let me check');
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(0);
    expect(wrapper.findAll(COPY)).toHaveLength(0);
  });

  it('a message that joined the running turn stays inside its exchange: the step before it gets no timestamp or actions', () => {
    const turnId = 219;
    const b = settledTurn(turnId, [
      row('2190', 'user', 'compare two cities', turnId),
      row('2191', 'assistant', 'searching both', turnId, { tool_calls: calls('web_search') }),
      row('2192', 'user', 'add a third', turnId, { joined: true }),
      row('2193', 'assistant', 'all three compared', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // 'searching both' reads under its call's step block; the joined message
    // keeps the exchange open, so no footer stands before it.
    expect(layout(wrapper)).toEqual([
      'user:compare two cities',
      'step:ran web_search | searching both',
      'user:add a third',
      'bubble:all three compared',
      'footer',
    ]);
    expect(stepCalls(wrapper)).toEqual([['ran web_search']]);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
    expect(wrapper.findAll(COPY)).toHaveLength(1);
  });

  it('a cancelled exchange keeps its timestamp on the last thing Chalie said, and a step that came after it reads below that footer, in order', () => {
    const turnId = 206;
    // Stopped mid-turn: nothing was ever settled, and the last call died with
    // no text — so the footer closes 'partial', and the later step follows it.
    const b = settledTurn(turnId, [
      row('2060', 'user', 'dig into it', turnId),
      row('2061', 'assistant', 'partial', turnId, { tool_calls: calls('web_search') }),
      row('2062', 'assistant', '', turnId, { tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:dig into it',
      'step:ran web_search | partial',
      'footer',
      'step:ran read',
    ]);
    expect(stepCalls(wrapper)).toEqual([['ran web_search'], ['ran read']]);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
    // Nothing settled, so nothing was spoken: no read-aloud button anywhere.
    expect(wrapper.findAll(SPEAK)).toHaveLength(0);
  });

  it('each exchange of a thread keeps its own footer — one abandoned without a settled reply is closed when the next opens', () => {
    const turnId = 205;
    const b = settledTurn(turnId, [
      row('2050', 'user', 'first question', turnId),
      row('2051', 'assistant', 'first answer', turnId, { tool_calls: calls('web_search') }),
      row('2052', 'user', 'second question', turnId, { thread_message: true }),
      row('2053', 'assistant', 'second answer', turnId, { thread_message: true, settled: true, tool_calls: calls('memory_recall') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: true } });

    expect(layout(wrapper)).toEqual([
      'user:first question',
      'step:ran web_search | first answer',
      'footer',
      'user:second question',
      'step:ran memory_recall | second answer',
      'footer',
    ]);
  });

  it('a forked spine turn shows the opener exchange only: one footer, no line for the opener\'s seed call, none of the dropped continuation\'s calls', () => {
    const turnId = 207;
    const b = settledTurn(turnId, [
      row('2070', 'user', 'opener question', turnId, { tool_calls: calls('memory_recall') }),
      row('2071', 'assistant', 'opener reply', turnId),
      row('2072', 'user', 'thread continuation question', turnId, { thread_message: true }),
      row('2073', 'assistant', 'thread continuation reply', turnId, { thread_message: true, tool_calls: calls('web_search') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(wrapper.attributes('data-forked')).toBe('true');
    expect(wrapper.text()).not.toContain('thread continuation');
    expect(layout(wrapper)).toEqual([
      'user:opener question',
      'bubble:opener reply',
      'footer',
    ]);
    expect(stepCalls(wrapper)).toEqual([]);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
  });

  it('the thread pill rides the closing footer, never a step block — not even the step after the settle, which reads below the footer', () => {
    const turnId = 209;
    const b = settledTurn(turnId, [
      row('2090', 'user', 'opener question', turnId),
      row('2091', 'assistant', 'opener', turnId, { settled: true }),
      row('2092', 'assistant', '', turnId, { tool_calls: calls('read') }),
      row('2093', 'user', 'a reply inside the thread', turnId, { thread_message: true }),
    ]);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false, threadPill: { status: 'working', label: 'museum hours' } },
    });

    expect(layout(wrapper)).toEqual(['user:opener question', 'bubble:opener', 'footer', 'step:ran read']);
    expect(wrapper.findAll('.thread-pill')).toHaveLength(1);
    const withTimestamp = wrapper.findAll('.msg-row').filter((r) => r.find('.speech-form__timestamp').exists());
    expect(withTimestamp).toHaveLength(1);
    expect(withTimestamp[0]!.find('.thread-pill').exists()).toBe(true);
    expect(wrapper.find('.step .thread-pill').exists()).toBe(false);
  });

  /** A fork whose opener's only row is a silent step that never settled, then a
   *  thread continuation that did answer. */
  function forkedUnsettledOpener(turnId: number): ConversationTurnBlock {
    return settledTurn(turnId, [
      row(`${turnId}0`, 'user', 'opener question', turnId),
      row(`${turnId}1`, 'assistant', '', turnId, { settled: false, tool_calls: calls('read') }),
      row(`${turnId}2`, 'user', 'a question inside the thread', turnId, { thread_message: true }),
      row(`${turnId}3`, 'assistant', 'thread answer', turnId, { settled: true, thread_message: true }),
    ]);
  }

  it('a forked spine turn whose opener never settled keeps its timestamp and thread pill on the opener, not on the thread answer it hides', () => {
    const wrapper = mount(TurnView, {
      props: {
        block: forkedUnsettledOpener(220),
        type: ConfigType.USER,
        fullThread: false,
        threadPill: { status: 'done', label: 'museum hours' },
      },
    });

    expect(wrapper.text()).not.toContain('thread answer');
    expect(layout(wrapper)).toEqual(['user:opener question', 'step:ran read', 'footer']);
    const footer = footerOf(wrapper);
    expect(footer.find('.thread-pill').exists()).toBe(true);
    expect(wrapper.findAll('.thread-pill')).toHaveLength(1);
  });

  it('the thread panel closes both exchanges of a fork whose opener never settled: the silent opener step and the thread answer each keep a timestamp', () => {
    const wrapper = mount(TurnView, {
      props: { block: forkedUnsettledOpener(221), type: ConfigType.USER, fullThread: true },
    });

    expect(layout(wrapper)).toEqual([
      'user:opener question',
      'step:ran read',
      'footer',
      'user:a question inside the thread',
      'bubble:thread answer',
      'footer',
    ]);
  });

  it('a cancelled exchange where every row only called tools still closes on its last row, with a timestamp and actions, and says the turn ended', () => {
    const turnId = 222;
    const b = settledTurn(turnId, [
      row('2220', 'user', 'dig into it', turnId),
      row('2221', 'assistant', '', turnId, { settled: false, tool_calls: calls('web_search') }),
      row('2222', 'assistant', '', turnId, { settled: false, tool_calls: calls('read') }),
    ]);
    b.crashed = true;

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // Each row's step block stands above the closing footer; the crash note is
    // a row after it.
    expect(layout(wrapper)).toEqual(['user:dig into it', 'step:ran web_search', 'step:ran read', 'footer', 'other']);
    const closing = footerOf(wrapper);
    expect(closing.find(COPY).exists()).toBe(true);
    expect(closing.find(REPLY).exists()).toBe(true);
    expect(wrapper.find('.turn-crashed').exists()).toBe(true);
  });

  it('a still-working exchange of tool-only rows stays open: a step block each, no timestamp or actions on any row yet', () => {
    const turnId = 223;
    const b = block(turnId, [
      row('2230', 'user', 'dig into it', turnId),
      row('2231', 'assistant', '', turnId, { settled: false, tool_calls: calls('web_search') }),
      row('2232', 'assistant', '', turnId, { settled: false, tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:dig into it', 'step:ran web_search', 'step:ran read', 'live']);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(0);
    expect(wrapper.findAll(COPY)).toHaveLength(0);
    expect(wrapper.findAll(REPLY)).toHaveLength(0);
  });

  it('the footer reads the exchange\'s real duration — a joined row does not restart the clock', () => {
    const turnId = 230;
    // The exchange opens at 00:00:00Z; the tool-only step rides the running
    // exchange, the joined row lands at 00:01:00Z (it must NOT reset the
    // clock) and the settled answer at 00:02:48Z.
    const b = settledTurn(turnId, [
      row('2300', 'user', 'plan the visit', turnId),
      row('2301', 'assistant', '', turnId, { tool_calls: calls('web_search', 'read') }),
      row('2302', 'user', 'add a third day', turnId, { joined: true, created_at: '2026-01-01T00:01:00Z' }),
      row('2303', 'assistant', 'the plan is done', turnId, { settled: true, created_at: '2026-01-01T00:02:48Z' }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // One footer for the whole exchange — the joined row opened none — reading
    // 00:00:00Z → 00:02:48Z beside the timestamp, with no step count.
    expect(layout(wrapper)).toEqual([
      'user:plan the visit',
      'step:ran web_search, ran read',
      'user:add a third day',
      'bubble:the plan is done',
      'footer',
    ]);
    const meta = footerOf(wrapper).find('.speech-form__meta');
    expect(meta.text()).toContain('· 2m 48s');
    expect(meta.text()).not.toContain('steps');
  });

  /** What each footer reads beside its timestamp, top to bottom: the elapsed
   *  time, or null when it shows none. */
  function footerDurations(wrapper: VueWrapper): (string | null)[] {
    return wrapper.findAll('.speech-form__meta').map((m) => /· (\d+[hms](?: \d+[ms])?)/.exec(m.text())?.[1] ?? null);
  }

  it('a reply that did no work shows no duration — even right after an earlier reply in the same exchange that worked and was timed', () => {
    const turnId = 231;
    const b = settledTurn(turnId, [
      row('2310', 'user', 'book the table', turnId),
      row('2311', 'assistant', '', turnId, { tool_calls: calls('booking'), created_at: '2026-01-01T00:00:10Z' }),
      row('2312', 'assistant', 'Booked for eight.', turnId, { settled: true, created_at: '2026-01-01T00:00:30Z' }),
      // 20s pass in the second reply, all of it prose: still no duration.
      row('2313', 'assistant', 'They also keep a waitlist.', turnId, { created_at: '2026-01-01T00:00:35Z' }),
      row('2314', 'assistant', 'Anything else?', turnId, { settled: true, created_at: '2026-01-01T00:00:55Z' }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:book the table', 'step:ran booking', 'bubble:Booked for eight.', 'footer',
      'bubble:They also keep a waitlist.', 'bubble:Anything else?', 'footer',
    ]);
    expect(footerDurations(wrapper)).toEqual(['30s', null]);
  });

  it('a second settled reply in one exchange gets its own footer, timed from the first row after the earlier settle, not from the question', () => {
    const turnId = 232;
    const b = settledTurn(turnId, [
      row('2320', 'user', 'check both venues', turnId),
      row('2321', 'assistant', '', turnId, { tool_calls: calls('web_search'), created_at: '2026-01-01T00:00:10Z' }),
      row('2322', 'assistant', 'The first is open.', turnId, { settled: true, created_at: '2026-01-01T00:00:30Z' }),
      row('2323', 'assistant', '', turnId, { tool_calls: calls('read'), created_at: '2026-01-01T00:01:00Z' }),
      row('2324', 'assistant', 'The second is closed.', turnId, { settled: true, created_at: '2026-01-01T00:01:45Z' }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:check both venues',
      'step:ran web_search', 'bubble:The first is open.', 'footer',
      'step:ran read', 'bubble:The second is closed.', 'footer',
    ]);
    // 00:00:00 → 00:00:30, then 00:01:00 → 00:01:45.
    expect(footerDurations(wrapper)).toEqual(['30s', '45s']);
  });

  it('the running block is one block for the whole run — a refetch, and a new stored row landing, keep the same block', async () => {
    const turnId = 233;
    const asked = row('2330', 'user', 'find the opening hours', turnId);
    const searched = (): ConversationMessage => row('2331', 'assistant', '', turnId, { tool_calls: [
      { id: 2381, tool_name: 'web_search', summary: 'ran web_search', state: 'done', ended_at: null, delegate: null },
    ] });
    startLiveTool(ConfigType.USER, turnId, 2389, 'read', 'reading the venue page');
    const wrapper = mount(TurnView, { props: { block: block(turnId, [asked, searched()]), type: ConfigType.USER, fullThread: false } });
    const before = wrapper.get('.activity').element;

    // A refetch of the same rows: the same block.
    await wrapper.setProps({ block: block(turnId, [asked, searched()]) });
    expect(wrapper.get('.activity').element).toBe(before);

    // A new stored row lands: the same block, still carrying the running call.
    await wrapper.setProps({ block: block(turnId, [asked, searched(), row('2332', 'assistant', 'Still looking.', turnId)]) });
    expect(layout(wrapper)).toEqual([
      'user:find the opening hours', 'step:ran web_search', 'bubble:Still looking.', 'live',
    ]);
    expect(wrapper.get('.activity').element).toBe(before);
    expect(wrapper.findAll('.activity .call__summary').map((c) => c.text())).toEqual(['reading the venue page']);
  });
});

describe('rows a cancel would drop — the stop\'s hand-back marker', () => {
  const DROPPED = '[data-dropped-on-cancel]';

  function marked(wrapper: VueWrapper): string[] {
    return wrapper.findAll(DROPPED).map((r) => r.attributes('data-user-text') ?? '');
  }

  it('marks a working turn\'s opener that nothing has answered yet', () => {
    const turnId = 501;
    const wrapper = mount(TurnView, {
      props: { block: block(turnId, [msg('5010', 'user', 'plan my trip', turnId)]), type: ConfigType.USER },
    });

    expect(marked(wrapper)).toEqual(['plan my trip']);
  });

  it('marks nothing once a message has joined the running turn: a cancel keeps it and everything before it', () => {
    const turnId = 502;
    const wrapper = mount(TurnView, {
      props: {
        block: block(turnId, [
          msg('5020', 'user', 'plan my trip', turnId),
          row('5021', 'user', 'make it three days', turnId, { joined: true }),
        ]),
        type: ConfigType.USER,
      },
    });

    expect(marked(wrapper)).toEqual([]);
  });

  it('marks only the thread reply the turn has not answered, never the settled opener before it', () => {
    const turnId = 503;
    const wrapper = mount(TurnView, {
      props: {
        block: block(turnId, [
          msg('5030', 'user', 'plan my trip', turnId),
          row('5031', 'assistant', 'Rome it is', turnId, { settled: true }),
          msg('5032', 'user', 'what about Florence?', turnId, true),
        ]),
        type: ConfigType.USER,
        fullThread: true,
      },
    });

    expect(marked(wrapper)).toEqual(['what about Florence?']);
  });

  it('marks nothing on a turn that is no longer working, even one that ended on an unanswered message', () => {
    const turnId = 504;
    const b = settledTurn(turnId, [msg('5040', 'user', 'a question that crashed the turn', turnId)]);
    b.crashed = true;
    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER } });

    expect(marked(wrapper)).toEqual([]);
  });
});

describe('crashed-turn note', () => {
  it('renders an "ended unexpectedly" note for a crashed turn that produced no reply', () => {
    const turnId = 301;
    const b = block(turnId, [msg('3010', 'user', 'a question that crashed the turn', turnId)]);
    b.working = false;
    b.crashed = true;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.find('.turn-crashed').exists()).toBe(true);
    expect(wrapper.find('.turn-crashed').text()).toContain('ended unexpectedly');
  });

  it('does NOT render the note when the crashed turn still landed a reply before dying', () => {
    const turnId = 302;
    const b = block(turnId, [
      msg('3020', 'user', 'a question', turnId),
      msg('3021', 'assistant', 'a real answer landed before the crash', turnId),
    ]);
    b.working = false;
    b.crashed = true;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    // A crash that left content keeps the content; the note is only for the
    // "nothing came back" case.
    expect(wrapper.find('.turn-crashed').exists()).toBe(false);
  });

  it('does NOT render the note for a normal (non-crashed) settled turn', () => {
    const turnId = 303;
    const b = block(turnId, [msg('3030', 'user', 'a question', turnId)]);
    b.working = false;
    // crashed omitted (undefined) — a normal settled turn must never show it.

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.find('.turn-crashed').exists()).toBe(false);
  });

  it('still says so when the only rows that came back were tool-only steps', () => {
    const turnId = 304;
    const b = settledTurn(turnId, [
      row('3040', 'user', 'a question that crashed the turn', turnId),
      row('3041', 'assistant', '', turnId, { tool_calls: calls('web_search') }),
    ]);
    b.crashed = true;

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // The step stays visible as its block, but it said nothing: no text, and
    // the note names the absence.
    expect(wrapper.find('.turn-crashed').exists()).toBe(true);
    expect(wrapper.findAll('.speech-form__text')).toHaveLength(0);
    expect(stepCalls(wrapper)).toEqual([['ran web_search']]);
  });
});

describe('delegate (subagent) turns', () => {
  function delegateBlock(turnId: number, working: boolean, messages: ConversationMessage[]): DelegateTurnBlock {
    return {
      turn_id: turnId,
      gist: null,
      preview: messages[0]?.content ?? '',
      last_activity_at: null,
      working,
      duration_ms: 0,
      messages,
      type: null,
      channel: 'delegate:web_search',
    };
  }

  const stopButtons = (wrapper: ReturnType<typeof mount>) =>
    [...wrapper.element.querySelectorAll('button')].filter((b) => /stop/i.test(b.getAttribute('aria-label') ?? ''));

  it('is addressed by its channel and never by a type, while a user turn with the same id is addressed by its type and never a channel', () => {
    const turnId = 401;
    const rows = [msg('4010', 'user', 'look up the museum hours', turnId)];

    const delegate = mount(TurnView, { props: { block: delegateBlock(turnId, false, rows), fullThread: true } });
    const user = mount(TurnView, { props: { block: { ...block(turnId, rows), working: false }, type: ConfigType.USER } });

    expect(delegate.attributes('data-channel')).toBe('delegate:web_search');
    expect(delegate.attributes('data-type')).toBeUndefined();
    expect(user.attributes('data-type')).toBe(ConfigType.USER);
    expect(user.attributes('data-channel')).toBeUndefined();
  });

  it('captions the user-role row as a task from Chalie; the same row in a user turn is not captioned, and no assistant row ever is', () => {
    const turnId = 402;
    const rows = [
      msg('4020', 'user', 'look up the museum hours', turnId),
      msg('4021', 'assistant', 'The museum opens at nine.', turnId),
    ];

    const delegate = mount(TurnView, { props: { block: delegateBlock(turnId, false, rows), fullThread: true } });
    const user = mount(TurnView, { props: { block: { ...block(turnId, rows), working: false }, type: ConfigType.USER } });

    const delegateRows = delegate.findAll('.msg-row--user');
    expect(delegateRows).toHaveLength(1);
    expect(delegateRows[0]!.text()).toContain('Task from Chalie');
    expect(delegateRows[0]!.text()).toContain('look up the museum hours');
    expect(delegate.findAll('.msg-row--chalie').some((r) => r.text().includes('Task from Chalie'))).toBe(false);
    expect(user.text()).not.toContain('Task from Chalie');
  });

  it('offers a working delegate transcript one stop control, named "Stop subagent", while a working user turn with the same id keeps "Stop and undo"', () => {
    const turnId = 403;
    const rows = [msg('4030', 'user', 'look up the museum hours', turnId)];

    const delegate = mount(TurnView, { props: { block: delegateBlock(turnId, true, rows), fullThread: true } });
    const user = mount(TurnView, { props: { block: block(turnId, rows), type: ConfigType.USER } });

    expect(stopButtons(delegate).map((b) => b.getAttribute('aria-label'))).toEqual(['Stop subagent']);
    expect(stopButtons(user).map((b) => b.getAttribute('aria-label'))).toEqual(['Stop and undo']);
  });

  it('notes a stopped delegate transcript as stopped, but never one that ran to its answer nor a cancelled user turn', () => {
    const turnId = 405;
    const rows = [msg('4050', 'user', 'look up the museum hours', turnId)];
    const stopped = { ...delegateBlock(turnId, false, rows), cancelled: true };

    const delegate = mount(TurnView, { props: { block: stopped, fullThread: true } });
    const finished = mount(TurnView, { props: { block: delegateBlock(turnId, false, rows), fullThread: true } });
    const user = mount(TurnView, { props: { block: { ...block(turnId, rows), working: false, cancelled: true }, type: ConfigType.USER } });

    expect(delegate.text()).toContain('This subagent was stopped.');
    expect(finished.text()).not.toContain('stopped');
    expect(user.text()).not.toContain('stopped');
  });

  it('offers no stop control and no live working row once a delegate transcript has settled', () => {
    const turnId = 404;
    const rows = [
      msg('4040', 'user', 'look up the museum hours', turnId),
      msg('4041', 'assistant', 'The museum opens at nine.', turnId),
    ];

    const delegate = mount(TurnView, { props: { block: delegateBlock(turnId, false, rows), fullThread: true } });

    expect(delegate.text()).toContain('The museum opens at nine.');
    expect(delegate.find('.activity').exists()).toBe(false);
    expect(stopButtons(delegate)).toHaveLength(0);
  });
});

/** A plan of one task per status given, numbered and titled in order. */
function planOf(...statuses: PlanTask['status'][]): PlanTask[] {
  return statuses.map((status, i) => ({ number: i + 1, title: `task ${i + 1}`, task: `do task ${i + 1}`, status }));
}

/** Each plan task's status, read off its row's modifier class. */
function taskStatuses(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.plan-task').map((t) => t.classes().find((c) => c.startsWith('plan-task--'))!.slice(11));
}

// A request's plan stands sticky under the message it plans while the
// exchange works, ticking as the agent marks its tasks; once the exchange
// settles it leaves the column and sits behind the answer footer's plan link.
describe('the plan — live under its message, settled behind the footer', () => {
  it('stands the working exchange\'s plan right under its message and ahead of the live line, opening Chalie\'s side, the live line continuing it', () => {
    const turnId = 501;
    const wrapper = mount(TurnView, {
      props: {
        block: block(turnId, [
          row('5010', 'user', 'what is the weather', turnId, { plan: planOf('done', 'in_progress', 'todo') }),
          row('5011', 'assistant', '', turnId, { tool_calls: calls('plan', 'web_search') }),
        ]),
        type: ConfigType.USER,
      },
    });

    expect(layout(wrapper)).toEqual(['user:what is the weather', 'plan', 'step:ran plan, ran web_search', 'live']);
    const rows = wrapper.findAll('.msg-row');
    expect(rows[1]!.classes()).toEqual(expect.arrayContaining(['msg-row--chalie', 'msg-row--lead']));
    expect(rows[2]!.classes()).toContain('msg-row--cont');
    expect(rows[3]!.classes()).toContain('msg-row--cont');
    expect(rows[1]!.findAll('.plan-task__title').map((t) => t.text())).toEqual(['task 1', 'task 2', 'task 3']);
    expect(taskStatuses(wrapper)).toEqual(['done', 'in_progress', 'todo']);
    // Nothing has settled, so there is no footer and no plan link yet.
    expect(wrapper.find('.speech-form__meta').exists()).toBe(false);
  });

  it('ticks the same card in place when a refetch brings the plan\'s new statuses and an added task', async () => {
    const turnId = 502;
    const rows = (plan: PlanTask[]) => [row('5020', 'user', 'what is the weather', turnId, { plan })];
    const wrapper = mount(TurnView, { props: { block: block(turnId, rows(planOf('in_progress', 'todo'))), type: ConfigType.USER } });
    const card = wrapper.find('.plan-card').element;

    await wrapper.setProps({ block: block(turnId, rows(planOf('done', 'in_progress', 'todo'))) });

    expect(wrapper.find('.plan-card').element).toBe(card);
    expect(taskStatuses(wrapper)).toEqual(['done', 'in_progress', 'todo']);
  });

  it('keeps the one card under the message that opened the request when a message joins it mid-work', () => {
    const turnId = 503;
    const wrapper = mount(TurnView, {
      props: {
        block: block(turnId, [
          row('5030', 'user', 'what is the weather', turnId, { plan: planOf('in_progress', 'todo') }),
          row('5031', 'assistant', '', turnId, { tool_calls: calls('web_search') }),
          row('5032', 'user', 'and will it rain', turnId, { joined: true }),
        ]),
        type: ConfigType.USER,
      },
    });

    expect(layout(wrapper)).toEqual([
      'user:what is the weather', 'plan', 'step:ran web_search', 'user:and will it rain', 'live',
    ]);
  });

  it('draws no card while the exchange has no plan yet', () => {
    const turnId = 504;
    const wrapper = mount(TurnView, {
      props: { block: block(turnId, [row('5040', 'user', 'hello', turnId)]), type: ConfigType.USER },
    });

    expect(layout(wrapper)).toEqual(['user:hello', 'live']);
  });

  it('moves a settled exchange\'s plan behind its footer\'s "plan N/M" link, which opens the checklist below the footer and closes it again', async () => {
    const turnId = 505;
    const wrapper = mount(TurnView, {
      props: {
        block: settledTurn(turnId, [
          row('5050', 'user', 'what is the weather', turnId, { plan: planOf('done', 'cancelled', 'done') }),
          row('5051', 'assistant', 'Sunny, 24°.', turnId, { tool_calls: calls('plan', 'web_search'), settled: true }),
        ]),
        type: ConfigType.USER,
      },
    });

    expect(layout(wrapper)).toEqual([
      'user:what is the weather', 'step:ran plan, ran web_search | Sunny, 24°.', 'footer',
    ]);
    const footer = footerOf(wrapper);
    const link = footer.find('.speech-form__meta button.fold');
    expect(link.text()).toBe('· plan 2/3 ›');
    expect(link.attributes('aria-expanded')).toBe('false');
    expect(footer.find('.trace-body .plan-card').exists()).toBe(false);

    await link.trigger('click');

    expect(link.attributes('aria-expanded')).toBe('true');
    expect(footer.find('.trace-body').classes()).toContain('trace-body--open');
    expect(footer.find('.trace-body .plan-card').exists()).toBe(true);
    expect(taskStatuses(wrapper)).toEqual(['done', 'cancelled', 'done']);

    await link.trigger('click');

    expect(link.attributes('aria-expanded')).toBe('false');
    expect(footer.find('.trace-body').classes()).not.toContain('trace-body--open');
  });

  it('gives a settled exchange with no plan no plan link', () => {
    const turnId = 506;
    const wrapper = mount(TurnView, {
      props: {
        block: settledTurn(turnId, [
          row('5060', 'user', 'what is the weather', turnId),
          row('5061', 'assistant', 'Sunny, 24°.', turnId, { tool_calls: calls('web_search'), settled: true }),
        ]),
        type: ConfigType.USER,
      },
    });

    expect(layout(wrapper)).toEqual(['user:what is the weather', 'step:ran web_search | Sunny, 24°.', 'footer']);
    expect(footerOf(wrapper).find('.speech-form__meta button.fold').exists()).toBe(false);
  });

  it('settles an earlier exchange\'s plan into its footer while the working one\'s plan stands live — and the thread panel pins it to the panel\'s top, never the spine for a forked turn', () => {
    const turnId = 507;
    const messages = [
      row('5070', 'user', 'what is the weather', turnId, { plan: planOf('done', 'done') }),
      row('5071', 'assistant', 'Sunny, 24°.', turnId, { settled: true }),
      row('5072', 'user', 'and tomorrow', turnId, { thread_message: true, plan: planOf('in_progress') }),
    ];

    const spine = mount(TurnView, { props: { block: block(turnId, messages), type: ConfigType.USER, fullThread: false } });
    const panel = mount(TurnView, { props: { block: block(turnId, messages), type: ConfigType.USER, fullThread: true } });

    // The spine shows the forked turn's opener only — its plan behind the
    // footer link, and no live card: the thread pill carries the live state.
    expect(layout(spine)).toEqual(['user:what is the weather', 'bubble:Sunny, 24°.', 'footer']);
    expect(spine.find('.speech-form__meta button.fold').text()).toBe('· plan 2/2 ›');
    expect(spine.classes()).not.toContain('turn-view--panel');

    expect(layout(panel)).toEqual([
      'user:what is the weather', 'bubble:Sunny, 24°.', 'footer', 'user:and tomorrow', 'plan', 'live',
    ]);
    expect(panel.find('.msg-row--plan .plan-task').classes()).toContain('plan-task--in_progress');
    expect(panel.classes()).toContain('turn-view--panel');
  });
});
