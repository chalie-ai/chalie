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
 * Real component tree throughout (TurnView → ActCycle/BubbleFooter/
 * UserBubble/ChalieBubble), real Pinia, real `@chalie/shared` barrel — the
 * only thing given a DOM stand-in is happy-dom itself (this file opts in via
 * the docblock above; the suite's default `environment: 'node'` is
 * untouched elsewhere).
 *
 * The second half pins how a turn's rows are drawn: one provider call is one
 * assistant row, with its own bubble (only when it said something) and its own
 * trace strip (its tool calls and reasoning); only the row that closes an
 * exchange adds the timestamp and actions.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import ActCycle from './ActCycle.vue';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../../api/conversation';

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

/** One finished call per name, as the backend sends a row's chips. */
function calls(...names: string[]): ToolCalls {
  return names.map((tool_name) => ({
    tool_name, summary: `ran ${tool_name}`, state: 'done' as const, ended_at: null, delegate: null,
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

/** The rendered turn read top to bottom, one entry per row: a user bubble, a
 *  Chalie bubble, a strip (a trace with no timestamp), a footer (a timestamp,
 *  with or without a trace) or the live indicator. */
function layout(wrapper: VueWrapper): string[] {
  return wrapper.findAll('.msg-row').map((r) => {
    if (r.find('.user-message').exists()) return `user:${r.find('.user-message').attributes('data-user-text')}`;
    if (r.find('.speech-form--chalie').exists()) return `bubble:${r.find('.speech-form__text').text()}`;
    if (r.find('.speech-form__meta').exists()) return r.find('.speech-form__timestamp').exists() ? 'footer' : 'strip';
    if (r.findComponent(ActCycle).exists()) return 'live';
    return 'other';
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
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

    expect(wrapper.findComponent(ActCycle).exists()).toBe(true);
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

    expect(wrapper.findComponent(ActCycle).exists()).toBe(false);
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

    expect(wrapper.findComponent(ActCycle).exists()).toBe(true);
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


// One provider call is one assistant row, drawn on its own: its bubble when it
// said something, and under it that row's OWN tool calls and reasoning. Only the
// row that closes an exchange adds the timestamp and actions.
describe('tool-call trace placement — each row draws its own trace', () => {
  it('a call stored on the user\'s own row draws a strip right under the question, apart from the answer\'s footer', () => {
    const turnId = 201;
    const b = settledTurn(turnId, [
      row('2010', 'user', 'user question with pre-turn tools', turnId, { tool_calls: calls('memory_recall') }),
      row('2011', 'assistant', 'assistant answer', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:user question with pre-turn tools',
      'strip',
      'bubble:assistant answer',
      'footer',
    ]);
    const rows = wrapper.findAll('.msg-row');
    // The strip carries the call and nothing else...
    expect(rows[1]!.find('.trace-pill').text()).toContain('1 tool used');
    expect(rows[1]!.find('.call__fn').text()).toBe('memory_recall');
    expect(rows[1]!.find('.speech-form__timestamp').exists()).toBe(false);
    // ...and the answer's footer carries the timestamp but not that call.
    expect(rows[3]!.find('.speech-form__timestamp').exists()).toBe(true);
    expect(rows[3]!.find('.trace-pill').exists()).toBe(false);
  });

  it('a working turn whose only row is the user\'s shows that row\'s calls right under the question, ahead of the live indicator', async () => {
    const turnId = 203;
    const b = block(turnId, [
      row('2030', 'user', 'user question, still working', turnId, { tool_calls: calls('memory_recall') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:user question, still working', 'strip', 'live']);
    const pill = wrapper.find('.trace-pill');
    expect(pill.attributes('aria-expanded')).toBe('false');
    await pill.trigger('click');
    expect(pill.attributes('aria-expanded')).toBe('true');
    expect(wrapper.find('.call__fn').text()).toBe('memory_recall');
  });

  it('a step row that only asked for tools draws its trace and no bubble', () => {
    const turnId = 211;
    const b = settledTurn(turnId, [
      row('2110', 'user', 'what is the weather', turnId),
      row('2111', 'assistant', '', turnId, { tool_calls: calls('web_search') }),
      row('2112', 'assistant', 'answer', turnId, { settled: true }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    // One bubble for the whole exchange: the answer. The step row left nothing
    // behind but its trace, sitting before the answer it led to.
    const bubbles = wrapper.findAll('.speech-form--chalie');
    expect(bubbles).toHaveLength(1);
    expect(bubbles[0]!.text()).toBe('answer');
    expect(wrapper.find('[data-transcript-row-id="2111"]').exists()).toBe(false);
    expect(layout(wrapper)).toEqual(['user:what is the weather', 'strip', 'bubble:answer', 'footer']);
    expect(wrapper.findAll('.msg-row')[1]!.find('.trace-pill').text()).toContain('1 tool used');
  });

  it('every row of a multi-call exchange owns its trace: each pill lists only that row\'s calls and reasoning', () => {
    const wrapper = mount(TurnView, { props: { block: threeCallTurn(), type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual([
      'user:research the museum',
      'bubble:interim',
      'strip',
      'strip',
      'bubble:final',
      'footer',
    ]);
    const traces = wrapper.findAll('.speech-form__meta-wrap');
    expect(traces.map((t) => t.find('.trace-pill').text().includes('1 tool used'))).toEqual([true, true, true]);
    expect(traces.map((t) => t.findAll('.call__fn').map((c) => c.text()))).toEqual([
      ['web_search'],
      ['read'],
      ['memory_recall'],
    ]);
    // The reasoning belongs to the one row that did it.
    expect(traces.map((t) => t.text().includes('thought for 2s'))).toEqual([false, true, false]);
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

    expect(layout(wrapper)).toEqual(['user:summarise it', 'strip', 'bubble:answer', 'footer', 'live']);
    const footer = wrapper.findAll('.msg-row')[3]!;
    expect(footer.find('.speech-form__timestamp').exists()).toBe(true);
    expect(footer.findAll(SPEAK)).toHaveLength(1);
  });

  it('a running exchange has no timestamp or actions yet, only its calls so far', () => {
    const turnId = 218;
    const b = block(turnId, [
      row('2180', 'user', 'find the file', turnId),
      row('2181', 'assistant', 'let me check', turnId, { tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:find the file', 'bubble:let me check', 'strip', 'live']);
    expect(wrapper.find('.trace-pill').text()).toContain('1 tool used');
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(0);
    expect(wrapper.findAll(COPY)).toHaveLength(0);
  });

  it('a cancelled exchange keeps its timestamp on the last thing Chalie said, and a trailing step is only a strip', () => {
    const turnId = 206;
    // Stopped mid-turn: nothing was ever settled, and the last call died with
    // no text.
    const b = settledTurn(turnId, [
      row('2060', 'user', 'dig into it', turnId),
      row('2061', 'assistant', 'partial', turnId, { tool_calls: calls('web_search') }),
      row('2062', 'assistant', '', turnId, { tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:dig into it', 'bubble:partial', 'footer', 'strip']);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
    expect(wrapper.findAll('.msg-row')[2]!.find('.speech-form__timestamp').exists()).toBe(true);
    // Nothing settled, so nothing was spoken: no read-aloud button anywhere.
    expect(wrapper.findAll(SPEAK)).toHaveLength(0);
  });

  it('each exchange of a thread keeps its own footer and trace — one abandoned without a settled reply is closed when the next opens', () => {
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
      'bubble:first answer',
      'footer',
      'user:second question',
      'bubble:second answer',
      'footer',
    ]);
    // Each footer carries exactly its own exchange's single call.
    expect(wrapper.findAll('.speech-form__meta-wrap').map((t) => t.findAll('.call__fn').map((c) => c.text()))).toEqual([
      ['web_search'],
      ['memory_recall'],
    ]);
  });

  it('a forked spine turn shows the opener exchange only: one footer, the opener\'s own trace, none of the dropped continuation\'s calls', () => {
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
    // The opener's pre-turn call is its strip; the reply's footer is the one
    // and only timestamp.
    expect(layout(wrapper)).toEqual(['user:opener question', 'strip', 'bubble:opener reply', 'footer']);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(1);
    expect(wrapper.findAll('.call__fn').map((c) => c.text())).toEqual(['memory_recall']);
  });

  it('the thread pill rides the closing footer, never a step\'s strip', () => {
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

    expect(layout(wrapper)).toEqual(['user:opener question', 'bubble:opener', 'footer', 'strip']);
    expect(wrapper.findAll('.thread-pill')).toHaveLength(1);
    const withTimestamp = wrapper.findAll('.msg-row').filter((r) => r.find('.speech-form__timestamp').exists());
    expect(withTimestamp).toHaveLength(1);
    expect(withTimestamp[0]!.find('.thread-pill').exists()).toBe(true);
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
    expect(layout(wrapper)).toEqual(['user:opener question', 'footer']);
    const footer = wrapper.findAll('.msg-row')[1]!;
    expect(footer.find('.thread-pill').exists()).toBe(true);
    expect(wrapper.findAll('.thread-pill')).toHaveLength(1);
  });

  it('the thread panel closes both exchanges of a fork whose opener never settled: the silent opener step and the thread answer each keep a timestamp', () => {
    const wrapper = mount(TurnView, {
      props: { block: forkedUnsettledOpener(221), type: ConfigType.USER, fullThread: true },
    });

    expect(layout(wrapper)).toEqual([
      'user:opener question',
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

    // The first step is only a strip; the last row carries the footer (the
    // crash note is a row after them).
    expect(layout(wrapper).slice(0, 3)).toEqual(['user:dig into it', 'strip', 'footer']);
    const closing = wrapper.findAll('.msg-row')[2]!;
    expect(closing.find(COPY).exists()).toBe(true);
    expect(closing.find(REPLY).exists()).toBe(true);
    expect(wrapper.find('.turn-crashed').exists()).toBe(true);
  });

  it('a still-working exchange of tool-only rows stays open: no timestamp or actions on any row yet', () => {
    const turnId = 223;
    const b = block(turnId, [
      row('2230', 'user', 'dig into it', turnId),
      row('2231', 'assistant', '', turnId, { settled: false, tool_calls: calls('web_search') }),
      row('2232', 'assistant', '', turnId, { settled: false, tool_calls: calls('read') }),
    ]);

    const wrapper = mount(TurnView, { props: { block: b, type: ConfigType.USER, fullThread: false } });

    expect(layout(wrapper)).toEqual(['user:dig into it', 'strip', 'strip', 'live']);
    expect(wrapper.findAll('.speech-form__timestamp')).toHaveLength(0);
    expect(wrapper.findAll(COPY)).toHaveLength(0);
    expect(wrapper.findAll(REPLY)).toHaveLength(0);
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

    // The step's trace stays visible, but it is not a reply: no bubble, and the
    // note names the absence.
    expect(wrapper.find('.turn-crashed').exists()).toBe(true);
    expect(wrapper.findAll('.speech-form--chalie')).toHaveLength(0);
    expect(wrapper.find('.trace-pill').text()).toContain('1 tool used');
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
    expect(delegate.findComponent(ActCycle).exists()).toBe(false);
    expect(stopButtons(delegate)).toHaveLength(0);
  });
});
