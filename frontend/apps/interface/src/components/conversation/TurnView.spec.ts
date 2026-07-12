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
 * Real component tree throughout (TurnView → ActCycle/ActCycleGroup/
 * UserBubble/ChalieBubble), real Pinia, real `@chalie/shared` barrel — the
 * only thing given a DOM stand-in is happy-dom itself (this file opts in via
 * the docblock above; the suite's default `environment: 'node'` is
 * untouched elsewhere).
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import ActCycle from './ActCycle.vue';
import ActCycleGroup from './ActCycleGroup.vue';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';

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

describe('tool-call trace placement', () => {
  const toolCalls: NonNullable<ConversationMessage['tool_calls']> = [
    { tool_name: 'memory_recall', summary: 'recalled a memory', state: 'done', ended_at: null },
  ];

  it('rides a settled exchange\'s pre-turn tool call on the footer trace, not a separate group row', () => {
    const turnId = 201;
    const b = block(turnId, [
      msg('2010', 'user', 'user question with pre-turn tools', turnId, false, toolCalls),
      msg('2011', 'assistant', 'assistant answer', turnId),
    ]);
    b.working = false;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    const rows = wrapper.findAll('.msg-row');
    const kinds = rows.map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.findComponent(ActCycleGroup).exists()
          ? 'group'
          : r.text().includes('assistant answer')
            ? 'assistant'
            : r.text().includes('user question')
              ? 'user'
              : 'other',
    );

    // Settled exchange: the pre-turn tool call is aggregated onto that
    // exchange's footer as its inline "N tools used" trace — NO separate
    // collapsed-group row is emitted once there's a footer to carry it.
    expect(kinds).toEqual(['user', 'assistant', 'footer']);
    // The call is not dropped — it rides the footer's expandable trace.
    expect(wrapper.find('.trace-pill').text()).toContain('1 tool used');
    expect(wrapper.find('.call__fn').text()).toBe('memory_recall');
  });

  it('rides a settled exchange\'s assistant-step tool call on the footer trace', () => {
    const turnId = 202;
    const b = block(turnId, [
      msg('2020', 'user', 'plain user question', turnId),
      msg('2021', 'assistant', 'assistant answer with tools', turnId, false, toolCalls),
    ]);
    b.working = false;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    const rows = wrapper.findAll('.msg-row');
    const kinds = rows.map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.findComponent(ActCycleGroup).exists()
          ? 'group'
          : r.text().includes('assistant answer')
            ? 'assistant'
            : r.text().includes('plain user question')
              ? 'user'
              : 'other',
    );

    expect(kinds).toEqual(['user', 'assistant', 'footer']);
    expect(wrapper.find('.trace-pill').text()).toContain('1 tool used');
  });

  it('flushes a pending trail at the end when the block has no assistant row yet', () => {
    const turnId = 203;
    const b = block(turnId, [
      msg('2030', 'user', 'user question, still working', turnId, false, toolCalls),
    ]);
    b.working = true;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    const rows = wrapper.findAll('.msg-row');
    const userIdx = rows.findIndex((r) => r.text().includes('user question, still working'));
    const groupIdx = rows.findIndex((r) => r.findComponent(ActCycleGroup).exists());

    expect(userIdx).toBeGreaterThanOrEqual(0);
    expect(groupIdx).toBeGreaterThan(userIdx);
  });

  const stepTools: NonNullable<ConversationMessage['tool_calls']> = [
    { tool_name: 'web_search', summary: 'searched the web', state: 'done', ended_at: null },
  ];

  it('aggregates every tool call of a multi-step exchange onto its ONE footer trace', () => {
    // The turn_exchange contract: when one user question yields several assistant
    // steps that each ran tools, every call gathers onto that exchange's single
    // footer trace — never interleaved between an interim step and the reply,
    // never split across rows.
    const turnId = 204;
    const b = block(turnId, [
      msg('2040', 'user', 'the multi-step question', turnId),
      msg('2041', 'assistant', 'interim step', turnId, false, stepTools),
      msg('2042', 'assistant', 'final answer', turnId, false, toolCalls),
    ]);
    b.working = false;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    const rows = wrapper.findAll('.msg-row');
    const kinds = rows.map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.findComponent(ActCycleGroup).exists()
          ? 'group'
          : r.text().includes('final answer')
            ? 'assistant-final'
            : r.text().includes('interim step')
              ? 'assistant-interim'
              : r.text().includes('multi-step question')
                ? 'user'
                : 'other',
    );

    // Both steps' calls aggregate onto the single footer for this exchange,
    // which sits below the final reply — no separate chip rows.
    expect(kinds).toEqual([
      'user',
      'assistant-interim',
      'assistant-final',
      'footer',
    ]);
    // The footer trace counts BOTH calls (interim + final).
    expect(wrapper.find('.trace-pill').text()).toContain('2 tools used');
    expect(wrapper.findAll('.call__fn').map((c) => c.text())).toEqual(['web_search', 'memory_recall']);
  });

  it('gives each exchange its OWN footer trace across a multi-exchange thread', () => {
    // Two turn_exchanges in one block (thread panel): exchange one's calls ride
    // its own footer right after its reply and BEFORE exchange two opens;
    // exchange two's calls ride its own footer at the bottom.
    const turnId = 205;
    const b = block(turnId, [
      msg('2050', 'user', 'first question', turnId),
      msg('2051', 'assistant', 'first answer', turnId, false, stepTools),
      msg('2052', 'user', 'second question', turnId, true),
      msg('2053', 'assistant', 'second answer', turnId, true, toolCalls),
    ]);
    b.working = false;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: true },
    });

    const rows = wrapper.findAll('.msg-row');
    const kinds = rows.map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.findComponent(ActCycleGroup).exists()
          ? 'group'
          : r.text().includes('first answer')
            ? 'a1'
            : r.text().includes('second answer')
              ? 'a2'
              : r.text().includes('first question')
                ? 'u1'
                : r.text().includes('second question')
                  ? 'u2'
                  : 'other',
    );

    // Each exchange gets its OWN footer, right after its own reply — the
    // second exchange's footer never merges into or replaces the first's.
    expect(kinds).toEqual(['u1', 'a1', 'footer', 'u2', 'a2', 'footer']);
    expect(wrapper.findAll('.speech-form__meta')).toHaveLength(2);
    // Each footer carries exactly its own exchange's single call.
    expect(wrapper.findAll('.trace-pill').map((p) => p.text().includes('1 tool used'))).toEqual([true, true]);
  });

  it('renders exactly ONE footer for a forked spine turn — the openers, carrying its trace', () => {
    // The other half of the cardinality fix: on the spine (fullThread=false)
    // only the opener exchange survives the thread_message drop, so exactly
    // one footer renders — the thread's continuation carries no footer of
    // its own here (it isn't even rendered on the spine).
    const turnId = 206;
    const b = block(turnId, [
      msg('2060', 'user', 'opener question', turnId, false, toolCalls),
      msg('2061', 'assistant', 'opener reply', turnId),
      msg('2062', 'user', 'thread continuation question', turnId, true),
      msg('2063', 'assistant', 'thread continuation reply', turnId, true, stepTools),
    ]);
    b.working = false;

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.attributes('data-forked')).toBe('true');
    expect(wrapper.text()).not.toContain('thread continuation');

    const rows = wrapper.findAll('.msg-row');
    const kinds = rows.map((r) =>
      r.find('.speech-form__meta').exists()
        ? 'footer'
        : r.findComponent(ActCycleGroup).exists()
          ? 'group'
          : r.text().includes('opener reply')
            ? 'assistant'
            : r.text().includes('opener question')
              ? 'user'
              : 'other',
    );

    expect(kinds).toEqual(['user', 'assistant', 'footer']);
    expect(wrapper.findAll('.speech-form__meta')).toHaveLength(1);
    // The opener's own pre-turn call rides the single footer trace; the
    // spine-dropped continuation's tools never leak onto it.
    expect(wrapper.find('.trace-pill').text()).toContain('1 tool used');
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
});
