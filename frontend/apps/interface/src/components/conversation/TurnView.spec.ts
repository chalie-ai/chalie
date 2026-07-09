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
 * `thread_message: true` row, read through the real
 * `useConversationFeed().isForkedThread()` — so the block is upserted into
 * the real feed buffer before mount, exactly as the app does on turn fetch.
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
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';
import { useConversationFeed } from '../../composables/useConversationFeed';

function msg(
  id: string,
  role: ConversationMessage['role'],
  content: string,
  turnId: number,
  threadMessage = false,
): ConversationMessage {
  return {
    id,
    role,
    content,
    timestamp: '2026-01-01 00:00:00',
    turn_id: turnId,
    ...(threadMessage ? { thread_message: true } : {}),
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
    useConversationFeed(ConfigType.USER).upsertTurn(b);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: false },
    });

    expect(wrapper.findComponent(ActCycle).exists()).toBe(true);
    // settle0 rows always render regardless of the live-act guard.
    expect(wrapper.text()).toContain('settle0 question');
    expect(wrapper.text()).toContain('settle0 answer');
  });

  it('does NOT render the live-act row for a FORKED working turn (the thread pill owns that indicator)', () => {
    const turnId = 102;
    const b = block(turnId, [
      msg('1020', 'user', 'opener of the thread', turnId),
      msg('1021', 'assistant', 'settle0 reply', turnId),
      msg('1022', 'user', 'a reply inside the thread', turnId, true),
    ]);
    useConversationFeed(ConfigType.USER).upsertTurn(b);

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
    useConversationFeed(ConfigType.USER).upsertTurn(b);

    const wrapper = mount(TurnView, {
      props: { block: b, type: ConfigType.USER, fullThread: true },
    });

    expect(wrapper.findComponent(ActCycle).exists()).toBe(true);
    expect(wrapper.text()).toContain('opener of the panel thread');
    expect(wrapper.text()).toContain('settle0 panel reply');
    // fullThread renders the WHOLE thread, continuations included.
    expect(wrapper.text()).toContain('panel thread continuation');
  });
});
