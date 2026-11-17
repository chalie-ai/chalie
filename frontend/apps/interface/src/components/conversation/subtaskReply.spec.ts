// @vitest-environment happy-dom
/**
 * A subtask's reply — feature spec.
 *
 * A subtask is fire-and-forget: it outlives the call that started it, so its
 * answer lands later on the CALLER's channel as a user-role row carrying
 * `speaker: 'Reply from subtask #N'` and `reply_from`, the subagent's tool and
 * the one-line summary of the call that started it. The person reads that
 * reply as `Reply from <b>{summary}</b> agent` — the type's plain name in
 * place of a missing summary — and never the tool name or the number. Such a
 * reply on the user/schedule channel (a typed block — `channel` null) must
 * read as a collapsed chip: folded to its label, its text hidden until a click
 * opens it, another folds it back. It is the ONLY collapsible case: a subtask transcript's own
 * input rows (channel `delegate:subtask`, captioned 'Task from Chalie' /
 * 'Message from Chalie') and every other delegate transcript's rows keep the
 * plain caption and always show their text.
 *
 * Real component tree throughout (TurnView → UserBubble), real Pinia, no mocks
 * — the block is mounted straight in, exactly as it comes off the wire.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import TurnView from './TurnView.vue';
import type { ConversationMessage, ConversationTurnBlock, DelegateTurnBlock } from '../../api/conversation';

const TURN = 601;

let mounted: VueWrapper[] = [];

function track<T extends VueWrapper>(wrapper: T): T {
  mounted.push(wrapper);
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
});

afterEach(() => {
  for (const w of mounted) w.unmount();
  mounted = [];
  document.body.innerHTML = '';
});

/** Any row, with the optional fields a stored row can carry (speaker among
 *  them) — the shape the REST read returns it in. */
function row(
  id: string,
  role: ConversationMessage['role'],
  content: string,
  turnId: number,
  extra: Partial<ConversationMessage> = {},
): ConversationMessage {
  return {
    id,
    role,
    content,
    timestamp: '2026-01-01 00:00:00',
    day: '2026-01-01',
    created_at: '2026-01-01T00:00:00Z',
    turn_id: turnId,
    ...extra,
  };
}

/** A settled typed block on the user channel. */
function userBlock(turnId: number, messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: messages[0]?.content ?? '',
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    messages,
    type: 'user',
  };
}

/** A settled delegate transcript, read by its channel. */
function delegateBlock(channel: string, turnId: number, messages: ConversationMessage[]): DelegateTurnBlock {
  return {
    turn_id: turnId,
    gist: null,
    preview: messages[0]?.content ?? '',
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    messages,
    type: null,
    channel,
  };
}

/** The reply chip's label: its whole text, and the text it puts in bold. */
function replyLabel(wrapper: VueWrapper, speaker: string): { text: string; bold: string[] } {
  const label = wrapper.find(`.user-message[data-speaker="${speaker}"] button.user-message__label--toggle`);
  expect(label.exists()).toBe(true);
  return { text: label.text(), bold: label.findAll('b').map((b) => b.text()) };
}

describe('a subtask reply landing on the user channel', () => {
  it('is a collapsed chip — folded to "Reply from <b>{summary}</b> agent", a click opens it, another folds it', async () => {
    const wrapper = track(mount(TurnView, {
      props: {
        block: userBlock(TURN, [
          row('6010', 'user', 'look into the museum hours', TURN),
          // The subtask's answer arrives later, on the user's own channel: a
          // user-role row naming its subtask.
          row('6011', 'user', 'The answer.', TURN, {
            speaker: 'Reply from subtask #4',
            reply_from: { tool_name: 'subtask', summary: 'compare museum hours' },
          }),
        ]),
      },
      // Attach to the document: happy-dom only resolves an inline `display:
      // none` through getComputedStyle for a connected element, and
      // isVisible() walks the computed style up the tree.
      attachTo: document.body,
    }));

    const bubble = wrapper.find('.user-message[data-speaker="Reply from subtask #4"]');
    expect(bubble.exists()).toBe(true);
    const label = bubble.find('button.user-message__label--toggle');
    expect(label.exists()).toBe(true);
    expect(label.attributes('type')).toBe('button');
    expect(label.attributes('aria-expanded')).toBe('false');
    expect(label.text()).toBe('Reply from compare museum hours agent');
    expect(label.findAll('b').map((b) => b.text())).toEqual(['compare museum hours']);
    // Neither the tool name nor the subtask's number shows.
    expect(label.text()).not.toMatch(/subtask|\d/);
    // Folded: the reply's text is not visible yet.
    const text = bubble.find('.user-text');
    expect(text.isVisible()).toBe(false);

    await label.trigger('click');
    await wrapper.vm.$nextTick();
    expect(label.attributes('aria-expanded')).toBe('true');
    expect(text.isVisible()).toBe(true);
    expect(text.text()).toBe('The answer.');

    await label.trigger('click');
    await wrapper.vm.$nextTick();
    expect(label.attributes('aria-expanded')).toBe('false');
    expect(text.isVisible()).toBe(false);
  });
});

describe('a reply whose call carries no summary', () => {
  it('reads "Reply from <b>{type}</b> agent" — the plain type name, no tool name, no number — for every kind of subagent', () => {
    const kinds: [string, string][] = [
      ['subtask', 'Subtask'],
      ['web_search', 'Web search'],
      ['web_browse', 'Web browsing'],
      ['pim', 'Mail & calendar'],
      ['code_agent', 'Coding'],
    ];
    const wrapper = track(mount(TurnView, {
      props: {
        block: userBlock(TURN, [
          row('6020', 'user', 'plan my museum day', TURN),
          ...kinds.map(([tool], i) => row(`602${i + 1}`, 'user', 'The answer.', TURN, {
            speaker: `Reply from ${tool} #${i + 7}`,
            reply_from: { tool_name: tool, summary: i === 1 ? '  ' : '' },
          })),
        ]),
      },
      attachTo: document.body,
    }));

    kinds.forEach(([tool, name], i) => {
      const { text, bold } = replyLabel(wrapper, `Reply from ${tool} #${i + 7}`);
      expect(text).toBe(`Reply from ${name} agent`);
      expect(bold).toEqual([name]);
      expect(text).not.toMatch(/_|\d/);
      expect(text).not.toContain(tool);
    });
  });
});

describe('a reply whose call no longer exists', () => {
  it('still reads as its speaker label, folded', () => {
    const wrapper = track(mount(TurnView, {
      props: {
        block: userBlock(TURN, [
          row('6030', 'user', 'look into the museum hours', TURN),
          row('6031', 'user', 'The answer.', TURN, { speaker: 'Reply from a delegate' }),
        ]),
      },
      attachTo: document.body,
    }));

    const { text, bold } = replyLabel(wrapper, 'Reply from a delegate');
    expect(text).toBe('Reply from a delegate');
    expect(bold).toEqual([]);
  });
});

describe('a subtask transcript (channel delegate:subtask)', () => {
  it('shows its speaker-captions as plain labels, never a toggle, with the text always visible', () => {
    const wrapper = track(mount(TurnView, {
      props: {
        block: delegateBlock('delegate:subtask', 701, [
          row('7010', 'user', 'find the museum hours', 701, { speaker: 'Task from Chalie' }),
          row('7011', 'user', 'the answer is nine', 701, { speaker: 'Message from Chalie' }),
        ]),
        fullThread: true,
      },
      attachTo: document.body,
    }));

    const labels = wrapper.findAll('.user-message__label');
    expect(labels.map((l) => l.text())).toEqual(['Task from Chalie', 'Message from Chalie']);
    // A caption on a transcript is a plain span — nothing to fold, no button.
    for (const label of labels) expect(label.element.tagName).toBe('SPAN');
    expect(wrapper.find('.user-message__label--toggle').exists()).toBe(false);
    const texts = wrapper.findAll('.user-text');
    expect(texts.map((t) => t.isVisible())).toEqual([true, true]);
  });
});

describe('a delegate transcript on another channel', () => {
  it('still captions its speaker-less user row as a task from Chalie', () => {
    const wrapper = track(mount(TurnView, {
      props: {
        block: delegateBlock('delegate:web_search', 801, [
          row('8010', 'user', 'look up the museum hours', 801),
        ]),
        fullThread: true,
      },
      attachTo: document.body,
    }));

    const label = wrapper.find('.user-message__label');
    expect(label.exists()).toBe(true);
    expect(label.element.tagName).toBe('SPAN');
    expect(label.text()).toBe('Task from Chalie');
  });
});
