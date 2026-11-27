// @vitest-environment happy-dom
/**
 * A sent message that opens with quote blocks, read back in the chat — feature
 * spec. The message comes back from the server exactly as it was sent; the
 * chat draws its blocks first, each on its own fill with the quote over the
 * reply, then a faint divider, then the typed text alone. A message without
 * blocks draws as it always has.
 *
 * Real component tree (TurnView → UserBubble → QuoteBlocks), real Pinia; the
 * turn is mounted straight in, as it is fetched off the wire.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { ConfigType } from '@chalie/shared';
import TurnView from './TurnView.vue';
import type { ConversationMessage, ConversationTurnBlock } from '../../api/conversation';

/** Two blocks as the server stores them: the browser sent each line break as CR LF. */
const QUOTED =
  '[quote_block]{"quote": "Paris is the capital", "reply": "and Lyon?"}[/quote_block]\r\n' +
  '[quote_block]{"quote": "two\\nlines", "reply": ""}[/quote_block]';

function row(id: string, role: ConversationMessage['role'], content: string): ConversationMessage {
  return {
    id,
    role,
    content,
    timestamp: '2026-01-01 00:00:00',
    day: '2026-01-01',
    created_at: '2026-01-01T00:00:00Z',
    turn_id: 31,
  };
}

function turn(userContent: string): ConversationTurnBlock {
  return {
    turn_id: 31,
    gist: null,
    preview: userContent,
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    messages: [row('310', 'user', userContent), row('311', 'assistant', 'Lyon is the third city.')],
    type: 'user',
  };
}

let wrapper: VueWrapper | null = null;

function render(userContent: string): HTMLElement {
  wrapper = mount(TurnView, { props: { block: turn(userContent), type: ConfigType.USER, fullThread: false } });
  return wrapper.get('.user-message').element as HTMLElement;
}

/** The bubble's parts in the order they draw, by what each one is. */
function parts(bubble: HTMLElement): string[] {
  return [...bubble.children].map((el) => el.className.split(' ')[0]);
}

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe('a sent message with quote blocks, read back', () => {
  it('draws the blocks first, then a divider, then the typed text alone', () => {
    const bubble = render(`${QUOTED}\r\n\r\nAlso, what is the weather?`);

    expect(parts(bubble)).toEqual(['quote-blocks', 'user-message__divider', 'user-text']);
    const items = [...bubble.querySelectorAll('.quote-blocks__item')];
    expect(items.map((item) => item.querySelector('.quote-blocks__quote')?.textContent)).toEqual([
      'Paris is the capital',
      'two\nlines',
    ]);
    expect(items.map((item) => item.querySelector('.quote-blocks__reply')?.textContent ?? null)).toEqual([
      'and Lyon?',
      null,
    ]);
    expect(bubble.querySelector('.user-text')?.textContent?.trim()).toBe('Also, what is the weather?');
    // No marker leaks into what the user reads.
    expect(bubble.textContent).not.toContain('quote_block');
  });

  it('draws a message of blocks alone as the blocks, with no divider and no empty text', () => {
    const bubble = render(QUOTED);
    expect(parts(bubble)).toEqual(['quote-blocks']);
  });

  it('draws a message without blocks as plain text, as before', () => {
    const bubble = render('What is the weather?');
    expect(parts(bubble)).toEqual(['user-text']);
    expect(bubble.querySelector('.user-text')?.textContent?.trim()).toBe('What is the weather?');
  });
});
