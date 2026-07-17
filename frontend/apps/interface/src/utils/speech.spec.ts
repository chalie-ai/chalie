// @vitest-environment happy-dom
/**
 * speech — feature spec for the unfocused-tab background notification's
 * plaintext source (session._finishTurn → blockSpeechText). Real
 * `extractText` (composables/useMarkup.ts) runs unmocked — it needs a real
 * DOM to parse markup through a <template>, hence happy-dom.
 */
import { describe, expect, it } from 'vitest';
import type { ConversationMessage, ConversationTurnBlock } from '../api/conversation';
import { blockSpeechText } from './speech';

function block(messages: ConversationMessage[]): ConversationTurnBlock {
  return {
    turn_id: 1,
    gist: null,
    preview: '',
    last_activity_at: null,
    working: false,
    duration_ms: 0,
    messages,
    type: 'user',
  };
}

function msg(role: ConversationMessage['role'], overrides: Partial<ConversationMessage> = {}): ConversationMessage {
  return { id: '1', role, content: '', timestamp: '2026-01-01 00:00:00', turn_id: 1, ...overrides };
}

describe('blockSpeechText', () => {
  it('reads only assistant rows, skipping the user turn that prompted them', () => {
    const b = block([
      msg('user', { content: 'what is the weather' }),
      msg('assistant', { content: 'it is sunny today' }),
    ]);
    expect(blockSpeechText(b)).toBe('it is sunny today');
  });

  it('joins a segmented assistant row\'s segments into one string', () => {
    const b = block([
      msg('assistant', {
        segments: [
          { type: 'text', content: 'first part.' },
          { type: 'text', content: 'second part.' },
        ],
      }),
    ]);
    expect(blockSpeechText(b)).toBe('first part. second part.');
  });

  it('returns empty string for a block with no speakable assistant content', () => {
    expect(blockSpeechText(block([]))).toBe('');
    expect(blockSpeechText(block([msg('user', { content: 'only a user row' })]))).toBe('');
  });
});
