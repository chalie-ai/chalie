import type { ConversationMessage, ConversationTurnBlock } from '../api/conversation';
import { extractText } from '../composables/useMarkup';

/**
 * Plain-text for one ConversationMessage — segments concatenated, else the
 * single content path, both markup-stripped. Shared by the copy button and
 * turn-level speech aggregation so the two never diverge. Speech synthesis
 * does NOT read this: the backend reads the settled row itself, so the browser
 * never assembles text for the voice pipeline.
 */
export function messagePlaintext(msg: ConversationMessage): string {
  if (msg.segments?.length) {
    return extractText(msg.segments.map((s) => s.content ?? s.synthesis ?? '').join(' '));
  }
  return extractText(msg.content ?? '');
}

/**
 * TTS plaintext of every assistant row in a turn block — the unfocused-tab
 * background notification's source (session._finishTurn); reads straight off
 * a freshly-fetched block rather than a cached copy, since the DOM contract
 * holds no turn data of its own.
 */
export function blockSpeechText(block: ConversationTurnBlock): string {
  return block.messages
    .filter((m) => m.role === 'assistant' && !!(m.content || m.segments?.length))
    .map(messagePlaintext)
    .filter(Boolean)
    .join(' ');
}
