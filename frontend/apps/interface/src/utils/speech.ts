import type { ConversationMessage } from '../api/conversation';
import { extractText } from '../composables/useMarkup';

/**
 * Plain-text (TTS) for one ConversationMessage — segments concatenated, else
 * the single content path, both markup-stripped. Shared by the speak button and
 * turn-level speech aggregation so the two never diverge.
 */
export function messagePlaintext(msg: ConversationMessage): string {
  if (msg.segments?.length) {
    return extractText(msg.segments.map((s) => s.content ?? s.synthesis ?? '').join(' '));
  }
  return extractText(msg.content ?? '');
}
