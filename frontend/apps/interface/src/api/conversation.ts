import { api } from '@chalie/shared';

/** A single attachment served from /documents/<id>/preview. */
export interface ConversationAttachment {
  doc_id: string;
  filename: string;
  mime_type: string;
  is_image: boolean;
  url: string;
}

/** A rich-media or plain-text content segment inside an assistant turn. */
export interface ConversationSegment {
  type: 'text' | 'rich';
  content?: string;
  tag?: string;
  payload?: Record<string, unknown>;
  synthesis?: string;
}

/** A single turn returned by /conversation/recent. */
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  /**
   * The turn this row belongs to. Under the chain model a turn is many rows
   * (input → step rows → final synthesis row) sharing one `turn_id`; the feed
   * groups by this. Null for legacy rows written before turn tracking existed.
   */
  turn_id: number | null;
  /** Present on user turns when attachments were uploaded. */
  attachments?: ConversationAttachment[];
  /** Present on assistant turns — one or more content segments. */
  segments?: ConversationSegment[];
  /**
   * Present on assistant turns that drove tools — the chips THIS row emitted,
   * each carrying the ability's persisted `act_summary`. Under the chain model a
   * turn is many assistant rows, and each row owns its own tools; the refresh
   * path renders these as a collapsed (summary-only) tool group beneath the row,
   * mirroring how the live path collapses a step once it is superseded.
   */
  tool_calls?: { tool_name: string; summary: string }[];
}

export const conversation = {
  /**
   * Fetch recent conversation turns.
   * @param limit  Max turns to return (1–120, default 12).
   * @param offset Turn offset for scroll-up pagination — the backend
   *               (`/conversation/recent`) reads `offset`, counting back whole
   *               turns from the newest. `turns_returned` reports how many turns
   *               this page actually held, so the caller advances by turns.
   */
  recent(
    limit = 12,
    offset?: number,
  ): Promise<{ messages: ConversationMessage[]; has_more: boolean; turns_returned: number }> {
    const q = new URLSearchParams({ limit: String(limit) });
    if (offset != null) q.set('offset', String(offset));
    return api.get(`/conversation/recent?${q.toString()}`);
  },
};
