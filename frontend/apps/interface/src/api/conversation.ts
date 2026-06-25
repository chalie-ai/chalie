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
   * The turn this row belongs to — under the chain model a turn is many rows
   * (input → steps → synthesis) sharing one `turn_id`; the feed groups by this.
   * Null for legacy rows written before turn tracking existed.
   */
  turn_id: number | null;
  /** Present on user turns when attachments were uploaded. */
  attachments?: ConversationAttachment[];
  /** Present on assistant turns — one or more content segments. */
  segments?: ConversationSegment[];
  /**
   * Present on assistant turns that drove tools — the chips THIS row emitted,
   * each with the ability's persisted `act_summary`. Each assistant row owns its
   * own tools; the refresh path renders them as a collapsed (summary-only) group
   * beneath the row, mirroring how the live path collapses a superseded step.
   */
  tool_calls?: { tool_name: string; summary: string }[];
}

/** Collapsed thread metadata returned by /conversation/threads. */
export interface ConversationThread {
  turn_id: number | null;
  last_activity_at: string | null;
  last_row_id: number;
  row_count: number;
  preview: string;
}

export const conversation = {
  /**
   * GET /conversation/recent — `offset` counts back whole TURNS (not rows) from
   * the newest (limit 1–120); `turns_returned` reports the page size so the
   * caller advances pagination by turns.
   */
  recent(
    limit = 12,
    offset?: number,
  ): Promise<{ messages: ConversationMessage[]; has_more: boolean; turns_returned: number }> {
    const q = new URLSearchParams({ limit: String(limit) });
    if (offset != null) q.set('offset', String(offset));
    return api.get(`/conversation/recent?${q.toString()}`);
  },

  /**
   * GET /conversation/threads — collapsed thread metadata for the thread feed.
   * Returns the `limit` most-recently-active threads; `threads_returned` advances
   * pagination.
   */
  threads(
    limit = 50,
    offset?: number,
  ): Promise<{ threads: ConversationThread[]; has_more: boolean; threads_returned: number }> {
    const q = new URLSearchParams({ limit: String(limit) });
    if (offset != null) q.set('offset', String(offset));
    return api.get(`/conversation/threads?${q.toString()}`);
  },

  /**
   * GET /conversation/thread/<turn_id> — full row set of one thread (expand).
   */
  thread(turnId: number): Promise<{ messages: ConversationMessage[]; turn_id: number }> {
    return api.get(`/conversation/thread/${turnId}`);
  },
};
