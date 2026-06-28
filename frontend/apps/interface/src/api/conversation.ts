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

/** One row inside a turn block (see ConversationTurnBlock.messages). */
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  /**
   * The turn this row belongs to — a turn (thread) is many rows (input → steps →
   * synthesis → replies) sharing one `turn_id`; the feed groups by this. Null for
   * legacy rows written before turn tracking existed.
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
  /**
   * Set (true) on every row PAST this turn's settle0 — the reply continuation.
   * The main spine drops these (it renders only through settle0); a turn that
   * has any of them is a thread, so the feed shows its opener.
   */
  thread_message?: boolean;
}

/** Collapsed thread metadata returned by /api/threads. */
export interface ConversationThread {
  turn_id: number | null;
  last_activity_at: string | null;
  last_row_id: number;
  row_count: number;
  preview: string;
  /** Per-thread one-sentence gist (from thread_gist), null when not yet generated. */
  gist?: string | null;
  /** Unread reply count — drives the pill's "N new" badge. Absent until the
   *  backend tracks per-thread read state; the badge stays hidden until then. */
  unread?: number;
}

/**
 * One turn's full block. The single-turn read, every batch entry, and the
 * WS-refetch all return this exact shape (see backend `serialize_turn`): one
 * fetch fully determines a turn's render — its rows plus the collapsed-pill
 * metadata (gist/preview/last activity) and turn-level render state — with no
 * signal memory.
 */
export interface ConversationTurnBlock {
  turn_id: number;
  /** One-sentence gist (thread_gist), null until generated. */
  gist: string | null;
  /** First row's content, clipped — the collapsed-pill preview. */
  preview: string;
  last_activity_at: string | null;
  /** True while the turn is unsettled/in-flight (no settle0 row yet). */
  working: boolean;
  /** Row-span duration in ms (0 for a single-row turn). */
  duration_ms: number;
  messages: ConversationMessage[];
}

export const conversation = {
  /**
   * GET /api/threads — collapsed thread metadata for the feed (id +
   * gist/preview/last activity). Returns the `limit` most-recently-active
   * threads; `threads_returned` advances pagination.
   */
  threads(
    limit = 20,
    offset?: number,
  ): Promise<{ threads: ConversationThread[]; has_more: boolean; threads_returned: number }> {
    const q = new URLSearchParams({ limit: String(limit) });
    if (offset != null) q.set('offset', String(offset));
    return api.get(`/api/threads?${q.toString()}`);
  },

  /** GET /api/thread/<turn_id> — one turn's full block (expand + WS refetch). */
  thread(turnId: number): Promise<ConversationTurnBlock> {
    return api.get(`/api/thread/${turnId}`);
  },

  /** GET /api/threads/batch?id[]= — many blocks in one round-trip, a pure
   *  concatenation of the single-turn getter over the given ids (a read → GET). */
  batch(turnIds: number[]): Promise<{ blocks: ConversationTurnBlock[] }> {
    const q = new URLSearchParams();
    for (const id of turnIds) q.append('id[]', String(id));
    return api.get(`/api/threads/batch?${q.toString()}`);
  },
};
