import { api } from '@chalie/shared';

/** A single attachment served from /api/files/preview/<path> (URL is backend-provided). */
export interface ConversationAttachment {
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
   * The chips THIS row anchors — present on WHATEVER row drove the tools,
   * including the user input row (a step-1 tool-only call anchors there before
   * any assistant text). Each chip carries the ability's persisted `act_summary`
   * plus its persisted lifecycle `state` (started/done/error) + `ended_at`, so a
   * refetched error stays a red pill instead of downgrading to a neutral chip.
   * The refresh path renders them as a collapsed group beneath the row,
   * mirroring how the live path collapses a superseded step.
   */
  tool_calls?: { tool_name: string; summary: string; state: string; ended_at: string | null }[];
  /**
   * Set (true) on every row PAST this turn's settle0 — the reply continuation.
   * The main spine drops these (it renders only through settle0); a turn that
   * has any of them is a thread, so the feed shows its opener.
   */
  thread_message?: boolean;
  /**
   * True on the ONE assistant row that closes a turn_execution — the final
   * reply. Only that row is spoken, so it is the only one that carries a
   * speaker button.
   */
  settled?: boolean;
  /**
   * Pre-synthesis outcome for a settled row, as of this fetch: 'ready' (audio
   * stored), 'failed' (gave up after its attempts), or absent/null for a row
   * with no attempt on record — history from before pre-synthesis, whose first
   * press starts the pipeline. Live changes arrive as `voice_transcript` WS
   * frames instead (see stores/voiceTranscripts.ts).
   */
  voice_state?: 'pending' | 'ready' | 'failed' | null;
}

/** Collapsed thread metadata returned by /api/threads/all. */
export interface ConversationThread {
  turn_id: number | null;
  last_activity_at: string | null;
  row_count: number;
  preview: string;
  /** Per-thread one-sentence gist (from thread_gist), null when not yet generated. */
  gist?: string | null;
  /**
   * The ConfigType identity (`user`/`scheduled`/`discovery`) this summary was
   * resolved under. turn_id is only unique PER TYPE, so a caller that opens a
   * thread from this list must carry this forward rather than assume `user`.
   */
  type: string;
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
  /**
   * True when the turn's most recent execution ended CRASHED (an unhandled step
   * exception, or a process death the boot sweep stamped). A crash that produced
   * no reply row settles to working:false with no other trace, so this is the
   * only signal the render has to show an "ended unexpectedly" note rather than
   * a bare tool-trace footer. Absent (undefined) on legacy/non-crashed blocks.
   */
  crashed?: boolean;
  /** Row-span duration in ms (0 for a single-row turn). */
  duration_ms: number;
  messages: ConversationMessage[];
  /**
   * The ConfigType identity (`user`/`scheduled`/`discovery`) this block was
   * resolved under — turn_id is only unique PER TYPE, so a refetch must trust
   * this over whatever type it requested with. Required on the backend DTO,
   * so required here: a missing stamp is a contract violation, not a case to
   * fall back from.
   */
  type: string;
}

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const conversation = {
  /**
   * GET /api/threads/all — collapsed thread metadata for the feed (id +
   * gist/preview/last activity). Returns the `limit` most-recently-active
   * threads; `threads_returned` advances pagination. With `query`, the same
   * getter filters to matching threads (the search overlay's source).
   *
   * `type` (the ProcessorConfig identity) is forwarded to the backend, which
   * resolves it to a transcript channel; defaults to "user" server-side when omitted.
   *
   * The backend uses page/limit pagination; callers pass offset (default 0).
   * We derive `page = floor(offset/limit)+1` and recompute `has_more` from the
   * server-supplied pagination.total so the consumer's shape is unchanged.
   */
  threads(
    limit = 20,
    offset?: number,
    query?: string,
    type?: string,
  ): Promise<{ threads: ConversationThread[]; has_more: boolean; threads_returned: number }> {
    const page = Math.floor((offset ?? 0) / limit) + 1;
    const params = new URLSearchParams({ limit: String(limit), page: String(page) });
    if (query) params.set('q', query);
    if (type) params.set('type', type);
    return api.get<ListingEnvelope<ConversationThread>>(`/api/threads/all?${params.toString()}`).then((body) => {
      const threads = body.result;
      const has_more = page * limit < body.pagination.total;
      return { threads, has_more, threads_returned: threads.length };
    });
  },

  /**
   * GET /api/threads/<turn_id> — one turn's full block (expand + WS refetch).
   *
   * `type` is forwarded when supplied; the backend resolves it to a transcript
   * channel and defaults to "user" when omitted.
   */
  thread(turnId: number, type?: string): Promise<ConversationTurnBlock> {
    const params = type ? `?type=${encodeURIComponent(type)}` : '';
    return api.get<SingleEnvelope<ConversationTurnBlock>>(`/api/threads/${turnId}${params}`).then((body) => body.result);
  },

  /**
   * GET /api/threads/batch?id[]= — many blocks in one round-trip, a pure
   * concatenation of the single-turn getter over the given ids (a read → GET).
   *
   * `type` is forwarded when supplied; the backend resolves it to a transcript channel.
   */
  batch(turnIds: number[], type?: string): Promise<{ blocks: ConversationTurnBlock[] }> {
    const q = new URLSearchParams();
    for (const id of turnIds) q.append('id[]', String(id));
    if (type) q.set('type', type);
    return api.get<SingleEnvelope<{ blocks: ConversationTurnBlock[] }>>(`/api/threads/batch?${q.toString()}`).then((body) => body.result);
  },
};
