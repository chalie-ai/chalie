/**
 * Thread API — the /api/threads + /api/thread surface, scoped to a config type.
 *
 * The Brain Auto Research view reads the discovery loop's output through this
 * (the same thread API the interface app's chat feed uses) instead of the
 * removed /observability/research endpoints. Callers pass the ConfigType; the
 * backend resolves it to a transcript channel server-side.
 */
import { api } from '@chalie/shared';

/** One collapsed feed row from GET /api/threads. */
export interface ThreadSummary {
  turn_id: number | null;
  last_activity_at: string | null;
  preview: string;
  row_count: number;
  gist?: string | null;
  working?: boolean;
}

/** One message row inside a turn block (the subset the Brain view renders). */
export interface ThreadMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  tool_calls?: { tool_name: string; summary: string }[];
  [key: string]: unknown;
}

/** One turn's full block from GET /api/thread/<turn_id>. */
export interface TurnBlock {
  turn_id: number;
  gist: string | null;
  preview: string;
  last_activity_at: string | null;
  working: boolean;
  duration_ms: number;
  messages: ThreadMessage[];
}

export const threads = {
  /** GET /api/threads — the collapsed thread feed for a config type. */
  list(
    type: string,
    limit = 20,
    offset?: number,
  ): Promise<{ threads: ThreadSummary[]; has_more: boolean; threads_returned: number }> {
    const params = new URLSearchParams({ type, limit: String(limit) });
    if (offset != null) params.set('offset', String(offset));
    return api.get(`/api/threads?${params.toString()}`);
  },

  /** GET /api/thread/<turn_id> — one turn's full block for a config type. */
  item(turnId: number, type: string): Promise<TurnBlock> {
    return api.get(`/api/thread/${turnId}?type=${encodeURIComponent(type)}`);
  },
};
