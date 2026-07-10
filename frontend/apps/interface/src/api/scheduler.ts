import { api } from '@chalie/shared';

/**
 * One active prompt-schedule thread, collapsed to its turn_id (§13.5).
 * Mirrors backend SchedulerTurn DTO exactly. `minute`/`hour`/`day`/`month`/
 * `weekday` are the schedule's crontab field expressions (`*` = every).
 */
export interface SchedulerTurn {
  turn_id: number;
  gist: string | null;
  preview: string;
  minute: string;
  hour: string;
  day: string;
  month: string;
  weekday: string;
}

/** An active delegate (backgrounded tool call) row from GET /api/subagents/all. */
export interface ActiveSubagent {
  sub_id: string;
  /** The delegate's tool name — the row's subtitle. */
  tool_name: string;
  /** The model's act-summary of what it's doing — the row's title; may be absent. */
  summary: string | null;
  /** ISO-8601 UTC start timestamp — drives the row's elapsed timer. */
  started_at: string;
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

export const scheduler = {
  /**
   * GET /api/subagents/all — running async delegates. Backed by
   * backend/api/endpoints/subagents.py; envelope is
   * { success, result: [...], pagination }, fetched at the backend's max
   * page size (100 — comfortably above any realistic concurrent-delegate count).
   */
  async subagentsActive(): Promise<{ subagents: ActiveSubagent[] }> {
    const body = await api.get<ListingEnvelope<ActiveSubagent>>('/api/subagents/all?limit=100');
    return { subagents: body.result };
  },

  /**
   * DELETE /api/subagents/<subId> — cancel a running delegate. Envelope is
   * { success, result: { cancelled?, reason? } }; `cancelled: true` on
   * success, `reason: 'not_found'` for an unknown/finished sub_id (still 200
   * — see backend/api/endpoints/subagents.py).
   */
  async subagentStop(subId: string): Promise<{ cancelled?: boolean; reason?: string }> {
    const body = await api.del<SingleEnvelope<{ cancelled?: boolean; reason?: string }>>(
      `/api/subagents/${encodeURIComponent(subId)}`,
    );
    return body.result;
  },

  /** GET /api/scheduler/turns — active prompt-schedule threads for the dock. */
  turns(): Promise<SchedulerTurn[]> {
    return api.get('/api/scheduler/turns');
  },
};
