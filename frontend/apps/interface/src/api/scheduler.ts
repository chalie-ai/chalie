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

/** An active delegate (backgrounded tool call) from /api/subagents. */
export interface ActiveSubagent {
  sub_id: string;
  /** The delegate's tool name — the row's subtitle. */
  tool_name: string;
  /** The model's act-summary of what it's doing — the row's title; may be absent. */
  summary: string | null;
  /** ISO-8601 UTC start timestamp — drives the row's elapsed timer. */
  started_at: string;
}

export const scheduler = {
  /** GET /api/subagents — running async delegates. */
  subagentsActive(): Promise<{ subagents: ActiveSubagent[] }> {
    return api.get('/api/subagents');
  },

  /**
   * DELETE /api/subagent/<subId> — cancel a running delegate. Returns
   * { ok, cancelled } on success, { ok, reason: 'not_found' } for unknown sub_id.
   */
  subagentStop(subId: string): Promise<{ ok: boolean; cancelled?: boolean; reason?: string }> {
    return api.del(`/api/subagent/${encodeURIComponent(subId)}`);
  },

  /** GET /api/scheduler/turns — active prompt-schedule threads for the dock. */
  turns(): Promise<SchedulerTurn[]> {
    return api.get('/api/scheduler/turns');
  },
};
