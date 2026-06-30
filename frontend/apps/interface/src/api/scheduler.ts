import { api } from '@chalie/shared';

/** A single scheduled item from the scheduler API. */
export interface ScheduledItem {
  id: string;
  item_type: string;
  message: string;
  due_at: string | null;
  recurrence: string | null;
  window_start: string | null;
  window_end: string | null;
  status: string;
  channel: string | null;
  created_by_session: string | null;
  created_at: string | null;
  last_fired_at: string | null;
  group_id: string | null;
  is_prompt: boolean;
  source: string | null;
  external_uid: string | null;
}

/**
 * One active prompt-schedule thread, collapsed to its turn_id (§13.5).
 * Mirrors backend SchedulerTurn DTO exactly. Datetimes arrive as ISO-8601
 * UTC strings (the backend's DTO base serialises them that way).
 */
export interface SchedulerTurn {
  turn_id: number;
  gist: string | null;
  preview: string;
  recurrence: string | null;
  last_fired_at: string | null;
  next_due_at: string | null;
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
  /** GET /api/scheduler?status=pending — list pending scheduled items. */
  pending(): Promise<ScheduledItem[]> {
    return api.get('/api/scheduler?status=pending');
  },

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
