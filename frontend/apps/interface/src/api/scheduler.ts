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

/** An active subagent entry from /chat/subagents/active. */
export interface ActiveSubagent {
  sub_id: string;
}

export const scheduler = {
  /** GET /scheduler?status=pending — list pending scheduled items. */
  pending(): Promise<{ items: ScheduledItem[]; total: number; limit: number; offset: number }> {
    return api.get('/scheduler?status=pending');
  },

  /** GET /chat/subagents/active — running async delegates. */
  subagentsActive(): Promise<{ subagents: ActiveSubagent[] }> {
    return api.get('/chat/subagents/active');
  },

  /**
   * POST /chat/subagent/<subId>/stop — cancel a running delegate. Returns
   * { ok, cancelled } on success, { ok, reason: 'not_found' } for unknown sub_id.
   */
  subagentStop(subId: string): Promise<{ ok: boolean; cancelled?: boolean; reason?: string }> {
    return api.post(`/chat/subagent/${encodeURIComponent(subId)}/stop`, {});
  },
};
