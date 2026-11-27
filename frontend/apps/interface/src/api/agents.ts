import { api } from '@chalie/shared';

/** How an agent stands: running, waiting for its caller's reply, or how it
 *  ended. */
export type AgentStatus = 'running' | 'waiting' | 'done' | 'cancelled' | 'failed';

/**
 * One subagent some turn spawned, as the server sends it — the same fields on
 * `GET /api/agents/all` and on every `agent_changed` frame. `type` is the
 * spawn tool's name, `channel` and `turn_id` the subagent's own turn, and
 * `summary` the one-line summary of the call that started it (empty when the
 * call gave none).
 */
export interface Agent {
  agent_id: number;
  spawned_by_channel: string;
  spawned_in_turn: number;
  type: string;
  prompt: string;
  status: AgentStatus;
  response_id: number | null;
  turn_id: number;
  channel: string;
  summary: string;
}

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

/** The most rows the server returns on one page. */
const PAGE_LIMIT = 100;

export const agents = {
  /**
   * GET /api/agents/all — every agent still running or waiting for a reply,
   * oldest first, read page by page until the listing's total is reached.
   */
  async active(): Promise<Agent[]> {
    const rows: Agent[] = [];
    for (let page = 1; ; page++) {
      const body = await api.get<ListingEnvelope<Agent>>(`/api/agents/all?page=${page}&limit=${PAGE_LIMIT}`);
      rows.push(...body.result);
      if (body.result.length === 0 || rows.length >= body.pagination.total) return rows;
    }
  },
};
