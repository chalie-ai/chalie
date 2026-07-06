/**
 * Scheduler API — endpoints for managing scheduled prompts.
 *
 * GET    /api/scheduler          → list schedules (one row per series)
 * POST   /api/scheduler          → create schedule (enabled is a field on the body)
 * PUT    /api/scheduler/:id      → update schedule, incl. enable/disable
 * DELETE /api/scheduler/:id      → delete/cancel schedule
 */
import { api } from '@chalie/shared';

export interface ScheduleItem {
  id: number;
  message?: string | null;
  prompt?: string | null;
  enabled: number;
  start_at?: string | null;
  day: number | null;
  hour: number | null;
  minute: number | null;
  channel?: string | null;
  created_by_session?: string | null;
  created_at?: string;
  [key: string]: unknown;
}

export interface ScheduleInput {
  message: string;
  start_at?: string;
  day: number | null;
  hour: number | null;
  minute: number | null;
  enabled?: boolean;
}

export const scheduler = {
  list(): Promise<ScheduleItem[]> {
    return api.get('/api/scheduler');
  },

  create(body: ScheduleInput): Promise<{ schedule?: ScheduleItem }> {
    return api.post('/api/scheduler', body);
  },

  update(id: number, body: Partial<ScheduleInput>): Promise<{ schedule?: ScheduleItem }> {
    return api.put(`/api/scheduler/${id}`, body);
  },

  delete(id: number): Promise<unknown> {
    return api.del(`/api/scheduler/${id}`);
  },
};
