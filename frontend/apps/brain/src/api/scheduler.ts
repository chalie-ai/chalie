/**
 * Scheduler API — endpoints derived from frontend/brain/scheduler.js fetch calls.
 *
 * GET    /scheduler              → list schedules { schedules: [...] }
 * POST   /scheduler              → create schedule
 * PUT    /scheduler/:id          → update schedule
 * DELETE /scheduler/:id          → delete/cancel schedule
 */
import { api } from '@chalie/shared';

export interface ScheduleItem {
  id: string | number;
  message?: string | null;
  prompt?: string | null;
  status: string;
  due_at?: string | null;
  due?: string | null;
  recurrence?: string | null;
  type?: string | null;
  item_type?: string | null;
  [key: string]: unknown;
}

export interface ScheduleInput {
  message: string;
  due_at: string;
  type?: string;
  recurrence?: string | null;
}

export const scheduler = {
  list(): Promise<{ schedules?: ScheduleItem[]; items?: ScheduleItem[] }> {
    return api.get('/api/scheduler');
  },

  create(body: ScheduleInput): Promise<{ schedule?: ScheduleItem }> {
    return api.post('/api/scheduler', body);
  },

  update(id: string | number, body: Partial<ScheduleInput>): Promise<{ schedule?: ScheduleItem }> {
    return api.put(`/api/scheduler/${id}`, body);
  },

  delete(id: string | number): Promise<unknown> {
    return api.del(`/api/scheduler/${id}`);
  },
};
