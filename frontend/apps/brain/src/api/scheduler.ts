/**
 * Scheduler API — endpoints derived from frontend/brain/scheduler.js fetch calls.
 *
 * GET    /scheduler              → list schedules { schedules: [...] }
 * POST   /scheduler              → create schedule
 * PUT    /scheduler/:id          → update schedule
 * DELETE /scheduler/:id          → delete/cancel schedule
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

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
    const api = useApiClient();
    return withAuth(() => api.get('/scheduler'));
  },

  create(body: ScheduleInput): Promise<{ schedule?: ScheduleItem }> {
    const api = useApiClient();
    return withAuth(() => api.post('/scheduler', body));
  },

  update(id: string | number, body: Partial<ScheduleInput>): Promise<{ schedule?: ScheduleItem }> {
    const api = useApiClient();
    return withAuth(() => api.put(`/scheduler/${id}`, body));
  },

  delete(id: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.del(`/scheduler/${id}`));
  },
};
