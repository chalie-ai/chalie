/**
 * Scheduler API — thin wrapper over the Endpoint-contract routes in
 * backend/api/endpoints/scheduler.py + backend/api/actions/scheduler/turns.py.
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 *
 *   GET    /api/scheduler/all           paginated listing (fetchAllPages).
 *   POST   /api/scheduler/-1            create schedule → 201, result DTO.
 *   POST   /api/scheduler/<id>          update schedule → result DTO (legacy PUT → POST).
 *   DELETE /api/scheduler/<id>          → 204 empty body.
 */
import { api } from '@chalie/shared';
import { fetchAllPages } from './paginate';

export interface ScheduleItem {
  id: number;
  message?: string | null;
  prompt?: string | null;
  enabled: number;
  start_at?: string | null;
  /** Crontab field expressions mirroring cron_minute/hour/dom/month/dow (`*` = every). */
  minute: string;
  hour: string;
  day: string;
  month: string;
  weekday: string;
  channel?: string | null;
  created_by_session?: string | null;
  created_at?: string;
  [key: string]: unknown;
}

export interface ScheduleInput {
  message: string;
  start_at?: string;
  /** Crontab field expressions — values, ranges, steps and comma-unions; `*` = every. */
  minute: string;
  hour: string;
  day: string;
  month: string;
  weekday: string;
  enabled?: boolean;
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const scheduler = {
  list(): Promise<ScheduleItem[]> {
    return fetchAllPages<ScheduleItem>('/api/scheduler/all');
  },

  async create(body: ScheduleInput): Promise<ScheduleItem> {
    const res = await api.post<SingleEnvelope<ScheduleItem>>('/api/scheduler/-1', body);
    return res.result;
  },

  async update(id: number, body: Partial<ScheduleInput>): Promise<ScheduleItem> {
    const res = await api.post<SingleEnvelope<ScheduleItem>>(`/api/scheduler/${id}`, body);
    return res.result;
  },

  delete(id: number): Promise<unknown> {
    return api.del(`/api/scheduler/${id}`);
  },
};
