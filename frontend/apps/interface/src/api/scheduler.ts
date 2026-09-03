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

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

export const scheduler = {
  /**
   * GET /api/scheduler/turns — active prompt-schedule threads for the dock.
   * Backed by backend/api/actions/scheduler/turns.py; envelope is
   * { success, result: [...], pagination }.
   */
  async turns(): Promise<SchedulerTurn[]> {
    const body = await api.get<ListingEnvelope<SchedulerTurn>>('/api/scheduler/turns');
    return body.result;
  },
};
