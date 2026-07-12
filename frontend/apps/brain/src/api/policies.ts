/**
 * Policies API — thin wrapper over the new Endpoint-contract routes.
 *
 * Envelope: listing responses are { success, result: [...], pagination };
 * mutations answer 204 with an empty body (the shared client returns
 * undefined for those).
 *
 * Routes consumed here:
 *   GET    /api/policies/all?page=N&limit=M  paginated listing (fetchAllPages).
 *   POST   /api/policies/-1                  upsert one { channel, permission,
 *                                           setting } → 204.
 *   GET    /api/policies/blocked             blocked-action log (listing;
 *                                           backend caps at its default of
 *                                           the 50 most recent entries).
 */
import { api } from '@chalie/shared';
import { fetchAllPages, type ListingEnvelope } from './paginate';

export interface PolicyRow {
  channel: string;
  permission: string;
  setting: string;
  label?: string;
  group?: string;
}

export interface BlockedEntry {
  action_id: number;
  context: string;
  reason: string;
  created_at: string;
  [key: string]: unknown;
}

export interface PolicyUpdate {
  channel: string;
  permission: string;
  setting: string;
}

export const policies = {
  list(): Promise<PolicyRow[]> {
    return fetchAllPages<PolicyRow>('/api/policies/all');
  },

  async blocked(): Promise<BlockedEntry[]> {
    const body = await api.get<ListingEnvelope<BlockedEntry>>('/api/policies/blocked');
    return body.result;
  },

  async update(body: PolicyUpdate): Promise<void> {
    await api.post('/api/policies/-1', body); // 204 → empty body
  },
};
