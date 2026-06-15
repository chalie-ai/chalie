/**
 * Policies API — endpoints derived from frontend/brain/policies.js.
 *
 * GET /api/policies              → { policies: PolicyRow[] }
 * GET /api/policies/blocked      → { entries: BlockedEntry[] }
 * PUT /api/policies              → update one policy { channel, permission, setting }
 */
import { api } from '@chalie/shared';

export interface PolicyRow {
  channel: string;
  permission: string;
  setting: string;
  label?: string;
  group?: string;
}

export interface BlockedEntry {
  action_id?: string | null;
  context?: string | null;
  created_at?: string | null;
  [key: string]: unknown;
}

export interface PolicyUpdate {
  channel: string;
  permission: string;
  setting: string;
}

export const policies = {
  list(): Promise<{ policies: PolicyRow[] }> {
    return api.get('/api/policies');
  },

  blocked(): Promise<{ entries: BlockedEntry[] }> {
    return api.get('/api/policies/blocked');
  },

  update(body: PolicyUpdate): Promise<unknown> {
    return api.put('/api/policies', body);
  },
};
