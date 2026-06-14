/**
 * Policies API — endpoints derived from frontend/brain/policies.js.
 *
 * GET /api/policies              → { policies: PolicyRow[] }
 * GET /api/policies/blocked      → { entries: BlockedEntry[] }
 * PUT /api/policies              → update one policy { channel, permission, setting }
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface PolicyRow {
  channel: string;
  permission: string;
  setting: string;
}

export interface BlockedEntry {
  [key: string]: unknown;
}

export interface PolicyUpdate {
  channel: string;
  permission: string;
  setting: string;
}

export const policies = {
  list(): Promise<{ policies: PolicyRow[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/policies'));
  },

  blocked(): Promise<{ entries: BlockedEntry[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/policies/blocked'));
  },

  update(body: PolicyUpdate): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put('/api/policies', body));
  },
};
