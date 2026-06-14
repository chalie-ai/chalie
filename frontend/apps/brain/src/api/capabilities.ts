/**
 * Capabilities API — endpoints derived from frontend/brain/capabilities.js.
 *
 * GET  /api/capabilities              → { capabilities: Capability[] }
 * GET  /api/capabilities/:id          → { capability: Capability }
 * POST /api/capabilities/:id/setup    → configure a capability
 * POST /api/capabilities/:id/disconnect → disconnect a capability
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface Capability {
  id: string;
  name: string;
  enabled?: boolean;
  connected?: boolean;
  [key: string]: unknown;
}

export const capabilities = {
  list(): Promise<{ capabilities: Capability[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/capabilities'));
  },

  get(id: string): Promise<{ capability: Capability }> {
    const api = useApiClient();
    return withAuth(() => api.get(`/api/capabilities/${id}`));
  },

  setup(id: string, body: Record<string, unknown>): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.post(`/api/capabilities/${id}/setup`, body));
  },

  disconnect(id: string): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.post(`/api/capabilities/${id}/disconnect`, {}));
  },
};
