/**
 * Capabilities API — endpoints derived from frontend/brain/capabilities.js.
 *
 * GET  /api/capabilities                → { capabilities: Capability[] }
 * GET  /api/capabilities/:id            → flat Capability (id, name, version, connected, last_sync_at, config)
 * POST /api/capabilities/:id/setup      → configure a capability
 * POST /api/capabilities/:id/disconnect → disconnect a capability
 */
import { api } from '@chalie/shared';

export interface CapabilityFieldOption {
  value: string;
  label: string;
}

export interface CapabilityField {
  name: string;
  label: string;
  type: string; // 'checkbox' | 'select' | 'textarea' | 'text' | 'password'
  placeholder?: string;
  default?: unknown;
  required?: boolean;
  options?: CapabilityFieldOption[];
  show_when?: Record<string, unknown>;
}

export interface Capability {
  id: string;
  name: string;
  version?: string | null;
  connected?: boolean;
  status?: string | null;
  platform?: string | null;
  last_sync_at?: string | null;
  fields?: CapabilityField[];
  config?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export const capabilities = {
  list(): Promise<{ capabilities: Capability[] }> {
    return api.get('/api/capabilities');
  },

  // Backend returns a FLAT object (no { capability } wrapper): { id, name, version, connected, last_sync_at, config }.
  get(id: string): Promise<Capability> {
    return api.get(`/api/capabilities/${id}`);
  },

  setup(id: string, body: Record<string, unknown>): Promise<unknown> {
    return api.post(`/api/capabilities/${id}/setup`, body);
  },

  disconnect(id: string): Promise<unknown> {
    return api.post(`/api/capabilities/${id}/disconnect`, {});
  },
};
