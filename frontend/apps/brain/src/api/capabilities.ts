/**
 * Capabilities API — thin wrapper over the Endpoint-contract routes in
 * backend/api/endpoints/capabilities.py + backend/api/actions/capabilities/*.
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 *
 *   GET  /api/capabilities/all                paginated listing (fetchAllPages).
 *   GET  /api/capabilities/<id>                → result DTO (detail, config non-secret only).
 *   POST /api/capabilities/setup/<id>          configure a capability → 204 empty body.
 *   POST /api/capabilities/disconnect/<id>     disconnect a capability → 204 empty body.
 */
import { api } from '@chalie/shared';
import { fetchAllPages } from './paginate';

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

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const capabilities = {
  list(): Promise<Capability[]> {
    return fetchAllPages<Capability>('/api/capabilities/all');
  },

  async get(id: string): Promise<Capability> {
    const res = await api.get<SingleEnvelope<Capability>>(`/api/capabilities/${id}`);
    return res.result;
  },

  setup(id: string, body: Record<string, unknown>): Promise<unknown> {
    return api.post(`/api/capabilities/setup/${id}`, body);
  },

  disconnect(id: string): Promise<unknown> {
    return api.post(`/api/capabilities/disconnect/${id}`, {});
  },
};
