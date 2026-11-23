/**
 * Capabilities API — thin wrapper over the Endpoint-contract routes in
 * backend/api/endpoints/capabilities.py + backend/api/actions/capabilities/*.
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 *
 *   GET  /api/capabilities/all                paginated listing (fetchAllPages).
 *   GET  /api/capabilities/<id>                → result DTO (detail, config non-secret only).
 *   GET  /api/capabilities/status/<id>         → result CapabilityStatus (pending while setup is
 *                                                still being confirmed in the background).
 *   POST /api/capabilities/setup/<id>          configure a capability → 204 empty body, also
 *                                                while its setup is still being confirmed;
 *                                                422 { error, reason? } when its settings fail.
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
  /** Each named field's value, or any of a list of values, that shows this field. */
  show_when?: Record<string, unknown>;
  /** Plain-text steps for filling this field in, then any notes that follow them. */
  guide?: { steps: string[]; notes?: string[] };
}

export interface Capability {
  id: string;
  name: string;
  version?: string | null;
  description?: string | null;
  available?: boolean;
  unavailable_reason?: string | null;
  connected?: boolean;
  status_note?: string | null;
  setup_code?: string | null;
  status?: string | null;
  platform?: string | null;
  last_sync_at?: string | null;
  fields?: CapabilityField[];
  config?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface CapabilityStatus {
  id: string;
  available: boolean;
  unavailable_reason?: string | null;
  connected: boolean;
  pending: boolean;
  status_note?: string | null;
  setup_code?: string | null;
  last_sync_at?: string | null;
}

/** Why a capability's settings failed — the 422 body's plain `reason`, worded by the Brain. */
export type SetupFailureReason =
  | 'credentials_rejected'
  | 'certificate_untrusted'
  | 'unreachable'
  | 'authenticator_key_missing'
  | 'authenticator_key_invalid'
  | 'authenticator_code_refused'
  | 'too_many_attempts'
  | 'invalid_configuration';

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

  async status(id: string): Promise<CapabilityStatus> {
    const res = await api.get<SingleEnvelope<CapabilityStatus>>(`/api/capabilities/status/${id}`);
    return res.result;
  },

  setup(id: string, body: Record<string, unknown>): Promise<unknown> {
    return api.post(`/api/capabilities/setup/${id}`, body);
  },

  disconnect(id: string): Promise<unknown> {
    return api.post(`/api/capabilities/disconnect/${id}`, {});
  },
};
