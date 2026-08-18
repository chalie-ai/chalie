/**
 * Providers API — thin wrapper over the Endpoint-contract routes in
 * backend/api/endpoints/providers.py + backend/api/actions/providers/*.
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 *
 *   GET    /api/providers/all           paginated listing (fetchAllPages).
 *   POST   /api/providers/-1            create provider → 201, result DTO.
 *   POST   /api/providers/<id>          update provider → result DTO.
 *   DELETE /api/providers/<id>          → 204 empty body.
 *   GET    /api/providers/api-key/<id>  stored API key, decrypted → result DTO.
 *   GET    /api/providers/catalog       curated presets (fetchAllPages).
 *   POST   /api/providers/list-models   list models for credentials → result DTO.
 *   POST   /api/providers/test          test connection → result DTO.
 *   GET    /api/providers/selected      selected (main) provider role → result DTO.
 *   POST   /api/providers/selected      pin the main provider → result DTO.
 *   GET    /api/providers/vision        vision provider role → result DTO.
 *   POST   /api/providers/vision        pin/clear the vision provider → result DTO.
 *   GET    /api/providers/delegate      delegate provider role → result DTO.
 *   POST   /api/providers/delegate      pin/clear the delegate provider → result DTO.
 *
 * ``selected`` is reshaped to the { provider, source } role envelope shared
 * with vision/delegate (the legacy route returned a bare Provider|null).
 */
import { api } from '@chalie/shared';
import { fetchAllPages } from './paginate';

export interface Provider {
  id: number;
  name: string;
  platform: string;
  model: string | null;
  host: string | null;
  supports_vision: boolean;
  decrypt_failed?: boolean;
  role?: string | null;
  context_window: number | null;
}

export interface CatalogEntry {
  id: string;
  name: string;
  platform: string;
  host: string;
  needs_key: boolean;
  needs_host: boolean;
}

export interface ProviderRole {
  provider: Provider | null;
  source: string;
}

export interface ModelInfo {
  id: string;
  display_name: string | null;
}

export interface ProviderInput {
  name: string;
  platform: string;
  model: string;
  host?: string;
  api_key?: string;
  // null means "ask the client" — Gemini/Ollama query their API for the real
  // window; the backend clamps whatever is sent to its own ceiling.
  context_window?: number | null;
}

// The connectivity test never uses `name` (it probes platform+model+creds), and
// the backend DTO forbids unknown fields — so `name` is omitted, not just optional.
export interface TestInput extends Omit<ProviderInput, 'name'> {
  provider_id?: number;
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const providers = {
  list(): Promise<Provider[]> {
    return fetchAllPages<Provider>('/api/providers/all');
  },

  async getSelected(): Promise<ProviderRole> {
    const res = await api.get<SingleEnvelope<ProviderRole>>('/api/providers/selected');
    return res.result;
  },

  async setSelected(providerId: number): Promise<ProviderRole> {
    const res = await api.post<SingleEnvelope<ProviderRole>>('/api/providers/selected', {
      provider_id: providerId,
    });
    return res.result;
  },

  getCatalog(): Promise<CatalogEntry[]> {
    return fetchAllPages<CatalogEntry>('/api/providers/catalog');
  },

  async getVision(): Promise<ProviderRole> {
    const res = await api.get<SingleEnvelope<ProviderRole>>('/api/providers/vision');
    return res.result;
  },

  // provider_id null clears the explicit pin → vision falls back to the main
  // provider when it can see. Backend returns the resolved status.
  async setVision(providerId: number | null): Promise<ProviderRole> {
    const res = await api.post<SingleEnvelope<ProviderRole>>('/api/providers/vision', {
      provider_id: providerId,
    });
    return res.result;
  },

  async getDelegate(): Promise<ProviderRole> {
    const res = await api.get<SingleEnvelope<ProviderRole>>('/api/providers/delegate');
    return res.result;
  },

  // provider_id null clears the explicit pin → subagent work falls back to the
  // main provider (no "disabled" state). Backend returns the resolved status.
  async setDelegate(providerId: number | null): Promise<ProviderRole> {
    const res = await api.post<SingleEnvelope<ProviderRole>>('/api/providers/delegate', {
      provider_id: providerId,
    });
    return res.result;
  },

  async create(input: ProviderInput): Promise<Provider> {
    const res = await api.post<SingleEnvelope<Provider>>('/api/providers/-1', input);
    return res.result;
  },

  async update(id: number, input: Partial<ProviderInput>): Promise<Provider> {
    const res = await api.post<SingleEnvelope<Provider>>(`/api/providers/${id}`, input);
    return res.result;
  },

  delete(id: number): Promise<unknown> {
    return api.del(`/api/providers/${id}`);
  },

  // Only ever called from the edit form's "Change API key" button — the key is
  // pulled on demand so the browser never holds it just for having the page open.
  async getApiKey(id: number): Promise<string> {
    const res = await api.get<SingleEnvelope<{ id: number; api_key: string | null }>>(
      `/api/providers/api-key/${id}`,
    );
    return res.result.api_key ?? '';
  },

  // provider_id lets the backend fill host/api_key from the stored row, so the
  // model list refreshes on an edit whose key was never revealed.
  async listModels(body: {
    platform: string;
    host?: string;
    api_key?: string;
    provider_id?: number;
  }): Promise<{ models: ModelInfo[]; error?: string | null }> {
    const res = await api.post<SingleEnvelope<{ models: ModelInfo[]; error?: string | null }>>(
      '/api/providers/list-models',
      body,
    );
    return res.result;
  },

  async test(
    body: TestInput,
  ): Promise<{ success: boolean; message?: string; error?: string; hint?: string }> {
    const res = await api.post<
      SingleEnvelope<{ success: boolean; message?: string; error?: string; hint?: string }>
    >('/api/providers/test', body);
    return res.result;
  },
};
