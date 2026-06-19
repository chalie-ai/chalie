/**
 * Providers API — endpoints derived from frontend/brain/providers.js fetch calls.
 *
 * GET  /providers               → list all providers
 * GET  /providers/selected      → get the selected provider
 * PUT  /providers/selected      → select a provider { provider_id }
 * GET  /providers/catalog       → get the provider catalog
 * GET  /providers/vision        → get the vision provider
 * PUT  /providers/vision        → set vision provider { provider_id }
 * GET  /providers/delegate      → get the delegate provider
 * PUT  /providers/delegate      → set delegate provider { provider_id } (null = use main)
 * POST /providers               → create provider
 * PUT  /providers/:id           → update provider
 * DELETE /providers/:id         → delete provider
 * POST /providers/list-models   → list models for credentials
 * POST /providers/test          → test connection
 */
import { api } from '@chalie/shared';

export interface Provider {
  id: number;
  name: string;
  platform: string;
  model: string | null;
  host: string | null;
  supports_vision: boolean;
  decrypt_failed?: boolean;
  role?: string | null;
}

export interface CatalogEntry {
  id: string;
  name: string;
  platform: string;
  host: string;
  needs_key: boolean;
}

export interface ProviderInput {
  name: string;
  platform: string;
  model: string;
  host?: string;
  api_key?: string;
}

export interface TestInput extends ProviderInput {
  provider_id?: number;
}

export const providers = {
  list(): Promise<{ providers: Provider[] }> {
    return api.get('/providers');
  },

  getSelected(): Promise<{ provider: Provider | null }> {
    return api.get('/providers/selected');
  },

  setSelected(providerId: number): Promise<unknown> {
    return api.put('/providers/selected', { provider_id: providerId });
  },

  getCatalog(): Promise<{ catalog: CatalogEntry[] }> {
    return api.get('/providers/catalog');
  },

  getVision(): Promise<{ provider: Provider | null; source: string }> {
    return api.get('/providers/vision');
  },

  setVision(providerId: number): Promise<unknown> {
    return api.put('/providers/vision', { provider_id: providerId });
  },

  getDelegate(): Promise<{ provider: Provider | null; source: string }> {
    return api.get('/providers/delegate');
  },

  // provider_id null clears the explicit pin → subagent work falls back to the
  // main provider (no "disabled" state). Backend returns the resolved status.
  setDelegate(providerId: number | null): Promise<{ provider: Provider | null; source: string }> {
    return api.put('/providers/delegate', { provider_id: providerId });
  },

  create(input: ProviderInput): Promise<{ provider: Provider }> {
    return api.post('/providers', input);
  },

  update(id: number, input: Partial<ProviderInput>): Promise<{ provider: Provider }> {
    return api.put(`/providers/${id}`, input);
  },

  delete(id: number): Promise<unknown> {
    return api.del(`/providers/${id}`);
  },

  listModels(body: { platform: string; host?: string; api_key?: string }): Promise<{ models: (string | { id: string })[] }> {
    return api.post('/providers/list-models', body);
  },

  test(body: TestInput): Promise<{ success: boolean; message?: string; error?: string }> {
    return api.post('/providers/test', body);
  },
};
