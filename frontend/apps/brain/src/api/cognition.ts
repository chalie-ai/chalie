/**
 * Cognition API — endpoints derived from frontend/brain/cognition.js fetch calls.
 *
 * GET  /system/observability/records?source=&limit=&offset=&q=  → memory records
 * GET  /system/observability/tools                               → tool list
 * GET  /system/observability/world-state                         → world state
 * GET  /settings/personality                                     → personality tuple + voice
 * PUT  /settings/personality                                     → update personality
 * GET  /system/observability/errors                              → recent errors
 * GET  /system/observability/token-usage?window=                 → token usage
 * GET  /system/observability/compaction                          → compaction summary
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface MemoryRecord {
  created: string | null;
  last_accessed: string | null;
  location?: string | null;
  key?: string | null;
  value: string | null;
}

export interface MemoryResponse {
  rows: MemoryRecord[];
  has_more: boolean;
  generated_at: string | null;
}

export interface Tool {
  name: string;
  description?: string | null;
  [key: string]: unknown;
}

export interface WorldState {
  [key: string]: unknown;
}

export interface Personality {
  tuple: [number, number, number, number, number];
  voice: string;
}

export interface ErrorEntry {
  [key: string]: unknown;
}

export interface UsageResponse {
  [key: string]: unknown;
}

export interface CompactionEntry {
  [key: string]: unknown;
}

export const cognition = {
  memory(params: {
    source?: string;
    limit?: number;
    offset?: number;
    q?: string;
  }): Promise<MemoryResponse> {
    const api = useApiClient();
    const p = new URLSearchParams();
    if (params.source) p.set('source', params.source);
    if (params.limit != null) p.set('limit', String(params.limit));
    if (params.offset != null) p.set('offset', String(params.offset));
    if (params.q) p.set('q', params.q);
    return withAuth(() => api.get(`/system/observability/records?${p.toString()}`));
  },

  tools(): Promise<{ tools: Tool[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/system/observability/tools'));
  },

  worldState(): Promise<WorldState> {
    const api = useApiClient();
    return withAuth(() => api.get('/system/observability/world-state'));
  },

  personality(): Promise<Personality> {
    const api = useApiClient();
    return withAuth(() => api.get('/settings/personality'));
  },

  setPersonality(data: Partial<Personality>): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put('/settings/personality', data));
  },

  errors(): Promise<{ errors: ErrorEntry[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/system/observability/errors'));
  },

  tokenUsage(window: string = 'day'): Promise<UsageResponse> {
    const api = useApiClient();
    return withAuth(() => api.get(`/system/observability/token-usage?window=${encodeURIComponent(window)}`));
  },

  compaction(): Promise<{ compaction: CompactionEntry | null }> {
    const api = useApiClient();
    return withAuth(() => api.get('/system/observability/compaction'));
  },
};
